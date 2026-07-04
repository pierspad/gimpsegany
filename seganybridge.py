"""
Script to generate Meta Segment Anything masks.

Adapted from:
https://github.com/facebookresearch/segment-anything-2
https://github.com/facebookresearch/segment-anything

Author: Shrinivas Kulkarni

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import contextlib
import os
import sys
import threading
import time

# These BLAS-level thread-pool env vars only take effect if set BEFORE
# numpy/torch/cv2 first touch the underlying OpenBLAS/MKL library, so this
# has to happen here, ahead of those imports. We only set a value the user
# hasn't already pinned themselves (e.g. via a shell profile), and we cap
# it to the actual core count so we don't oversubscribe on shared/CI boxes.
_CPU_COUNT = os.cpu_count() or 1
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_var, str(_CPU_COUNT))

import torch
import numpy as np
import cv2
from PIL import Image

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# SAM1 imports
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator as SamAutomaticMaskGenerator_SAM1,
    SamPredictor,
)

# SAM3 is intentionally NOT imported eagerly like SAM1/SAM2 above. Unlike
# those two (which LazyGimp always installs together, cheaply, so an eager
# import never fails), the `sam3` package is a heavy, separate opt-in
# dependency: Python 3.12+, a gated Hugging Face checkpoint the user must be
# individually approved for, and typically a real GPU. Forcing it on every
# SAM1/SAM2 invocation would break the common case for an uncommon one, so
# it's imported lazily inside SAM3Strategy with a clear error if missing.

# --- Progress reporting ----------------------------------------------------
#
# GIMP invokes this script as a subprocess and streams its stdout back to the
# user, line by line, as it arrives (see seganyplugin.py). Every long-running
# step below MUST print *something* every few seconds, otherwise the plug-in
# has nothing to show and the whole thing looks hung even though it is
# working perfectly fine — this was the single biggest cause of "GIMP froze,
# nothing ever happens": a multi-minute CPU-bound call with zero stdout.


def stage(name):
    print(f"[stage] {name}", flush=True)


@contextlib.contextmanager
def heartbeat(label, interval=3.0):
    """Print a progress line every `interval` seconds while a long call runs.

    Runs in a daemon thread so it can report progress even while the main
    thread is stuck inside a native (C++/torch) call that never returns to
    the Python interpreter until it's done — those calls still release the
    GIL for the bulk of their work, so the timer thread keeps ticking.
    """
    stop = threading.Event()
    t0 = time.time()

    def _tick():
        while not stop.wait(interval):
            print(f"[progress] {label}: {time.time() - t0:.0f}s elapsed", flush=True)

    t = threading.Thread(target=_tick, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1)
        print(f"[progress] {label}: done in {time.time() - t0:.1f}s", flush=True)


# --- Device selection --------------------------------------------------------
#
# Deliberately generic: NVIDIA (CUDA), AMD (ROCm builds of torch report
# through the same torch.cuda.* API), Apple Silicon (MPS) and a CPU fallback
# that is tuned to actually use every core — by default torch sometimes
# leaves threads on the table in containerized/virtualized environments.


def pick_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"CUDA/ROCm GPU ({name})"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps"), "Apple Silicon (MPS)"
    torch.set_num_threads(_CPU_COUNT)
    cv2.setNumThreads(_CPU_COUNT)
    return torch.device("cpu"), f"CPU ({_CPU_COUNT} threads)"


def points_per_batch_for(device):
    """How many grid points to run through the mask decoder in one forward
    pass. On GPU, bigger batches keep thousands of cores fed at once — the
    library default (64) already undersells a modern GPU. On CPU there's no
    such thing as an idle "core" to feed this way: torch dispatches each
    matmul to every thread via set_num_threads() above regardless of batch
    size, so a smaller batch just means lower peak memory with no speed
    penalty, which matters more on typically memory-constrained CPU boxes.
    """
    return 128 if device.type in ("cuda", "mps") else 32


# --- Utility Functions ---


def packBoolArray(filepath, arr):
    packed_data = bytearray()
    num_rows = len(arr)
    num_cols = len(arr[0])
    packed_data.extend(
        [num_rows >> 24, (num_rows >> 16) & 255, (num_rows >> 8) & 255, num_rows & 255]
    )
    packed_data.extend(
        [num_cols >> 24, (num_cols >> 16) & 255, (num_cols >> 8) & 255, num_cols & 255]
    )
    current_byte = 0
    bit_position = 0
    for row in arr:
        for boolean_value in row:
            if boolean_value:
                current_byte |= 1 << bit_position
            bit_position += 1
            if bit_position == 8:
                packed_data.append(current_byte)
                current_byte = 0
                bit_position = 0
    if bit_position > 0:
        packed_data.append(current_byte)
    with open(filepath, "wb") as f:
        f.write(packed_data)
    return packed_data


def saveMask(filepath, maskArr, formatBinary):
    if formatBinary:
        packBoolArray(filepath, maskArr)
    else:
        with open(filepath, "w") as f:
            for row in maskArr:
                f.write("".join(str(int(val)) for val in row) + "\n")


def saveMasks(masks, saveFileNoExt, formatBinary):
    for i, mask in enumerate(masks):
        filepath = saveFileNoExt + str(i) + ".seg"
        arr = [[val for val in row] for row in mask]
        saveMask(filepath, arr, formatBinary)


def resizeMaskToOriginal(mask, targetShape):
    """Nearest-neighbour resize a boolean mask back to (h, w) = targetShape."""
    h, w = targetShape
    resized = cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


# --- Strategy Pattern Implementation ---
#
# Segmentation "resolution" for Auto mode is dominated by points_per_side:
# the automatic mask generator runs one decoder pass PER GRID POINT (32x32 =
# 1024 decoder calls at the old hardcoded default), so this — not the input
# image's pixel size — is what makes Auto mode take seconds vs. tens of
# minutes on a CPU. Both SAM1 and SAM2 share the same generator API, so the
# same knobs apply to both; previously SAM1 silently ignored the "Resolution"
# dropdown entirely and always ran the heaviest possible grid.
POINTS_PER_SIDE_BY_RES = {"Low": 8, "Medium": 16, "High": 32}
DEFAULT_POINTS_PER_SIDE = POINTS_PER_SIDE_BY_RES["Medium"]


class SegmentationStrategy:
    def get_model_type_from_filename(self, model_filename):
        raise NotImplementedError

    def load_model(self, checkPtFilePath, modelType, device):
        raise NotImplementedError

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        raise NotImplementedError

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
        raise NotImplementedError

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
    ):
        raise NotImplementedError

    def run_test(self, sam):
        raise NotImplementedError

    def cleanup(self):
        pass


class SAM1Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam_vit_h_4b8939": "vit_h",
        "sam_vit_l_0b3195": "vit_l",
        "sam_vit_b_01ec64": "vit_b",
    }

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM1 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM1 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names: {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def load_model(self, checkPtFilePath, modelType, device):
        try:
            sam = sam_model_registry[modelType](checkpoint=checkPtFilePath)
            sam.to(device=device)
            print(f"SAM1 Model loaded successfully on {device}!")
            return sam
        except Exception as e:
            print(f"Error loading SAM1 model: {e}")
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        points_per_side = POINTS_PER_SIDE_BY_RES.get(
            kwargs.get("segRes"), DEFAULT_POINTS_PER_SIDE
        )
        mask_generator = SamAutomaticMaskGenerator_SAM1(
            sam,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch_for(kwargs.get("device") or torch.device("cpu")),
            crop_n_layers=kwargs.get("cropNLayers", 0),
            min_mask_region_area=kwargs.get("minMaskArea", 0),
        )
        with heartbeat(f"segmenting (grid {points_per_side}x{points_per_side})"):
            masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )


class SAM2Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam2_hiera_large": "sam2_hiera_large",
        "sam2_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2_hiera_small": "sam2_hiera_small",
        "sam2_hiera_tiny": "sam2_hiera_tiny",
        "sam2.1_hiera_large": "sam2_hiera_large",
        "sam2.1_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2.1_hiera_small": "sam2_hiera_small",
        "sam2.1_hiera_tiny": "sam2_hiera_tiny",
    }

    def __init__(self):
        self._temp_pth_path = None

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM2 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM2 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names (or their .safetensors/.pt equivalents): {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def _convert_safetensors_to_pth(self, safetensors_path, pth_path):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
            checkpoint = {"model": state_dict}
            torch.save(checkpoint, pth_path)
            return True
        except Exception as e:
            print(f"Error converting safetensors to pth: {e}")
            return False

    def load_model(self, checkPtFilePath, modelType, device):
        model_configs = {
            "sam2_hiera_tiny": "sam2_hiera_t.yaml",
            "sam2_hiera_small": "sam2_hiera_s.yaml",
            "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
            "sam2_hiera_large": "sam2_hiera_l.yaml",
        }
        config_file = model_configs.get(modelType, "sam2_hiera_l.yaml")
        actual_checkpoint_path = checkPtFilePath
        if checkPtFilePath.endswith(".safetensors"):
            print("Converting safetensors to pth format...")
            self._temp_pth_path = checkPtFilePath.replace(".safetensors", "_temp.pth")
            if self._convert_safetensors_to_pth(checkPtFilePath, self._temp_pth_path):
                actual_checkpoint_path = self._temp_pth_path
                print(f"Converted to: {self._temp_pth_path}")
            else:
                print("Failed to convert safetensors file")
                return None
        try:
            sam = build_sam2(
                config_file, actual_checkpoint_path, device=str(device)
            )
            print(f"SAM2 Model loaded successfully on {device}!")
            return sam
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            self.cleanup()
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        points_per_side = POINTS_PER_SIDE_BY_RES.get(
            kwargs.get("segRes"), DEFAULT_POINTS_PER_SIDE
        )
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch_for(kwargs.get("device") or torch.device("cpu")),
            crop_n_layers=kwargs.get("cropNLayers", 0),
            min_mask_region_area=kwargs.get("minMaskArea", 0),
        )
        with heartbeat(f"segmenting (grid {points_per_side}x{points_per_side})"):
            masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )

    def cleanup(self):
        if self._temp_pth_path and os.path.exists(self._temp_pth_path):
            os.remove(self._temp_pth_path)
            print(f"Removed temporary file: {self._temp_pth_path}")


class SAM3Strategy(SegmentationStrategy):
    """SAM3 ("Segment Anything with Concepts", facebookresearch/sam3) is a
    fundamentally different tool from SAM1/2: instead of points/boxes on ONE
    object, you give it a short noun phrase (e.g. "car") and it finds every
    matching instance in the image. There is no "Auto" grid-search mode and
    no per-model size variants — Meta ships a single checkpoint.

    The checkpoint is not a lone .pth file but a gated Hugging Face snapshot
    (several files). `build_sam3_image_model()` takes no path argument at
    all in the upstream API — it resolves the model from the ambient HF
    cache/auth, which is exactly why our installer's job for SAM3 is to run
    `huggingface_hub` login + snapshot_download with the user's access
    token, not to hand this bridge a checkpoint path to open directly.
    """

    def get_model_type_from_filename(self, model_filename):
        return "sam3"

    def load_model(self, checkPtFilePath, modelType, device):
        try:
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:
            print(
                "Error: the 'sam3' package isn't installed in this "
                f"interpreter ({e}). Run the gimpsegany installer's SAM3 "
                "setup step (pip install -e . from a clone of "
                "https://github.com/facebookresearch/sam3), then try again."
            )
            return None
        try:
            model = build_sam3_image_model()
            model.to(device=device)
            print(f"SAM3 model loaded successfully on {device}!")
            return model
        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            print(
                "This is most often either (1) no Hugging Face access token "
                "configured — run the installer's 'Authenticate with "
                "Hugging Face' step after being approved for "
                "facebook/sam3.1 — or (2) the checkpoint hasn't been "
                "downloaded yet."
            )
            return None

    def _processor(self, sam):
        from sam3.model.sam3_image_processor import Sam3Processor

        return Sam3Processor(sam)

    def segment_text(self, sam, pilImage, maskType, textPrompt, saveFileNoExt, formatBinary):
        processor = self._processor(sam)
        inference_state = processor.set_image(pilImage)
        output = processor.set_text_prompt(state=inference_state, prompt=textPrompt)
        masks = output["masks"]
        scores = output.get("scores")
        try:
            masks = masks.detach().cpu().numpy()
        except AttributeError:
            masks = np.asarray(masks)
        masks = masks.astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if maskType == "Single" and len(masks) > 1:
            best = 0
            if scores is not None:
                try:
                    best = int(np.argmax(np.asarray(scores)))
                except Exception:
                    best = 0
            masks = masks[best : best + 1]
        print(f"SAM3 found {len(masks)} instance(s) of '{textPrompt}'")
        saveMasks(list(masks), saveFileNoExt, formatBinary)

    # Box/Selection prompting is documented as supported by SAM3's processor
    # (see the SAM1/2-parity notebooks in the upstream repo), but the exact
    # method name isn't confirmed at the time of writing — rather than guess
    # and risk a confusing failure, these clearly signpost what to do.
    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
        raise NotImplementedError(
            "SAM3 'Box' mode isn't wired up yet in this bridge — use "
            "Segmentation Type 'Text' with a short phrase instead "
            "(e.g. \"car\"). See github.com/facebookresearch/sam3 examples "
            "for the box-prompt API if you want to add it."
        )

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
    ):
        raise NotImplementedError(
            "SAM3 'Selection' mode isn't wired up yet in this bridge — use "
            "Segmentation Type 'Text' with a short phrase instead."
        )

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        raise NotImplementedError(
            "SAM3 has no grid-search 'Auto' mode — use Segmentation Type "
            "'Text' with a short phrase describing what to segment."
        )

    def run_test(self, sam):
        img = Image.new("RGB", (64, 64), (255, 255, 255))
        processor = self._processor(sam)
        state = processor.set_image(img)
        processor.set_text_prompt(state=state, prompt="square")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python seganybridge.py <model_type|auto> <checkpoint_path> [options]"
        )
        return

    t_start = time.time()
    modelType = sys.argv[1]
    checkPtFilePath = sys.argv[2]
    model_filename = os.path.basename(checkPtFilePath.rstrip(os.sep))

    if model_filename.lower() == "sam3" or modelType.lower() == "sam3":
        strategy = SAM3Strategy()
    elif model_filename.lower().startswith("sam_"):
        strategy = SAM1Strategy()
    elif model_filename.lower().startswith("sam2"):
        strategy = SAM2Strategy()
    else:
        print(
            f"Error: Could not determine model family from filename: {model_filename}"
        )
        print(
            "Filename must start with 'sam_' for SAM1, 'sam2' for SAM2, or "
            "be a folder named 'sam3' for SAM3."
        )
        return

    if modelType.lower() == "auto":
        modelType = strategy.get_model_type_from_filename(model_filename)
        if not modelType:
            return

    if not os.path.exists(checkPtFilePath):
        print(f"Error: Checkpoint file not found: {checkPtFilePath}")
        return

    device, device_desc = pick_device()
    print(f"Using device: {device_desc}", flush=True)

    stage("loading_model")
    with heartbeat("loading model"):
        sam = strategy.load_model(checkPtFilePath, modelType, device)
    if sam is None:
        return

    if len(sys.argv) == 3:
        strategy.run_test(sam)
        print("Success!!")
        strategy.cleanup()
        return

    ipFile = sys.argv[3]
    segType = sys.argv[4]
    maskType = sys.argv[5]
    saveFileNoExt = sys.argv[6]
    formatBinary = sys.argv[7] == "True" if len(sys.argv) > 7 else True

    stage("reading_image")
    cvImage = cv2.imread(ipFile)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)

    try:
        if segType == "Auto":
            auto_kwargs = {"device": device}
            if len(sys.argv) > 8:
                auto_kwargs["segRes"] = sys.argv[8]
            if len(sys.argv) > 9:
                auto_kwargs["cropNLayers"] = int(sys.argv[9])
            if len(sys.argv) > 10:
                auto_kwargs["minMaskArea"] = int(sys.argv[10])
            maxAutoDim = int(sys.argv[11]) if len(sys.argv) > 11 else 0

            originalShape = cvImage.shape[:2]  # (h, w)
            workImage = cvImage
            if maxAutoDim > 0 and max(originalShape) > maxAutoDim:
                scale = maxAutoDim / max(originalShape)
                newSize = (
                    max(1, int(round(originalShape[1] * scale))),
                    max(1, int(round(originalShape[0] * scale))),
                )
                print(
                    f"Downscaling {originalShape[1]}x{originalShape[0]} -> "
                    f"{newSize[0]}x{newSize[1]} for a faster Auto pass "
                    "(masks are upscaled back before saving)",
                    flush=True,
                )
                workImage = cv2.resize(cvImage, newSize, interpolation=cv2.INTER_AREA)

            stage("segmenting")
            if workImage is cvImage:
                strategy.segment_auto(
                    sam, workImage, saveFileNoExt, formatBinary, **auto_kwargs
                )
            else:
                # Segment at the reduced resolution, then upscale each mask
                # back to the original image size before persisting it.
                import tempfile as _tempfile

                tmpPrefix = saveFileNoExt + "__lowres__"
                strategy.segment_auto(
                    sam, workImage, tmpPrefix, formatBinary=False, **auto_kwargs
                )
                idx = 0
                while True:
                    lowResPath = tmpPrefix + str(idx) + ".seg"
                    if not os.path.exists(lowResPath):
                        break
                    with open(lowResPath, "r") as f:
                        rows = [
                            [c == "1" for c in line.rstrip("\n")]
                            for line in f.readlines()
                        ]
                    os.remove(lowResPath)
                    lowResMask = np.array(rows, dtype=bool)
                    fullResMask = resizeMaskToOriginal(lowResMask, originalShape)
                    saveMask(
                        saveFileNoExt + str(idx) + ".seg", fullResMask, formatBinary
                    )
                    idx += 1
        elif segType in {"Selection", "Box-Selection"}:
            selFile = sys.argv[8]
            boxCos = (
                [float(val.strip()) for val in sys.argv[9].split(",")]
                if len(sys.argv) > 9
                else None
            )
            stage("segmenting")
            with heartbeat("segmenting selection"):
                strategy.segment_sel(
                    sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
                )
        elif segType == "Box":
            boxCos = [float(val.strip()) for val in sys.argv[9].split(",")]
            stage("segmenting")
            with heartbeat("segmenting box"):
                strategy.segment_box(
                    sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary
                )
        elif segType == "Text":
            textPrompt = sys.argv[8] if len(sys.argv) > 8 else ""
            if not textPrompt.strip():
                print('Error: "Text" segmentation type needs a non-empty prompt.')
                return
            stage("segmenting")
            with heartbeat(f'segmenting concept "{textPrompt}"'):
                strategy.segment_text(
                    sam,
                    Image.fromarray(cvImage),
                    maskType,
                    textPrompt,
                    saveFileNoExt,
                    formatBinary,
                )
        else:
            print(f"Unknown segmentation type: {segType}")
    finally:
        print(f"Done! (total {time.time() - t_start:.1f}s)", flush=True)
        strategy.cleanup()


if __name__ == "__main__":
    main()
