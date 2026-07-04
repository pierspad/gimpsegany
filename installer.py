#!/usr/bin/env python3
"""
gimpsegany installer — a small interactive GUI (Tkinter, no extra
dependencies to just launch it) that sets up the Segment Anything plug-in
for GIMP end to end, regardless of whether anything is already installed:

  1. Detects whether GIMP is installed at all, and where its plug-ins
     folder lives.
  2. Detects whether the plug-in files are already there.
  3. Detects which SAM checkpoints are already downloaded.
  4. Lets you install/update the plug-in files, (re)build the Python
     backend (a dedicated venv + PyTorch + SAM1/SAM2 packages), and
     install/remove individual models — including SAM3, which is gated on
     Hugging Face and needs a personal access token.

Run it with:  python3 installer.py

Every check is re-run fresh on startup and after each action, so this is
safe to run repeatedly — it never assumes a clean slate and never wipes
something that's already working.
"""

from __future__ import annotations

import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))

# Shared with seganyplugin.py's discover_lazygimp_backend() and with
# LazyGimp's lib/segany_backend.sh — whichever tool sets this directory up,
# the plug-in finds it automatically. Not actually LazyGimp-specific; the
# name is just kept for backward compatibility with existing installs.
BACKEND_DIR = os.path.expanduser("~/.local/share/lazygimp/segany")
VENV_DIR = os.path.join(BACKEND_DIR, "venv")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")

TORCH_INDEX_URLS = {
    "CPU (universal, smaller download)": "https://download.pytorch.org/whl/cpu",
    "NVIDIA CUDA 12.6": "https://download.pytorch.org/whl/cu126",
    "NVIDIA CUDA 12.8": "https://download.pytorch.org/whl/cu128",
    "AMD ROCm 6.2": "https://download.pytorch.org/whl/rocm6.2",
}

SAM1_PIP_SPEC = "git+https://github.com/facebookresearch/segment-anything.git"
SAM2_PIP_SPEC = "git+https://github.com/facebookresearch/segment-anything-2.git"
SAM3_REPO_URL = "https://github.com/facebookresearch/sam3.git"
SAM3_HF_PAGE = "https://huggingface.co/facebook/sam3.1"
SAM3_HF_REPO_ID = "facebook/sam3.1"


@dataclass
class ModelSpec:
    key: str
    family: str  # "SAM1", "SAM2", "SAM3"
    label: str
    size: str
    note: str
    filename: str | None = None  # None for SAM3 (a folder, not a file)
    url: str | None = None  # None for SAM3 (gated, downloaded via token)


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        "sam_vit_b", "SAM1", "vit_b", "375 MB", "lightest, fastest on CPU",
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    ModelSpec(
        "sam_vit_l", "SAM1", "vit_l", "1.2 GB", "balanced (recommended)",
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    ModelSpec(
        "sam_vit_h", "SAM1", "vit_h", "2.5 GB", "best quality, slow on CPU",
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
    ModelSpec(
        "sam2_hiera_tiny", "SAM2", "hiera_tiny", "150 MB", "",
        "sam2_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    ),
    ModelSpec(
        "sam2_hiera_small", "SAM2", "hiera_small", "180 MB", "",
        "sam2_hiera_small.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    ),
    ModelSpec(
        "sam2_hiera_base_plus", "SAM2", "hiera_base_plus", "320 MB", "",
        "sam2_hiera_base_plus.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    ),
    ModelSpec(
        "sam2_hiera_large", "SAM2", "hiera_large", "900 MB", "",
        "sam2_hiera_large.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    ),
    ModelSpec(
        "sam3", "SAM3", "sam3.1", "~3.4 GB", "gated — needs HF approval + a real GPU",
        None, None,
    ),
]


def model_path(spec: ModelSpec) -> str:
    if spec.family == "SAM3":
        return os.path.join(MODELS_DIR, "sam3")
    return os.path.join(MODELS_DIR, spec.filename)


def model_installed(spec: ModelSpec) -> bool:
    p = model_path(spec)
    if spec.family == "SAM3":
        return os.path.isdir(p) and len(os.listdir(p)) > 0
    return os.path.isfile(p)


# --------------------------------------------------------------------------
# Detection helpers — all read-only, safe to call anytime.
# --------------------------------------------------------------------------

def find_gimp_binary() -> str | None:
    return shutil.which("gimp") or shutil.which("gimp-3.0") or shutil.which("gimp-2.10")


def gimp_config_dirs() -> list[str]:
    """Every ~/.config/GIMP/X.Y directory that actually exists, newest first."""
    base = os.path.expanduser("~/.config/GIMP")
    if not os.path.isdir(base):
        return []
    dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]

    def version_key(path):
        name = os.path.basename(path)
        parts = []
        for chunk in name.split("."):
            try:
                parts.append(int(chunk))
            except ValueError:
                parts.append(0)
        return parts

    dirs.sort(key=version_key, reverse=True)
    return dirs


def find_plugins_dir() -> str | None:
    for d in gimp_config_dirs():
        candidate = os.path.join(d, "plug-ins")
        if os.path.isdir(candidate) or os.path.isdir(d):
            return candidate
    return None


def plugin_install_status(plugins_dir: str | None) -> tuple[bool, str | None]:
    if not plugins_dir:
        return False, None
    dest = os.path.join(plugins_dir, "seganyplugin", "seganyplugin.py")
    return os.path.isfile(dest), dest


def venv_status() -> bool:
    return os.path.isfile(VENV_PYTHON) and os.access(VENV_PYTHON, os.X_OK)


# --------------------------------------------------------------------------
# Background job runner — every long action (pip installs, downloads) runs
# in a thread and streams lines back to the GUI through a queue, so the
# installer never freezes. This mirrors exactly the lesson learned fixing
# the GIMP plug-in itself: never block the UI thread on a slow subprocess.
# --------------------------------------------------------------------------

class Job:
    def __init__(self, log_queue: "queue.Queue[str]"):
        self.log_queue = log_queue

    def log(self, msg: str):
        self.log_queue.put(msg)

    def run_cmd(self, cmd: list[str], **kw) -> int:
        self.log("$ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, **kw,
        )
        for line in iter(proc.stdout.readline, ""):
            if line:
                self.log(line.rstrip("\n"))
        proc.wait()
        return proc.returncode

    def download(self, url: str, dest: str, headers: dict | None = None):
        import urllib.request

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        req = urllib.request.Request(url, headers=headers or {})
        self.log(f"Downloading {url}")
        with urllib.request.urlopen(req) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            read = 0
            chunk = 1024 * 256
            last_pct = -1
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                read += len(buf)
                if total:
                    pct = int(read * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        self.log(f"  {pct}%  ({read // (1024*1024)} MB / {total // (1024*1024)} MB)")
                        last_pct = pct
        os.replace(part, dest)
        self.log(f"Saved to {dest}")


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class InstallerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("gimpsegany installer")
        root.geometry("880x680")

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.busy = False

        self._build_widgets()
        self.refresh_status()
        self.root.after(150, self._drain_log_queue)

    # ---- layout -----------------------------------------------------

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 6}

        status_frame = ttk.LabelFrame(self.root, text="Status")
        status_frame.pack(fill="x", **pad)
        self.status_labels = {}
        for key, text in [
            ("gimp", "GIMP:"),
            ("plugin", "Plug-in:"),
            ("backend", "Python backend:"),
        ]:
            row = ttk.Frame(status_frame)
            row.pack(fill="x", padx=8, pady=2)
            ttk.Label(row, text=text, width=16).pack(side="left")
            lbl = ttk.Label(row, text="checking…")
            lbl.pack(side="left")
            self.status_labels[key] = lbl

        actions = ttk.LabelFrame(self.root, text="Setup")
        actions.pack(fill="x", **pad)

        row1 = ttk.Frame(actions)
        row1.pack(fill="x", padx=8, pady=4)
        ttk.Button(row1, text="Install / Update Plug-in Files", command=self.on_install_plugin).pack(side="left")
        ttk.Button(row1, text="Choose GIMP plug-ins folder…", command=self.on_choose_plugins_dir).pack(side="left", padx=6)

        row2 = ttk.Frame(actions)
        row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="PyTorch build:").pack(side="left")
        self.torch_choice = tk.StringVar(value=list(TORCH_INDEX_URLS.keys())[0])
        ttk.Combobox(
            row2, textvariable=self.torch_choice, values=list(TORCH_INDEX_URLS.keys()),
            state="readonly", width=32,
        ).pack(side="left", padx=6)
        ttk.Button(row2, text="Create / Repair Python Backend (SAM1 + SAM2)", command=self.on_setup_backend).pack(side="left", padx=6)

        models_frame = ttk.LabelFrame(self.root, text="Models")
        models_frame.pack(fill="both", expand=False, **pad)

        columns = ("family", "label", "size", "note", "status")
        self.tree = ttk.Treeview(models_frame, columns=columns, show="headings", height=8, selectmode="browse")
        for col, width, heading in [
            ("family", 60, "Family"),
            ("label", 140, "Model"),
            ("size", 80, "Size"),
            ("note", 260, "Note"),
            ("status", 120, "Status"),
        ]:
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="x", padx=8, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self.on_model_selected)

        model_actions = ttk.Frame(models_frame)
        model_actions.pack(fill="x", padx=8, pady=4)
        self.install_btn = ttk.Button(model_actions, text="Download Selected", command=self.on_install_model)
        self.install_btn.pack(side="left")
        self.uninstall_btn = ttk.Button(model_actions, text="Remove Selected", command=self.on_uninstall_model)
        self.uninstall_btn.pack(side="left", padx=6)

        self.sam3_frame = ttk.LabelFrame(self.root, text="SAM3 — gated Hugging Face model")
        self.sam3_frame.pack(fill="x", **pad)
        info = (
            "SAM3 checkpoints require requesting access on Hugging Face and being approved "
            "before you can download them. This also needs Python 3.12+, a recent PyTorch, "
            "and a real GPU — it will not run well on CPU-only machines."
        )
        ttk.Label(self.sam3_frame, text=info, wraplength=820, justify="left").pack(anchor="w", padx=8, pady=(4, 8))
        row3 = ttk.Frame(self.sam3_frame)
        row3.pack(fill="x", padx=8, pady=4)
        ttk.Button(row3, text="1. Request access on Hugging Face", command=self.on_open_hf).pack(side="left")
        ttk.Button(row3, text="2. Set up the sam3 package (clone + pip install)", command=self.on_setup_sam3_package).pack(side="left", padx=6)
        row4 = ttk.Frame(self.sam3_frame)
        row4.pack(fill="x", padx=8, pady=4)
        ttk.Label(row4, text="3. Hugging Face token:").pack(side="left")
        self.hf_token = tk.StringVar()
        ttk.Entry(row4, textvariable=self.hf_token, show="*", width=48).pack(side="left", padx=6)
        ttk.Button(row4, text="Download SAM3 checkpoint", command=self.on_download_sam3).pack(side="left", padx=6)

        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(log_frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=12, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

        self._refresh_model_table()

    # ---- status -------------------------------------------------------

    def refresh_status(self):
        gimp_bin = find_gimp_binary()
        plugins_dir = find_plugins_dir()
        installed, dest = plugin_install_status(plugins_dir)

        self.plugins_dir = plugins_dir

        if gimp_bin:
            self.status_labels["gimp"].config(text=f"found ({gimp_bin})")
        elif gimp_config_dirs():
            self.status_labels["gimp"].config(text="config found, but 'gimp' not on PATH")
        else:
            self.status_labels["gimp"].config(text="not found — install GIMP first")

        if not plugins_dir:
            self.status_labels["plugin"].config(text="cannot locate a GIMP plug-ins folder")
        elif installed:
            self.status_labels["plugin"].config(text=f"installed ({dest})")
        else:
            self.status_labels["plugin"].config(text=f"not installed (target: {plugins_dir})")

        if venv_status():
            self.status_labels["backend"].config(text=f"ready ({VENV_DIR})")
        else:
            self.status_labels["backend"].config(text="not set up yet")

        self._refresh_model_table()

    def _refresh_model_table(self):
        self.tree.delete(*self.tree.get_children())
        for spec in MODEL_REGISTRY:
            status = "installed" if model_installed(spec) else "not installed"
            self.tree.insert(
                "", "end", iid=spec.key,
                values=(spec.family, spec.label, spec.size, spec.note, status),
            )

    def on_model_selected(self, _event=None):
        pass  # selection only matters when a button is clicked

    # ---- logging / busy state -----------------------------------------

    def _drain_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log_queue)

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for child in self.root.winfo_children():
            self._set_widget_state(child, state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            self.refresh_status()

    def _set_widget_state(self, widget, state):
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_state(child, state)

    def run_in_background(self, fn):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running — check the log.")
            return
        self._set_busy(True)

        def wrapper():
            job = Job(self.log_queue)
            try:
                fn(job)
            except Exception as e:
                job.log(f"ERROR: {e}")
            finally:
                self.log_queue.put("--- done ---")
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=wrapper, daemon=True).start()

    # ---- actions: plug-in files -----------------------------------------

    def on_choose_plugins_dir(self):
        chosen = filedialog.askdirectory(title="Select your GIMP plug-ins folder")
        if chosen:
            self.plugins_dir = chosen
            self.status_labels["plugin"].config(text=f"(manual) {chosen}")

    def on_install_plugin(self):
        plugins_dir = getattr(self, "plugins_dir", None) or find_plugins_dir()
        if not plugins_dir:
            messagebox.showerror(
                "No GIMP plug-ins folder",
                "Could not find a GIMP plug-ins folder automatically. Install "
                "GIMP first, run it once, or use 'Choose GIMP plug-ins folder…'.",
            )
            return

        def task(job: Job):
            src_plugin = os.path.join(HERE, "seganyplugin.py")
            src_bridge = os.path.join(HERE, "seganybridge.py")
            if not (os.path.isfile(src_plugin) and os.path.isfile(src_bridge)):
                job.log(f"ERROR: expected seganyplugin.py/seganybridge.py next to this "
                        f"installer ({HERE}) — did you move installer.py out of the repo?")
                return
            dest_dir = os.path.join(plugins_dir, "seganyplugin")
            job.log(f"Installing into {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_plugin, dest_dir)
            shutil.copy2(src_bridge, dest_dir)
            for f in (src_plugin, src_bridge):
                dest_file = os.path.join(dest_dir, os.path.basename(f))
                os.chmod(dest_file, 0o755)
            job.log("Plug-in files installed. Restart GIMP to load the change.")

        self.run_in_background(task)

    # ---- actions: python backend ----------------------------------------

    def on_setup_backend(self):
        torch_index = TORCH_INDEX_URLS[self.torch_choice.get()]

        def task(job: Job):
            os.makedirs(BACKEND_DIR, exist_ok=True)
            if not venv_status():
                job.log(f"Creating virtualenv at {VENV_DIR}")
                rc = job.run_cmd([sys.executable, "-m", "venv", VENV_DIR])
                if rc != 0:
                    job.log("Failed to create the virtualenv (is python3-venv installed?)")
                    return
            else:
                job.log(f"Reusing existing virtualenv at {VENV_DIR}")

            pip = os.path.join(VENV_DIR, "bin", "pip")
            job.run_cmd([pip, "install", "--upgrade", "pip"])
            job.log(f"Installing PyTorch from {torch_index}")
            job.run_cmd([pip, "install", "torch", "torchvision", "--index-url", torch_index])
            job.log("Installing image dependencies")
            job.run_cmd([pip, "install", "numpy", "pillow", "opencv-python-headless"])
            job.log("Installing SAM1 (segment-anything)")
            job.run_cmd([pip, "install", SAM1_PIP_SPEC])
            job.log("Installing SAM2 (segment-anything-2) — this can take a few minutes")
            rc = job.run_cmd([pip, "install", SAM2_PIP_SPEC])
            if rc != 0:
                job.log(
                    "SAM2 failed to build — SAM1 models will still work. This "
                    "usually means a C/C++ toolchain is missing; install one "
                    "and re-run this step."
                )
            job.log("Python backend ready.")

        self.run_in_background(task)

    # ---- actions: SAM1/SAM2 models ---------------------------------------

    def _selected_spec(self) -> ModelSpec | None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No model selected", "Select a model in the list first.")
            return None
        key = sel[0]
        return next((m for m in MODEL_REGISTRY if m.key == key), None)

    def on_install_model(self):
        spec = self._selected_spec()
        if spec is None:
            return
        if spec.family == "SAM3":
            messagebox.showinfo(
                "Use the SAM3 panel below",
                "SAM3 is gated — use the 'SAM3' section below (request access, "
                "then paste your token and click 'Download SAM3 checkpoint').",
            )
            return

        def task(job: Job):
            dest = model_path(spec)
            if os.path.isfile(dest):
                job.log(f"{spec.label} already downloaded at {dest}")
                return
            job.download(spec.url, dest)

        self.run_in_background(task)

    def on_uninstall_model(self):
        spec = self._selected_spec()
        if spec is None:
            return
        dest = model_path(spec)
        if not model_installed(spec):
            messagebox.showinfo("Not installed", f"{spec.label} is not currently installed.")
            return
        if not messagebox.askyesno("Remove model", f"Delete {dest}?"):
            return

        def task(job: Job):
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            else:
                os.remove(dest)
            job.log(f"Removed {dest}")

        self.run_in_background(task)

    # ---- actions: SAM3 ----------------------------------------------------

    def on_open_hf(self):
        webbrowser.open(SAM3_HF_PAGE)
        messagebox.showinfo(
            "Hugging Face access",
            "Log in to Hugging Face, request access on the model page, and "
            "wait for approval (usually fast, sometimes a day or two).\n\n"
            "Then create a personal access token at "
            "huggingface.co/settings/tokens (read access is enough) and "
            "paste it into the field below.",
        )

    def on_setup_sam3_package(self):
        def task(job: Job):
            if not venv_status():
                job.log("Set up the Python backend first (button above) — SAM3 "
                        "is installed into the same virtualenv.")
                return
            pip = os.path.join(VENV_DIR, "bin", "pip")
            job.run_cmd([pip, "install", "huggingface_hub"])
            src_dir = os.path.join(BACKEND_DIR, "sam3-src")
            if not os.path.isdir(os.path.join(src_dir, ".git")):
                job.log(f"Cloning {SAM3_REPO_URL}")
                rc = job.run_cmd(["git", "clone", SAM3_REPO_URL, src_dir])
                if rc != 0:
                    job.log("git clone failed — is git installed?")
                    return
            else:
                job.log(f"Reusing existing clone at {src_dir}")
                job.run_cmd(["git", "-C", src_dir, "pull"])
            job.log(
                "Installing the sam3 package (pip install -e .) — this needs "
                "Python 3.12+ and a recent PyTorch; expect this to fail on "
                "older/CPU-only setups, that's expected and OK if you don't "
                "plan on using SAM3."
            )
            job.run_cmd([pip, "install", "-e", src_dir])
            job.log("Done. Next: get an HF token approved (button above), "
                    "then use 'Download SAM3 checkpoint' below.")

        self.run_in_background(task)

    def on_download_sam3(self):
        token = self.hf_token.get().strip()
        if not token:
            messagebox.showerror("Missing token", "Paste your Hugging Face access token first.")
            return
        if not venv_status():
            messagebox.showerror("Backend missing", "Set up the Python backend first.")
            return

        def task(job: Job):
            dest = os.path.join(MODELS_DIR, "sam3")
            os.makedirs(dest, exist_ok=True)
            py = VENV_PYTHON
            script = (
                "import sys\n"
                "from huggingface_hub import snapshot_download\n"
                f"snapshot_download(repo_id={SAM3_HF_REPO_ID!r}, local_dir={dest!r}, "
                f"token={token!r})\n"
                "print('SAM3 checkpoint downloaded to', " + repr(dest) + ")\n"
            )
            job.log(f"Downloading {SAM3_HF_REPO_ID} to {dest} (this is several GB, be patient)")
            rc = job.run_cmd([py, "-c", script])
            if rc != 0:
                job.log(
                    "Download failed. Common causes: access not yet approved "
                    "on the Hugging Face model page, an invalid/expired "
                    "token, or huggingface_hub not installed (run "
                    "'Set up the sam3 package' first)."
                )
            else:
                job.log("SAM3 checkpoint ready.")

        self.run_in_background(task)


def main():
    root = tk.Tk()
    InstallerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
