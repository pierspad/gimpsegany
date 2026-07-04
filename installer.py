#!/usr/bin/env python3
"""
gimpsegany installer — a small guided wizard that sets up the Segment
Anything plug-in for GIMP end to end, regardless of what's already in
place: it detects GIMP, the plug-in, the Python backend and your hardware
on its own, explains what it found in plain language, and only ever asks
you to confirm rather than hunt for folders yourself.

Run it with:  python3 installer.py

Every check re-runs fresh each time the wizard is shown, so this is safe
to launch repeatedly — it never assumes a clean slate and never touches
something that's already working.
"""

from __future__ import annotations

import glob
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))

# Shared with seganyplugin.py's discover_lazygimp_backend() and with
# LazyGimp's lib/segany_backend.sh — whichever tool sets this directory up,
# the plug-in finds it automatically. The name predates this standalone
# installer; it is not actually LazyGimp-specific.
BACKEND_DIR = os.path.expanduser("~/.local/share/lazygimp/segany")
VENV_DIR = os.path.join(BACKEND_DIR, "venv")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")

ACCENT = "#3daee9"  # KDE Breeze blue, to feel at home on this desktop
ACCENT_DARK = "#2f8fc4"
OK_GREEN = "#27ae60"
WARN_ORANGE = "#e67e22"
MUTED = "#6b6b6b"

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

FAMILY_INFO = {
    "SAM1": "SAM1 — Segment Anything (2023). General purpose, the most battle-tested with this plug-in.",
    "SAM2": "SAM2 — Segment Anything 2 (2024). Faster, similar quality, still point/box/auto based.",
    "SAM3": "SAM3 — Segment Anything 3 (2025). Text-prompt \"describe what to select\" segmentation. Gated on Hugging Face and GPU-only.",
}


@dataclass
class ModelSpec:
    key: str
    family: str  # "SAM1", "SAM2", "SAM3"
    label: str
    size: str
    note: str
    filename: str | None = None  # None for SAM3 (a folder, not a file)
    url: str | None = None  # None for SAM3 (gated, downloaded via token)
    min_python: tuple | None = None
    requires_gpu: bool = False


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        "sam_vit_b", "SAM1", "vit_b", "375 MB", "lightest, fastest on CPU",
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    ModelSpec(
        "sam_vit_l", "SAM1", "vit_l", "1.2 GB", "balanced (recommended default)",
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    ModelSpec(
        "sam_vit_h", "SAM1", "vit_h", "2.5 GB", "best quality, slow without a GPU",
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
    ModelSpec(
        "sam2_hiera_tiny", "SAM2", "hiera_tiny", "150 MB", "smallest SAM2 checkpoint",
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
        "sam2_hiera_large", "SAM2", "hiera_large", "900 MB", "best SAM2 quality",
        "sam2_hiera_large.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    ),
    ModelSpec(
        "sam3", "SAM3", "sam3.1", "~3.4 GB", "gated — needs HF approval",
        None, None, min_python=(3, 12), requires_gpu=True,
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


def model_incompatibility_reasons(spec: ModelSpec, hw: "Hardware") -> list[str]:
    """Empty list = compatible. Otherwise, plain-language reasons why this
    installer won't offer to download it on this machine."""
    reasons = []
    if spec.min_python and sys.version_info[:2] < spec.min_python:
        want = ".".join(map(str, spec.min_python))
        have = platform.python_version()
        reasons.append(f"needs Python {want}+ (this installer is running on Python {have})")
    if spec.requires_gpu and not hw.gpu:
        reasons.append("needs a dedicated NVIDIA or AMD GPU — none was detected on this machine")
    return reasons


# --------------------------------------------------------------------------
# Detection — all read-only, safe to call anytime, no side effects.
# --------------------------------------------------------------------------

@dataclass
class Hardware:
    cpu_cores: int
    python_version: str
    gpu: dict | None  # {"vendor": ..., "name": ..., "driver_ready": bool} or None


def detect_gpu() -> dict | None:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            line = next((l for l in out.stdout.splitlines() if l.strip()), None)
            if line:
                return {"vendor": "NVIDIA", "name": line.strip(), "driver_ready": True}
        except Exception:
            pass
    if shutil.which("rocminfo"):
        try:
            out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)
            name = next(
                (l.split(":", 1)[1].strip() for l in out.stdout.splitlines() if "Marketing Name" in l),
                None,
            )
            return {"vendor": "AMD (ROCm)", "name": name or "AMD GPU", "driver_ready": True}
        except Exception:
            pass
    if shutil.which("lspci"):
        try:
            out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            for line in out.stdout.splitlines():
                low = line.lower()
                if "vga" in low or "3d controller" in low:
                    desc = line.split(":", 2)[-1].strip()
                    if "nvidia" in low:
                        return {"vendor": "NVIDIA", "name": desc, "driver_ready": False}
                    if "amd" in low or "advanced micro devices" in low or "radeon" in low:
                        return {"vendor": "AMD", "name": desc, "driver_ready": False}
        except Exception:
            pass
    return None


def detect_hardware() -> Hardware:
    return Hardware(
        cpu_cores=os.cpu_count() or 1,
        python_version=platform.python_version(),
        gpu=detect_gpu(),
    )


def find_gimp_binary() -> str | None:
    return shutil.which("gimp") or shutil.which("gimp-3.0") or shutil.which("gimp-2.10")


def gimp_config_dirs() -> list[str]:
    """Every ~/.config/GIMP/X.Y directory that actually exists, newest first."""
    base = os.path.expanduser("~/.config/GIMP")
    if not os.path.isdir(base):
        return []
    dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]

    def version_key(path):
        parts = []
        for chunk in os.path.basename(path).split("."):
            try:
                parts.append(int(chunk))
            except ValueError:
                parts.append(0)
        return parts

    dirs.sort(key=version_key, reverse=True)
    return dirs


def find_plugins_dir() -> str | None:
    for d in gimp_config_dirs():
        return os.path.join(d, "plug-ins")
    return None


def plugin_install_status(plugins_dir: str | None) -> tuple[bool, str | None]:
    if not plugins_dir:
        return False, None
    dest = os.path.join(plugins_dir, "seganyplugin", "seganyplugin.py")
    return os.path.isfile(dest), dest


def venv_status() -> bool:
    return os.path.isfile(VENV_PYTHON) and os.access(VENV_PYTHON, os.X_OK)


def choose_directory_native(title: str, start_dir: str | None) -> str | None:
    """Open the desktop's own folder picker instead of Tk's generic one —
    on KDE that's the native Plasma/KIO dialog (the same one Dolphin uses),
    on GNOME/others it's zenity's portal-backed dialog. Falls back to Tk's
    picker only if neither is available."""
    start_dir = start_dir or os.path.expanduser("~")
    if shutil.which("kdialog"):
        try:
            out = subprocess.run(
                ["kdialog", "--title", title, "--getexistingdirectory", start_dir],
                capture_output=True, text=True, timeout=180,
            )
            path = out.stdout.strip()
            return path or None
        except Exception:
            pass
    if shutil.which("zenity"):
        try:
            out = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title", title,
                 f"--filename={start_dir}/"],
                capture_output=True, text=True, timeout=180,
            )
            path = out.stdout.strip()
            return path or None
        except Exception:
            pass
    return filedialog.askdirectory(title=title, initialdir=start_dir) or None


# --------------------------------------------------------------------------
# Background job runner — long actions (pip installs, downloads) run in a
# thread and stream lines back to the GUI through a queue, so the window
# never freezes. Same lesson as fixing the GIMP plug-in itself: never
# block the UI thread on a slow subprocess.
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

    def download(self, url: str, dest: str):
        import urllib.request

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        self.log(f"Downloading {url}")
        with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
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
# Wizard shell
# --------------------------------------------------------------------------

STEPS = ["Overview", "Plug-in", "Backend", "Models"]


class InstallerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("gimpsegany installer")
        root.geometry("980x720")
        root.minsize(860, 620)

        self._style()

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.busy = False
        self.hw = detect_hardware()
        self.plugins_dir = find_plugins_dir()

        self._build_shell()
        self.show_step(0)
        self.root.after(150, self._drain_log_queue)

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#f5f6f7")
        style.configure("TFrame", background="#f5f6f7")
        style.configure("Card.TFrame", background="white")
        style.configure("Sidebar.TFrame", background="#eef1f3")
        style.configure("TLabel", background="#f5f6f7", font=("Sans", 10))
        style.configure("Card.TLabel", background="white", font=("Sans", 10))
        style.configure("Header.TLabel", background="#f5f6f7", font=("Sans", 15, "bold"))
        style.configure("Card.Header.TLabel", background="white", font=("Sans", 13, "bold"))
        style.configure("Muted.TLabel", background="#f5f6f7", foreground=MUTED, font=("Sans", 9))
        style.configure("Card.Muted.TLabel", background="white", foreground=MUTED, font=("Sans", 9))
        style.configure("Ok.TLabel", background="white", foreground=OK_GREEN, font=("Sans", 10, "bold"))
        style.configure("Warn.TLabel", background="white", foreground=WARN_ORANGE, font=("Sans", 10, "bold"))
        style.configure("TButton", padding=7)
        style.configure("Accent.TButton", padding=8, foreground="white", background=ACCENT)
        style.map("Accent.TButton", background=[("active", ACCENT_DARK), ("disabled", "#a9d3e8")])
        style.configure("Step.TButton", padding=(14, 12), anchor="w")
        style.configure("StepActive.TButton", padding=(14, 12), anchor="w")

    def _build_shell(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        # --- sidebar ---
        sidebar = ttk.Frame(outer, style="Sidebar.TFrame", width=210)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ttk.Label(
            sidebar, text="gimpsegany", style="Header.TLabel", background="#eef1f3",
        ).pack(anchor="w", padx=16, pady=(20, 0))
        ttk.Label(
            sidebar, text="setup wizard", style="Muted.TLabel", background="#eef1f3",
        ).pack(anchor="w", padx=16, pady=(0, 16))

        self.step_buttons = []
        for i, name in enumerate(STEPS):
            btn = tk.Button(
                sidebar, text=f"{i + 1}.  {name}", anchor="w",
                relief="flat", bd=0, bg="#eef1f3", activebackground="#dde3e7",
                font=("Sans", 11), padx=16, pady=10,
                command=lambda i=i: self.show_step(i),
            )
            btn.pack(fill="x")
            self.step_buttons.append(btn)

        # --- content ---
        right = ttk.Frame(outer)
        right.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(right)
        self.content.pack(fill="both", expand=True, padx=24, pady=20)

        log_frame = ttk.LabelFrame(right, text="Activity log")
        log_frame.pack(fill="both", padx=24, pady=(0, 16), side="bottom")
        self.progress = ttk.Progressbar(log_frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=8, state="disabled", wrap="word",
                                 bg="#1e1e1e", fg="#d0d0d0", font=("Monospace", 9))
        self.log_text.pack(fill="both", padx=8, pady=8)

        self.steps = [
            OverviewStep(self),
            PluginStep(self),
            BackendStep(self),
            ModelsStep(self),
        ]

    def show_step(self, index: int):
        self.current_index = index
        for i, btn in enumerate(self.step_buttons):
            btn.configure(
                bg=ACCENT if i == index else "#eef1f3",
                fg="white" if i == index else "black",
                font=("Sans", 11, "bold" if i == index else "normal"),
            )
        for child in self.content.winfo_children():
            child.destroy()
        self.steps[index].build(self.content)

    def refresh_all(self):
        self.hw = detect_hardware()
        self.show_step(self.current_index)

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

    def set_busy(self, busy: bool):
        self.busy = busy
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def run_in_background(self, fn, on_done=None):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running — check the log.")
            return
        self.set_busy(True)

        def wrapper():
            job = Job(self.log_queue)
            try:
                fn(job)
            except Exception as e:
                job.log(f"ERROR: {e}")
            finally:
                self.log_queue.put("--- done ---")
                self.root.after(0, lambda: (self.set_busy(False), on_done() if on_done else self.refresh_all()))

        threading.Thread(target=wrapper, daemon=True).start()


def card(parent) -> ttk.Frame:
    outer = ttk.Frame(parent)
    outer.pack(fill="both", expand=True)
    c = tk.Frame(outer, bg="white", highlightbackground="#dcdfe2", highlightthickness=1)
    c.pack(fill="both", expand=True)
    return c


# --------------------------------------------------------------------------
# Step 1 — Overview
# --------------------------------------------------------------------------

class OverviewStep:
    def __init__(self, app: InstallerApp):
        self.app = app

    def build(self, parent):
        ttk.Label(parent, text="What's on this machine", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Detected automatically — nothing below required you to hunt for a folder.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        app = self.app
        hw = app.hw
        gimp_bin = find_gimp_binary()
        installed, dest = plugin_install_status(app.plugins_dir)
        backend_ready = venv_status()
        n_installed = sum(1 for m in MODEL_REGISTRY if model_installed(m))

        c = card(parent)
        rows = [
            ("GIMP", f"found at {gimp_bin}" if gimp_bin else "not found on PATH — install GIMP first", bool(gimp_bin)),
            ("Plug-in files", f"installed at {dest}" if installed else f"not installed yet (target: {app.plugins_dir or 'unknown'})", installed),
            ("Python backend", f"ready ({VENV_DIR})" if backend_ready else "not set up yet", backend_ready),
            ("Models downloaded", f"{n_installed} of {len(MODEL_REGISTRY)}", n_installed > 0),
        ]
        for label, detail, ok in rows:
            row = tk.Frame(c, bg="white")
            row.pack(fill="x", padx=18, pady=8)
            mark = "✓" if ok else "○"
            style = "Ok.TLabel" if ok else "Card.Muted.TLabel"
            ttk.Label(row, text=mark, style=style, width=2).pack(side="left")
            ttk.Label(row, text=label, style="Card.Header.TLabel", width=18).pack(side="left")
            ttk.Label(row, text=detail, style="Card.TLabel", wraplength=560, justify="left").pack(side="left")

        ttk.Separator(c).pack(fill="x", padx=18, pady=8)

        hw_row = tk.Frame(c, bg="white")
        hw_row.pack(fill="x", padx=18, pady=(0, 6))
        ttk.Label(hw_row, text=" ", width=2, background="white").pack(side="left")
        ttk.Label(hw_row, text="Hardware", style="Card.Header.TLabel", width=18).pack(side="left")
        gpu_desc = f"{hw.gpu['vendor']} — {hw.gpu['name']}" if hw.gpu else "no dedicated GPU detected"
        if hw.gpu and not hw.gpu.get("driver_ready", True):
            gpu_desc += " (present, but drivers don't look installed/loaded)"
        ttk.Label(
            hw_row,
            text=f"{hw.cpu_cores} CPU cores · {gpu_desc} · Python {hw.python_version}",
            style="Card.TLabel", wraplength=560, justify="left",
        ).pack(side="left")

        explain = tk.Frame(c, bg="white")
        explain.pack(fill="x", padx=18, pady=(4, 16))
        if hw.gpu and hw.gpu.get("driver_ready"):
            note = (
                "A GPU is available, so segmentation will run there and use its "
                "native parallelism automatically — no configuration needed. "
                "SAM3 will also be offered, since it requires a GPU."
            )
        else:
            note = (
                "No usable GPU was found, so segmentation will run on CPU. "
                "This plug-in already configures PyTorch/OpenCV to use all "
                f"{hw.cpu_cores} cores for every operation — there is no extra "
                "toggle to enable, more cores here is a hardware upgrade, not a "
                "setting. SAM3 needs a real GPU per its own requirements, so it "
                "won't be offered for download on this machine."
            )
        ttk.Label(explain, text=note, style="Card.Muted.TLabel", wraplength=760, justify="left").pack(anchor="w")

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(16, 0))
        ttk.Button(btns, text="Re-check", command=self.app.refresh_all).pack(side="left")
        ttk.Button(btns, text="Continue →", style="Accent.TButton",
                   command=lambda: self.app.show_step(1)).pack(side="left", padx=8)


# --------------------------------------------------------------------------
# Step 2 — Plug-in files
# --------------------------------------------------------------------------

class PluginStep:
    def __init__(self, app: InstallerApp):
        self.app = app

    def build(self, parent):
        app = self.app
        ttk.Label(parent, text="Plug-in files", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="This copies seganyplugin.py and seganybridge.py into GIMP's plug-ins folder.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        c = card(parent)
        inner = tk.Frame(c, bg="white")
        inner.pack(fill="both", expand=True, padx=18, pady=18)

        installed, dest = plugin_install_status(app.plugins_dir)
        status_style = "Ok.TLabel" if installed else "Card.Muted.TLabel"
        status_text = "Already installed here — this will just update it to the latest version." if installed else "Not installed yet."
        ttk.Label(inner, text=status_text, style=status_style).pack(anchor="w", pady=(0, 12))

        ttk.Label(inner, text="Target folder (auto-detected):", style="Card.Header.TLabel").pack(anchor="w")
        row = tk.Frame(inner, bg="white")
        row.pack(fill="x", pady=(4, 4))
        self.path_var = tk.StringVar(value=app.plugins_dir or "(none found — choose manually)")
        entry = ttk.Entry(row, textvariable=self.path_var, width=70)
        entry.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Change…", command=self.on_change_dir).pack(side="left", padx=6)

        if app.plugins_dir:
            ttk.Label(
                inner,
                text="Found by scanning ~/.config/GIMP for the newest version folder — just confirm it's right.",
                style="Card.Muted.TLabel",
            ).pack(anchor="w", pady=(0, 12))
        else:
            ttk.Label(
                inner,
                text="Couldn't find a GIMP config folder. Launch GIMP once so it creates one, or pick the "
                     "plug-ins folder manually.",
                style="Card.Muted.TLabel",
            ).pack(anchor="w", pady=(0, 12))

        ttk.Button(
            inner, text="Install / Update Plug-in Files", style="Accent.TButton",
            command=self.on_install,
        ).pack(anchor="w", pady=(8, 0))

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(16, 0))
        ttk.Button(btns, text="← Back", command=lambda: app.show_step(0)).pack(side="left")
        ttk.Button(btns, text="Continue →", style="Accent.TButton",
                   command=lambda: app.show_step(2)).pack(side="left", padx=8)

    def on_change_dir(self):
        chosen = choose_directory_native("Select your GIMP plug-ins folder", self.app.plugins_dir)
        if chosen:
            self.app.plugins_dir = chosen
            self.app.show_step(1)

    def on_install(self):
        plugins_dir = self.path_var.get().strip()
        if not plugins_dir or not os.path.isdir(os.path.dirname(plugins_dir) or plugins_dir):
            messagebox.showerror("No folder", "Pick a valid GIMP plug-ins folder first.")
            return
        self.app.plugins_dir = plugins_dir

        def task(job: Job):
            src_plugin = os.path.join(HERE, "seganyplugin.py")
            src_bridge = os.path.join(HERE, "seganybridge.py")
            if not (os.path.isfile(src_plugin) and os.path.isfile(src_bridge)):
                job.log(f"ERROR: expected seganyplugin.py/seganybridge.py next to this "
                        f"installer ({HERE}).")
                return
            dest_dir = os.path.join(plugins_dir, "seganyplugin")
            job.log(f"Installing into {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            for f in (src_plugin, src_bridge):
                shutil.copy2(f, dest_dir)
                os.chmod(os.path.join(dest_dir, os.path.basename(f)), 0o755)
            job.log("Plug-in files installed. Restart GIMP to load the change.")

        self.app.run_in_background(task, on_done=lambda: self.app.show_step(1))


# --------------------------------------------------------------------------
# Step 3 — Python backend
# --------------------------------------------------------------------------

class BackendStep:
    def __init__(self, app: InstallerApp):
        self.app = app

    def build(self, parent):
        app = self.app
        ttk.Label(parent, text="Python backend", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="A dedicated virtual environment with PyTorch, OpenCV and the SAM1/SAM2 packages.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        c = card(parent)
        inner = tk.Frame(c, bg="white")
        inner.pack(fill="both", expand=True, padx=18, pady=18)

        ready = venv_status()
        style = "Ok.TLabel" if ready else "Card.Muted.TLabel"
        ttk.Label(
            inner,
            text=f"Ready at {VENV_DIR}" if ready else "Not set up yet.",
            style=style,
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(inner, text="PyTorch build:", style="Card.Header.TLabel").pack(anchor="w")
        default_choice = list(TORCH_INDEX_URLS.keys())[0]
        hw = app.hw
        if hw.gpu and hw.gpu.get("driver_ready"):
            if "NVIDIA" in hw.gpu["vendor"]:
                default_choice = "NVIDIA CUDA 12.8"
            elif "AMD" in hw.gpu["vendor"]:
                default_choice = "AMD ROCm 6.2"
        self.torch_choice = tk.StringVar(value=default_choice)
        row = tk.Frame(inner, bg="white")
        row.pack(fill="x", pady=(4, 4))
        ttk.Combobox(
            row, textvariable=self.torch_choice, values=list(TORCH_INDEX_URLS.keys()),
            state="readonly", width=34,
        ).pack(side="left")
        ttk.Label(
            inner,
            text="Pre-selected based on the hardware detected in step 1 — change it if that's wrong.",
            style="Card.Muted.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        note = tk.Frame(inner, bg="white")
        note.pack(fill="x", pady=(4, 12))
        ttk.Label(
            note,
            text=("Why this matters for speed: on a GPU build, PyTorch dispatches every operation across "
                  "thousands of GPU cores automatically — that parallelism is inherent to CUDA/ROCm, nothing "
                  "to configure. On the CPU build, this plug-in already sets OMP_NUM_THREADS/MKL_NUM_THREADS "
                  "and torch's own thread pool to use every core on this machine for each matrix operation; "
                  "there's no further manual parallelization that would help beyond that."),
            style="Card.Muted.TLabel", wraplength=760, justify="left",
        ).pack(anchor="w")

        ttk.Button(
            inner, text="Create / Repair Backend (SAM1 + SAM2)", style="Accent.TButton",
            command=self.on_setup,
        ).pack(anchor="w")

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(16, 0))
        ttk.Button(btns, text="← Back", command=lambda: app.show_step(1)).pack(side="left")
        ttk.Button(btns, text="Continue →", style="Accent.TButton",
                   command=lambda: app.show_step(3)).pack(side="left", padx=8)

    def on_setup(self):
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

        self.app.run_in_background(task, on_done=lambda: self.app.show_step(2))


# --------------------------------------------------------------------------
# Step 4 — Models
# --------------------------------------------------------------------------

class ModelsStep:
    def __init__(self, app: InstallerApp):
        self.app = app

    def build(self, parent):
        app = self.app
        ttk.Label(parent, text="Models", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Everything below is a hardcoded, known-good list of official checkpoints — pick what you need.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        c = card(parent)
        inner = tk.Frame(c, bg="white")
        inner.pack(fill="both", expand=True, padx=14, pady=14)

        columns = ("size", "note", "status")
        self.tree = ttk.Treeview(inner, columns=columns, show="tree headings", height=10)
        self.tree.heading("#0", text="Model")
        self.tree.column("#0", width=200, anchor="w")
        for col, width, heading in [("size", 90, "Size"), ("note", 260, "Note"), ("status", 140, "Status")]:
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(0, 8))
        self.tree.tag_configure("incompatible", foreground="#a0a0a0")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        family_nodes = {}
        for family, desc in FAMILY_INFO.items():
            node = self.tree.insert("", "end", iid=f"family:{family}", text=family, open=True)
            family_nodes[family] = node
        for spec in MODEL_REGISTRY:
            reasons = model_incompatibility_reasons(spec, app.hw)
            status = "installed" if model_installed(spec) else ("unavailable here" if reasons else "not installed")
            tags = ("incompatible",) if reasons else ()
            self.tree.insert(
                family_nodes[spec.family], "end", iid=spec.key,
                text=spec.label, values=(spec.size, spec.note, status), tags=tags,
            )

        self.detail = ttk.Label(inner, text="Select a model above to see details.",
                                 style="Card.Muted.TLabel", wraplength=760, justify="left")
        self.detail.pack(anchor="w", pady=(0, 8))

        action_row = tk.Frame(inner, bg="white")
        action_row.pack(fill="x")
        self.install_btn = ttk.Button(action_row, text="Download Selected", style="Accent.TButton",
                                       command=self.on_install_model, state="disabled")
        self.install_btn.pack(side="left")
        self.uninstall_btn = ttk.Button(action_row, text="Remove Selected", command=self.on_uninstall_model,
                                         state="disabled")
        self.uninstall_btn.pack(side="left", padx=6)

        self.sam3_frame = ttk.LabelFrame(parent, text="SAM3 — gated Hugging Face model, step by step")
        self.sam3_frame.pack(fill="x", pady=(14, 0))
        reasons3 = model_incompatibility_reasons(
            next(m for m in MODEL_REGISTRY if m.family == "SAM3"), app.hw
        )
        if reasons3:
            info = "Not available on this machine: " + "; ".join(reasons3) + "."
            ttk.Label(self.sam3_frame, text=info, foreground=WARN_ORANGE,
                      wraplength=820, justify="left").pack(anchor="w", padx=8, pady=(4, 8))
            self.sam3_override = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                self.sam3_frame, text="I understand the risk — let me try anyway",
                variable=self.sam3_override, command=self._sync_sam3_enabled,
            ).pack(anchor="w", padx=8)
        else:
            ttk.Label(
                self.sam3_frame,
                text="This machine meets SAM3's requirements. You still need Hugging Face approval before downloading.",
                wraplength=820, justify="left",
            ).pack(anchor="w", padx=8, pady=(4, 8))
            self.sam3_override = tk.BooleanVar(value=True)

        row3 = ttk.Frame(self.sam3_frame)
        row3.pack(fill="x", padx=8, pady=4)
        ttk.Button(row3, text="1. Request access on Hugging Face", command=self.on_open_hf).pack(side="left")
        self.sam3_setup_btn = ttk.Button(
            row3, text="2. Set up the sam3 package (clone + pip install)", command=self.on_setup_sam3_package,
        )
        self.sam3_setup_btn.pack(side="left", padx=6)
        row4 = ttk.Frame(self.sam3_frame)
        row4.pack(fill="x", padx=8, pady=(4, 10))
        ttk.Label(row4, text="3. Hugging Face token:").pack(side="left")
        self.hf_token = tk.StringVar()
        self.hf_entry = ttk.Entry(row4, textvariable=self.hf_token, show="*", width=44)
        self.hf_entry.pack(side="left", padx=6)
        self.sam3_download_btn = ttk.Button(row4, text="Download SAM3 checkpoint", command=self.on_download_sam3)
        self.sam3_download_btn.pack(side="left", padx=6)
        self._sync_sam3_enabled()

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(16, 0))
        ttk.Button(btns, text="← Back", command=lambda: app.show_step(2)).pack(side="left")

    def _sync_sam3_enabled(self):
        enabled = self.sam3_override.get()
        state = "normal" if enabled else "disabled"
        self.sam3_setup_btn.configure(state=state)
        self.hf_entry.configure(state=state)
        self.sam3_download_btn.configure(state=state)

    def _selected_spec(self) -> ModelSpec | None:
        sel = self.tree.selection()
        if not sel or self.tree.parent(sel[0]) == "":
            return None
        key = sel[0]
        return next((m for m in MODEL_REGISTRY if m.key == key), None)

    def on_select(self, _event=None):
        spec = self._selected_spec()
        if spec is None:
            self.install_btn.configure(state="disabled")
            self.uninstall_btn.configure(state="disabled")
            self.detail.configure(text="Select a model above to see details.")
            return
        reasons = model_incompatibility_reasons(spec, self.app.hw)
        installed = model_installed(spec)
        self.uninstall_btn.configure(state="normal" if installed else "disabled")
        if spec.family == "SAM3":
            self.install_btn.configure(state="disabled")
            self.detail.configure(text="Use the SAM3 panel below — it needs a token, not a plain download.")
        elif reasons:
            self.install_btn.configure(state="disabled")
            self.detail.configure(text="Can't download here: " + "; ".join(reasons) + ".")
        else:
            self.install_btn.configure(state="disabled" if installed else "normal")
            self.detail.configure(
                text=f"{FAMILY_INFO[spec.family]}\nSaved to {model_path(spec)}."
            )

    def on_install_model(self):
        spec = self._selected_spec()
        if spec is None or spec.family == "SAM3":
            return

        def task(job: Job):
            dest = model_path(spec)
            if os.path.isfile(dest):
                job.log(f"{spec.label} already downloaded at {dest}")
                return
            job.download(spec.url, dest)

        self.app.run_in_background(task)

    def on_uninstall_model(self):
        spec = self._selected_spec()
        if spec is None:
            return
        dest = model_path(spec)
        if not model_installed(spec):
            return
        if not messagebox.askyesno("Remove model", f"Delete {dest}?"):
            return

        def task(job: Job):
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            else:
                os.remove(dest)
            job.log(f"Removed {dest}")

        self.app.run_in_background(task)

    def on_open_hf(self):
        webbrowser.open(SAM3_HF_PAGE)
        messagebox.showinfo(
            "Hugging Face access",
            "Log in to Hugging Face, request access on the model page, and wait for approval.\n\n"
            "Then create a personal access token at huggingface.co/settings/tokens (read access is "
            "enough) and paste it into the field below.",
        )

    def on_setup_sam3_package(self):
        def task(job: Job):
            if not venv_status():
                job.log("Set up the Python backend first (previous step) — SAM3 installs into the same virtualenv.")
                return
            pip = os.path.join(VENV_DIR, "bin", "pip")
            job.run_cmd([pip, "install", "huggingface_hub"])
            src_dir = os.path.join(BACKEND_DIR, "sam3-src")
            if not os.path.isdir(os.path.join(src_dir, ".git")):
                job.log(f"Cloning {SAM3_REPO_URL}")
                if job.run_cmd(["git", "clone", SAM3_REPO_URL, src_dir]) != 0:
                    job.log("git clone failed — is git installed?")
                    return
            else:
                job.log(f"Reusing existing clone at {src_dir}")
                job.run_cmd(["git", "-C", src_dir, "pull"])
            job.log("Installing the sam3 package (pip install -e .) — needs Python 3.12+ and a recent PyTorch.")
            job.run_cmd([pip, "install", "-e", src_dir])
            job.log("Done. Next: get an HF token approved, then use 'Download SAM3 checkpoint'.")

        self.app.run_in_background(task)

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
            script = (
                "from huggingface_hub import snapshot_download\n"
                f"snapshot_download(repo_id={SAM3_HF_REPO_ID!r}, local_dir={dest!r}, token={token!r})\n"
                f"print('SAM3 checkpoint downloaded to', {dest!r})\n"
            )
            job.log(f"Downloading {SAM3_HF_REPO_ID} to {dest} (several GB, be patient)")
            rc = job.run_cmd([VENV_PYTHON, "-c", script])
            if rc != 0:
                job.log(
                    "Download failed. Common causes: access not yet approved on the Hugging Face "
                    "model page, an invalid/expired token, or huggingface_hub not installed yet "
                    "(run 'Set up the sam3 package' first)."
                )
            else:
                job.log("SAM3 checkpoint ready.")

        self.app.run_in_background(task)


def main():
    root = tk.Tk()
    InstallerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
