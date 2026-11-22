"""Seeding, timing, LR schedule, batching, snapshots and environment capture."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    forced = os.environ.get("SETFIT_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def wall_clock(device: torch.device) -> float:
    sync(device)
    return time.perf_counter()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device):
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    return None


class TimeLedger:
    """Accumulates wall time per named section, synchronising CUDA at the boundaries."""

    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = defaultdict(float)

    @contextlib.contextmanager
    def section(self, name: str):
        sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync(self.device)
            self.seconds[name] += time.perf_counter() - t0

    def get(self, name: str) -> float:
        return float(self.seconds.get(name, 0.0))


def linear_warmup_decay(total_steps: int, warmup_frac: float):
    """LambdaLR factor: linear warmup over the first warmup_frac of updates, then linear decay.

    LambdaLR evaluates the factor at t = number of scheduler steps taken so far, so
    update t+1 uses lr * f(t). The first update uses lr/warmup (not 0) and the last
    update uses a small positive lr.
    """
    warmup = max(1, int(round(total_steps * warmup_frac)))

    def factor(t: int) -> float:
        if t < warmup:
            return (t + 1) / warmup
        return max(0.0, (total_steps - t) / max(1, total_steps - warmup))

    return factor


def batches(n: int, batch_size: int, rng: np.random.Generator):
    """Endless epochs over range(n): reshuffle every epoch; the last batch may be short."""
    while True:
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            yield perm[i:i + batch_size]


def make_grad_scaler(enabled: bool):
    # FP16 autocast needs loss scaling; a disabled scaler is a no-op (CPU / FP32).
    return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=2.0 ** 10)


def snapshot(module: torch.nn.Module) -> dict:
    """CPU copy of a module's state (used to keep the best checkpoint in memory)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_FINGERPRINT = None


def code_fingerprint() -> str:
    """Short hash of this package's source, stored with every run."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        h = hashlib.sha256()
        for p in sorted(PACKAGE_DIR.glob("*.py")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
        _FINGERPRINT = h.hexdigest()[:16]
    return _FINGERPRINT


def env_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "code_fingerprint": code_fingerprint(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                    gpu_memory_gb=round(props.total_memory / 2**30, 1))
    for mod in ("transformers", "sklearn", "numpy", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional dependency
            info[mod] = None
    return info


def write_json(path, obj) -> None:
    """Write JSON atomically (temp file + rename) so a disconnect never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)
