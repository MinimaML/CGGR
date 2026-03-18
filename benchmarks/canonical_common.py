import os
import platform
import socket
import subprocess
from typing import Dict, List

import torch


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def resolve_device(requested: str, require_cuda: bool = False) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda"
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return "mps"
    if requested == "cpu":
        if require_cuda:
            raise RuntimeError("CUDA is required for this run but CPU was requested")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if require_cuda:
            raise RuntimeError("CUDA is required for this run but only MPS is available")
        return "mps"
    if require_cuda:
        raise RuntimeError("CUDA is required for this run but no CUDA device is available")
    return "cpu"


def collect_platform_metadata(device: str) -> Dict:
    gpu_names: List[str] = []
    gpu_total_memory_mb: List[float] = []

    if device == "cuda":
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu_names.append(props.name)
            gpu_total_memory_mb.append(round(props.total_memory / 1024**2, 2))

    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "system": platform.platform(),
        "torch": torch.__version__,
        "device_type": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "visible_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else ("Apple Silicon GPU" if device == "mps" else "cpu")
        ),
        "visible_gpu_names": gpu_names,
        "visible_gpu_total_memory_mb": gpu_total_memory_mb,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }


def build_artifact_metadata(tag: str, tier: str, device: str, require_cuda: bool) -> Dict:
    return {
        "tag": tag,
        "tier": tier,
        "device_type": device,
        "require_cuda": require_cuda,
    }
