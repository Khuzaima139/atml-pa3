"""Run provenance: git commit, dirty flag, host, hardware, timestamp."""
import datetime
import os
import platform
import socket
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info():
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def hardware_info():
    info = {"cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "gpus": []}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": p.name,
                                 "total_memory_gb": round(p.total_memory / 2**30, 2)})
    return info


def run_info():
    """Provenance block stored in every result JSON."""
    return {"git": git_info(), "hostname": socket.gethostname(),
            "hardware": hardware_info(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
