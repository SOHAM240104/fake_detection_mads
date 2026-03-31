import json
import os
import signal
import subprocess
import time
from typing import Any

from config import AVH_PYTHON_ALLOWLIST


def validate_python_exe(python_exe: str | None) -> str:
    """
    Enforce an allowlist for AVH subprocess execution.

    This is intentionally strict to reduce arbitrary code execution risk.
    """
    p = (python_exe or "").strip()
    if not p:
        raise ValueError("python_exe must be provided and must be on the allowlist.")
    if not os.path.isfile(p):
        raise ValueError(f"python_exe not found or not a file: {p}")
    if os.path.abspath(p) not in [os.path.abspath(x) for x in AVH_PYTHON_ALLOWLIST]:
        raise ValueError(
            f"python_exe is not allowed: {p}. Allowed values: {AVH_PYTHON_ALLOWLIST}"
        )
    return p


def run_subprocess_capture(
    cmd: list[str],
    *,
    cwd: str,
    timeout_s: int,
) -> dict[str, Any]:
    """
    Run a subprocess with:
      - its own process group (new session)
      - hard timeout that kills the entire group
      - stdout/stderr captured for debugging
    """
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # create a new process group/session on Unix
    )

    try:
        stdout, stderr = p.communicate(timeout=timeout_s)
        return {"returncode": p.returncode, "stdout": stdout, "stderr": stderr, "timed_out": False}
    except subprocess.TimeoutExpired:
        # Kill the whole process group.
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass

        # Give it a moment to exit gracefully.
        try:
            stdout, stderr = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
            stdout, stderr = p.communicate()

        return {
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
        }


def safe_read_json(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

