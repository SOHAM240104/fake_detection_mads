import bz2
import json
import os
import tempfile
import time
import urllib.request

import hashlib
from dataclasses import dataclass
from typing import Any

from config import PROJECT_ROOT


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    path: str
    # Optional download URL. If provided, artifact_manager can download missing artifacts.
    url: str | None = None
    # Optional expected sha256 for strict verification.
    expected_sha256: str | None = None
    # Optional download postprocess type.
    postprocess: str | None = None  # e.g. "bz2_extract"
    # For bz2_extract, the downloaded URL must be the .bz2 archive.


DEFAULT_LOCK_PATH = os.path.join(PROJECT_ROOT, "artifacts.lock.json")


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_lock(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".tmp")
    try:
        tmp.write(json.dumps(obj, indent=2, sort_keys=True))
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, path)


def _download_to_file(url: str, dest_path: str, timeout_s: int = 60) -> None:
    """
    Download URL to dest_path atomically (via temp file).
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".download", dir=os.path.dirname(dest_path) or None)
    tmp_path = tmp.name
    tmp.close()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fake-audio-detection/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def _bz2_extract(src_bz2: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with bz2.open(src_bz2, "rb") as f_in:
        with open(dest_path, "wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)


def default_artifacts() -> list[ArtifactSpec]:
    # URLs are sourced from AVH/README.md and the Grad-CAM/AVH setup instructions.
    return [
        ArtifactSpec(
            name="noma_model",
            path=os.path.join(PROJECT_ROOT, "model", "noma-1"),
            url=None,  # The repo vendors the file (LFS). If missing in a deployment, you need to provide it.
            expected_sha256=None,
        ),
        ArtifactSpec(
            name="avh_fusion_ckpt",
            path=os.path.join(PROJECT_ROOT, "AVH", "checkpoints", "AVH-Align_AV1M.pt"),
            url=None,  # Not published in README; currently repo-vendored.
            expected_sha256=None,
        ),
        ArtifactSpec(
            name="avhubert_ckpt",
            path=os.path.join(PROJECT_ROOT, "AVH", "av_hubert", "avhubert", "self_large_vox_433h.pt"),
            url="https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt",
            expected_sha256=None,
        ),
        ArtifactSpec(
            name="dlib_face_predictor",
            path=os.path.join(
                PROJECT_ROOT,
                "AVH",
                "av_hubert",
                "avhubert",
                "content",
                "data",
                "misc",
                "shape_predictor_68_face_landmarks.dat",
            ),
            url="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            postprocess="bz2_extract",
            expected_sha256=None,
        ),
        ArtifactSpec(
            name="mean_face",
            path=os.path.join(
                PROJECT_ROOT,
                "AVH",
                "av_hubert",
                "avhubert",
                "content",
                "data",
                "misc",
                "20words_mean_face.npy",
            ),
            url="https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy",
            expected_sha256=None,
        ),
    ]


def ensure_artifacts(
    *,
    artifacts: list[ArtifactSpec] | None = None,
    lock_path: str = DEFAULT_LOCK_PATH,
    write_lock_if_missing: bool = True,
    download_missing: bool = False,
    strict_hash: bool = False,
    # If strict_hash is False, we only compute SHA256 for artifacts <= this size.
    sha256_size_threshold_bytes: int = 200 * 1024 * 1024,
) -> dict[str, Any]:
    """
    Ensure artifacts exist and (optionally) verify pinned hashes via a lockfile.

    Lockfile format (per-artifact):
      {
        "path": "...",
        "size": <bytes>,
        "mtime": <float>,
        "sha256": "<hex>" | null,
        "verified_at": <unix_ts>,
      }
    """
    artifacts = artifacts or default_artifacts()
    lock = _load_lock(lock_path)

    changed = False
    results: dict[str, Any] = {"lock_path": lock_path, "artifacts": {}}

    for spec in artifacts:
        st: dict[str, Any] = {"path": spec.path}
        entry = lock.get(spec.name, {})

        exists = os.path.isfile(spec.path)
        if exists:
            size = os.path.getsize(spec.path)
            mtime = os.path.getmtime(spec.path)
            st["size"] = size
            st["mtime"] = mtime

            # Decide whether we can trust lock without recomputing.
            lock_match = bool(entry) and entry.get("size") == size and entry.get("mtime") == mtime
            sha_in_lock = entry.get("sha256")

            sha256_ok = None
            sha256_computed = False

            if sha_in_lock and lock_match:
                sha256_ok = True if spec.expected_sha256 is None else (sha_in_lock == spec.expected_sha256)
            elif strict_hash or (size <= sha256_size_threshold_bytes):
                sha = sha256_file(spec.path)
                sha256_computed = True
                sha256_ok = True if spec.expected_sha256 is None else (sha == spec.expected_sha256)
                st["sha256"] = sha
            else:
                # Avoid hashing huge files by default; record sha as null for determinism.
                st["sha256"] = None

            st["sha256_ok"] = sha256_ok
            st["status"] = "verified" if sha256_ok is not False else "hash_mismatch"

            # Update lock if needed.
            if write_lock_if_missing and (not entry or changed or sha256_computed):
                lock[spec.name] = {
                    "path": spec.path,
                    "size": size,
                    "mtime": mtime,
                    "sha256": st.get("sha256"),
                    "verified_at": int(time.time()),
                }
                changed = True

        else:
            st["size"] = None
            st["mtime"] = None
            st["sha256"] = None
            st["status"] = "missing"

            if download_missing and spec.url:
                os.makedirs(os.path.dirname(spec.path) or ".", exist_ok=True)
                tmp_dir = tempfile.mkdtemp(prefix=f"artifact_{spec.name}_")
                try:
                    dest_tmp_path = os.path.join(tmp_dir, os.path.basename(spec.path) + ".download")
                    _download_to_file(spec.url, dest_tmp_path, timeout_s=120)

                    if spec.postprocess == "bz2_extract":
                        # dest_tmp_path is the .bz2 file; extract to spec.path.
                        extracted_bz2 = dest_tmp_path
                        _bz2_extract(extracted_bz2, spec.path)
                    else:
                        os.replace(dest_tmp_path, spec.path)

                    size = os.path.getsize(spec.path)
                    mtime = os.path.getmtime(spec.path)
                    st["size"] = size
                    st["mtime"] = mtime

                    sha = None
                    if strict_hash or (size <= sha256_size_threshold_bytes) or (spec.expected_sha256 is not None):
                        sha = sha256_file(spec.path)
                        st["sha256"] = sha
                        st["sha256_ok"] = True if spec.expected_sha256 is None else (sha == spec.expected_sha256)
                    else:
                        st["sha256_ok"] = None
                        st["sha256"] = None

                    st["status"] = "downloaded"

                    if write_lock_if_missing:
                        lock[spec.name] = {
                            "path": spec.path,
                            "size": size,
                            "mtime": mtime,
                            "sha256": sha,
                            "verified_at": int(time.time()),
                        }
                        changed = True
                finally:
                    try:
                        import shutil

                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass
            elif download_missing and not spec.url:
                st["status"] = "missing_and_no_url"
            elif not download_missing:
                st["status"] = "missing"

        results["artifacts"][spec.name] = st

    if changed and write_lock_if_missing:
        _atomic_write_json(lock_path, lock)

    return results

