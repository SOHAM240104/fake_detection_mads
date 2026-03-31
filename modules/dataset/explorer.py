import os
from typing import List, Dict

import pandas as pd

from config import PROJECT_ROOT


def list_local_example_audios() -> List[str]:
    """
    List example audio files under the repo's `audios/` directory, if present.
    """
    root = os.path.join(PROJECT_ROOT, "audios")
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for fn in sorted(os.listdir(root)):
        if fn.lower().endswith((".wav", ".mp3", ".ogg")):
            out.append(os.path.join(root, fn))
    return out


def load_av1m_metadata(split: str = "train") -> pd.DataFrame | None:
    """
    Load AV1M-style metadata from AVH/av1m_metadata/*_metadata.csv if available.
    """
    from pathlib import Path

    valid = {"train", "val", "test"}
    if split not in valid:
        raise ValueError(f"split must be one of {valid}")

    csv_path = Path(PROJECT_ROOT) / "AVH" / "av1m_metadata" / f"{split}_metadata.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


def summarize_class_distribution(df: pd.DataFrame, label_col: str = "label") -> Dict[str, int]:
    if df is None or label_col not in df.columns:
        return {}
    counts = df[label_col].value_counts().to_dict()
    # Ensure keys are strings for JSON/Streamlit friendliness.
    return {str(k): int(v) for k, v in counts.items()}

