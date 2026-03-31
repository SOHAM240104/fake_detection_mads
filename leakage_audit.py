import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import PROJECT_ROOT


@dataclass(frozen=True)
class Split:
    name: str
    csv_path: str


def _extract_ids(path_str: str) -> dict[str, Any]:
    """
    AV1M metadata uses paths like:
      id04260/YRj7yYjVBYU/00021/real.mp4
    We derive:
      subject_id = first component
      video_id    = second component
      clip_id     = third component (frame/segment index)
    """
    comps = [c for c in str(path_str).split("/") if c]
    subject_id = comps[0] if len(comps) > 0 else None
    video_id = comps[1] if len(comps) > 1 else None
    clip_id = comps[2] if len(comps) > 2 else None
    return {"subject_id": subject_id, "video_id": video_id, "clip_id": clip_id}


def _load_split(split: Split) -> pd.DataFrame:
    df = pd.read_csv(split.csv_path)
    if "path" not in df.columns:
        raise ValueError(f"{split.csv_path} missing required column `path`")
    ids = df["path"].apply(_extract_ids)
    ids_df = pd.DataFrame(list(ids))
    df = pd.concat([df, ids_df], axis=1)
    df["split"] = split.name
    return df


def _pairwise_intersections(
    by_split: dict[str, set[str]],
    report: dict[str, Any],
    *,
    key_prefix: str = "",
) -> None:
    names = sorted(by_split.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = sorted(by_split[a].intersection(by_split[b]))
            report[f"{key_prefix}{a}_AND_{b}_overlap_count"] = len(inter)
            # Keep report compact.
            report[f"{key_prefix}{a}_AND_{b}_overlap_preview"] = inter[:50]


def audit_metadata_splits(metadata_dir: str, out_path: str) -> dict[str, Any]:
    train_csv = os.path.join(metadata_dir, "train_metadata.csv")
    val_csv = os.path.join(metadata_dir, "val_metadata.csv")
    test_csv = os.path.join(metadata_dir, "test_metadata.csv")

    for p in [train_csv, val_csv, test_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing metadata file: {p}")

    splits = [
        Split("train", train_csv),
        Split("val", val_csv),
        Split("test", test_csv),
    ]
    dfs = [_load_split(s) for s in splits]
    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    report: dict[str, Any] = {
        "metadata_dir": metadata_dir,
        "rows_total": int(len(all_df)),
    }

    for id_col in ["subject_id", "video_id", "clip_id"]:
        by_split: dict[str, set[str]] = {}
        for split_name, g in all_df.groupby("split"):
            vals = g[id_col].dropna().astype(str)
            by_split[split_name] = set(vals.tolist())
        report[id_col] = {k: len(v) for k, v in by_split.items()}
        _pairwise_intersections(by_split, report)

    # Leakage signal: any subject/video overlap is suspicious.
    leakage_subject = (
        report.get("train_AND_val_overlap_count", 0) > 0
        or report.get("train_AND_test_overlap_count", 0) > 0
        or report.get("val_AND_test_overlap_count", 0) > 0
    )
    # If clip_id overlaps but subject/video doesn't, it's usually expected.
    report["leakage_detected"] = bool(leakage_subject)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    return report


def audit_feature_dumps(
    *,
    features_dir: str,
    metadata_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Best-effort feature dump leakage audit.

    For AVH feature dumps produced by `AVH/dump_avh_features.py`, we expect each clip to have:
      <clip_id>.json  (with keys like video_path,label,clip_id)
      <clip_id>.npz
    """
    # Reload metadata splits to build id->split maps.
    metadata_dir = metadata_report["metadata_dir"]
    train_csv = os.path.join(metadata_dir, "train_metadata.csv")
    val_csv = os.path.join(metadata_dir, "val_metadata.csv")
    test_csv = os.path.join(metadata_dir, "test_metadata.csv")

    splits = [
        Split("train", train_csv),
        Split("val", val_csv),
        Split("test", test_csv),
    ]
    dfs = [_load_split(s) for s in splits]
    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    video_id_to_split: dict[str, set[str]] = {}
    subject_id_to_split: dict[str, set[str]] = {}
    for split_name, g in all_df.groupby("split"):
        for v in g["video_id"].dropna().astype(str).unique().tolist():
            video_id_to_split.setdefault(v, set()).add(split_name)
        for s in g["subject_id"].dropna().astype(str).unique().tolist():
            subject_id_to_split.setdefault(s, set()).add(split_name)

    feature_json_paths: list[str] = []
    for root, _, files in os.walk(features_dir):
        for fn in files:
            if fn.endswith(".json") and not fn.endswith("leakage_report.json"):
                feature_json_paths.append(os.path.join(root, fn))

    feature_json_paths = sorted(feature_json_paths)
    report: dict[str, Any] = {
        "features_dir": features_dir,
        "feature_json_rows_total": len(feature_json_paths),
    }

    if not feature_json_paths:
        report["feature_dump_audit_skipped"] = "no_json_files_found"
        return report

    feature_rows = []
    for p in feature_json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        video_path = meta.get("video_path") or meta.get("path") or meta.get("input") or None
        if not video_path:
            continue

        ids = _extract_ids(video_path)
        feature_rows.append(
            {
                "clip_meta_path": p,
                "video_path": video_path,
                **ids,
            }
        )

    report["feature_clips_with_video_path"] = len(feature_rows)

    # Derive assigned splits and check for ambiguity.
    assigned = []
    ambiguous = []
    unknown = 0
    for r in feature_rows:
        vid = r.get("video_id")
        sid = r.get("subject_id")
        splits_vid = video_id_to_split.get(str(vid), set()) if vid is not None else set()
        splits_sid = subject_id_to_split.get(str(sid), set()) if sid is not None else set()

        candidates = splits_vid.union(splits_sid)
        if not candidates:
            unknown += 1
            continue
        if len(candidates) > 1:
            ambiguous.append({"video_id": vid, "subject_id": sid, "splits": sorted(list(candidates))})
            continue
        assigned.append({"video_id": vid, "subject_id": sid, "split": sorted(list(candidates))[0]})

    report["feature_unknown_clips_count"] = int(unknown)
    report["feature_ambiguous_clips_count"] = int(len(ambiguous))
    report["feature_ambiguous_preview"] = ambiguous[:20]

    # Compute overlap within assigned splits (should be 0 for subject/video if dump is clean).
    by_split: dict[str, dict[str, set[str]]] = {
        "train": {"subject_id": set(), "video_id": set()},
        "val": {"subject_id": set(), "video_id": set()},
        "test": {"subject_id": set(), "video_id": set()},
    }
    for a in assigned:
        s = a["split"]
        if a.get("subject_id") is not None:
            by_split[s]["subject_id"].add(str(a["subject_id"]))
        if a.get("video_id") is not None:
            by_split[s]["video_id"].add(str(a["video_id"]))

    # Overlap analysis: subject_id is the most actionable leakage indicator.
    for id_col in ["subject_id", "video_id"]:
        by_id_split = {split_name: vals[id_col] for split_name, vals in by_split.items()}
        _pairwise_intersections(by_id_split, report=report, key_prefix=f"{id_col}__")

    report["leakage_detected_features"] = (
        report.get("subject_id__train_AND_val_overlap_count", 0) > 0
        or report.get("subject_id__train_AND_test_overlap_count", 0) > 0
        or report.get("subject_id__val_AND_test_overlap_count", 0) > 0
    )

    return report


def main():
    ap = argparse.ArgumentParser(description="Leakage/split validation for AV1M-style metadata.")
    ap.add_argument(
        "--metadata_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "AVH", "av1m_metadata"),
        help="Directory containing train_metadata.csv/val_metadata.csv/test_metadata.csv",
    )
    ap.add_argument(
        "--out_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "leakage_report.json"),
        help="Where to write leakage report JSON",
    )
    ap.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Optional AVH feature dump directory (expects <clip_id>.json metadata + <clip_id>.npz).",
    )
    args = ap.parse_args()

    report = audit_metadata_splits(metadata_dir=args.metadata_dir, out_path=args.out_path)

    if args.features_dir:
        report["features_audit"] = audit_feature_dumps(features_dir=args.features_dir, metadata_report=report)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report.get("leakage_detected") else 0)


if __name__ == "__main__":
    main()

