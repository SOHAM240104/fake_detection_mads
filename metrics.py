from collections import defaultdict
from typing import Dict, Tuple


_counters: Dict[Tuple[str, str], int] = defaultdict(int)
_latency_ms_sum: Dict[Tuple[str, str], float] = defaultdict(float)
_latency_ms_count: Dict[Tuple[str, str], int] = defaultdict(int)


def inc_counter(name: str, **labels: str) -> None:
    key = (name, _labels_key(labels))
    _counters[key] += 1


def observe_latency_ms(name: str, value_ms: float, **labels: str) -> None:
    key = (name, _labels_key(labels))
    _latency_ms_sum[key] += float(value_ms)
    _latency_ms_count[key] += 1


def snapshot() -> dict:
    out = {
        "counters": [],
        "latency_ms": [],
    }
    for (name, label_str), value in _counters.items():
        out["counters"].append({"name": name, "labels": _labels_from_str(label_str), "value": int(value)})
    for (name, label_str), s in _latency_ms_sum.items():
        c = _latency_ms_count[(name, label_str)]
        out["latency_ms"].append(
            {
                "name": name,
                "labels": _labels_from_str(label_str),
                "count": int(c),
                "sum": float(s),
                "avg": float(s / c) if c > 0 else 0.0,
            }
        )
    return out


def _labels_key(labels: Dict[str, str]) -> str:
    items = sorted(labels.items())
    return "|".join(f"{k}={v}" for k, v in items)


def _labels_from_str(s: str) -> Dict[str, str]:
    if not s:
        return {}
    out: Dict[str, str] = {}
    for part in s.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out

