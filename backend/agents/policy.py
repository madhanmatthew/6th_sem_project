import math
import zlib

import numpy as np


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return clamp((value - low) / (high - low))


def inverse_normalize(value: float, low: float, high: float) -> float:
    return 1.0 - normalize(value, low, high)


def softmax_confidence(scores: list, temperature: float = 0.15) -> float:
    """
    Softmax confidence of the top-ranked candidate.

    temperature controls sharpness:
      - temperature=1.0  → standard softmax (near-uniform when scores are close)
      - temperature=0.15 → sharpened: small score gaps produce meaningfully
                           different confidence values (realistic 45-85% range)

    With 8 ambulance units scoring in [0.39, 0.67], temperature=1.0 gives
    every scenario ~14-16% (uniform — useless). temperature=0.15 amplifies
    gaps so a clear winner (delta ~0.25) yields ~72% confidence while a
    tight race yields ~40%, which is scenario-dependent and interpretable.
    """
    if not scores:
        return 0.0
    max_score = max(scores)
    shifted = [(s - max_score) / temperature for s in scores]
    exp_scores = [math.exp(s) for s in shifted]
    total = sum(exp_scores)
    return round(max(exp_scores) / total, 2)


def ranked(options: list, score_key: str = "score") -> list:
    return sorted(options, key=lambda item: item[score_key], reverse=True)


def compact_scores(options: list, keep: tuple) -> list:
    rows = []
    for option in options:
        row = {key: option[key] for key in keep if key in option}
        row["score"] = round(option["score"], 3)
        rows.append(row)
    return rows


def stable_rng(*parts: object) -> np.random.Generator:
    text = "|".join(str(part) for part in parts)
    seed = zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(seed)
