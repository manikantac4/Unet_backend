"""
Water-body ranking engine.

The segmentation model answers "where is water". This module
answers "which detected water bodies rank higher, given the
selected/configured criteria" — a separate, transparent, and fully
configurable scoring step, not a scientific measurement.

Per the brief: weights are NOT scientifically derived. They are
explicit, documented defaults (config.settings.RANKING_WEIGHTS),
overridable per-request, and every criterion used is data that
already exists on the WaterBody object — nothing is invented
(no accessibility/persistence scoring, since that data doesn't
exist in this pipeline).
"""

from typing import List, Optional

from config.settings import RANKING_WEIGHTS
from water_bodies.schema import WaterBody


def _minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]  # all equal -> all get full score
    return [(v - lo) / (hi - lo) for v in values]


def rank_water_bodies(
    water_bodies: List[WaterBody],
    weights: Optional[dict] = None,
) -> List[WaterBody]:
    """
    Computes `ranking_score` (0-100) and `rank` (1 = best) for every
    WaterBody in place, and returns the list sorted by rank.

    weights: optional override of config.settings.RANKING_WEIGHTS.
    Must contain the same 4 keys and sum to ~1.0.

    Scoring criteria (all min-max normalized across the current
    result set, so scores are only meaningful WITHIN one analysis,
    not across different analyses):
        - area_score:               larger area -> higher score
        - probability_score:        higher mean model probability -> higher score
        - geometry_score:           higher compactness (closer to circular) -> higher score
          (a geometric descriptor only — not a water-quality claim)
        - high_confidence_fraction: more of the body has P>=0.75 -> higher score
    """
    w = weights or RANKING_WEIGHTS
    required_keys = {"area_score", "probability_score", "geometry_score", "high_confidence_fraction"}
    if set(w.keys()) != required_keys:
        raise ValueError(f"ranking weights must have exactly these keys: {required_keys}")
    if abs(sum(w.values()) - 1.0) > 1e-6:
        raise ValueError("ranking weights must sum to 1.0")

    if not water_bodies:
        return []

    areas = [wb.area_m2 for wb in water_bodies]
    probs = [wb.mean_probability for wb in water_bodies]
    geoms = [wb.compactness for wb in water_bodies]
    confs = [wb.fraction_p75 for wb in water_bodies]

    area_norm = _minmax_normalize(areas)
    prob_norm = _minmax_normalize(probs)
    geom_norm = _minmax_normalize(geoms)
    conf_norm = _minmax_normalize(confs)

    for wb, a, p, g, c in zip(water_bodies, area_norm, prob_norm, geom_norm, conf_norm):
        score = (
            w["area_score"] * a
            + w["probability_score"] * p
            + w["geometry_score"] * g
            + w["high_confidence_fraction"] * c
        )
        wb.ranking_score = round(score * 100, 2)

    ranked = sorted(water_bodies, key=lambda wb: wb.ranking_score, reverse=True)
    for i, wb in enumerate(ranked, start=1):
        wb.rank = i

    return ranked
