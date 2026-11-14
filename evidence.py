"""Ranking and gap logic over data/tables/master_tests.csv.

rank()  -> which NASA tests are closest to a mission scenario (oxygen + pressure)
gap()   -> per pillar and fuel type: is any test actually close, or does the evidence stop?

Pure pandas, no AI. Every number it returns comes from the master table.
Written with AI assistance (Claude); see README "Use of AI".
"""
from pathlib import Path

import numpy as np
import pandas as pd

MASTER = Path(__file__).resolve().parent / "master_tests.csv"

# Mission scenarios. Keep `source` visible in the UI so judges see where each number comes from.
PRESETS = {
    "ISS cabin": dict(o2=21.0, kpa=101.3, gravity="micro",
                      source="ISS sea-level cabin air"),
    "Moon base (challenge brief)": dict(o2=30.0, kpa=70.3, gravity="lunar",
                      source="Challenge text: 30%+ O2 at 10.2 psi or less"),
    "Exploration atmosphere": dict(o2=34.0, kpa=56.5, gravity="lunar",
                      source="NASA exploration atmosphere: 34% O2 at 8.2 psi"),
    "Spacesuit": dict(o2=100.0, kpa=29.6, gravity="lunar",
                      source="Pure O2 at 4.3 psi"),
    "Mars habitat (assumed)": dict(o2=34.0, kpa=56.5, gravity="mars",
                      source="ASSUMPTION: exploration atmosphere at Mars gravity"),
}

# How many "units" of difference count as one step of distance.
# 5 O2 points is about as different as 20 kPa for flammability purposes (tunable in the UI).
O2_SCALE = 5.0
KPA_SCALE = 20.0


def load(path=MASTER):
    return pd.read_csv(path)


def rank(df, preset, pillar=None, n=10, w_o2=1.0, w_kpa=1.0,
         o2_scale=O2_SCALE, kpa_scale=KPA_SCALE):
    """Return the n tests closest to `preset` (a PRESETS key or a dict with o2, kpa, gravity).

    distance  = sqrt((w_o2*dO2/o2_scale)^2 + (w_kpa*dP/kpa_scale)^2)
    relevance = 1 / (1 + distance)    -> 1.0 means identical conditions
    """
    p = PRESETS[preset] if isinstance(preset, str) else preset
    d = df if pillar is None else df[df["pillar"] == pillar]
    d = d.dropna(subset=["oxygen_pct", "pressure_kpa"]).copy()

    d["d_o2"] = d["oxygen_pct"] - p["o2"]
    d["d_kpa"] = d["pressure_kpa"] - p["kpa"]
    d["distance"] = np.sqrt((w_o2 * d["d_o2"] / o2_scale) ** 2 + (w_kpa * d["d_kpa"] / kpa_scale) ** 2)
    d["relevance"] = (1 / (1 + d["distance"])).round(3)
    d["gravity_match"] = d["gravity"] == p["gravity"]
    d["why"] = d.apply(lambda r: _explain(r, p), axis=1)
    return d.sort_values("distance").head(n)


def _explain(r, p):
    parts = [f"O2 {r.oxygen_pct:g}% ({r.d_o2:+.1f})", f"{r.pressure_kpa:g} kPa ({r.d_kpa:+.1f})"]
    if r.gravity != p["gravity"]:
        parts.append(f"gravity {r.gravity} vs {p['gravity']}")
    if isinstance(r.assumed_fields, str) and r.assumed_fields:
        parts.append(f"nominal value used for: {r.assumed_fields}")
    return "; ".join(parts)


def _span(s, unit):
    lo, hi = s.min(), s.max()
    return f"{lo:g}{unit}" if lo == hi else f"{lo:g}-{hi:g}{unit}"


def gap(df, preset, o2_tol=3.0, kpa_tol=10.0, o2_scale=O2_SCALE, kpa_scale=KPA_SCALE):
    """For each pillar x fuel type: how many tests fall within the tolerance box, and the closest one.

    A test is 'near' only if BOTH |dO2| <= o2_tol AND |dP| <= kpa_tol.
    Returns one row per group, with status 'covered' or 'GAP'.
    `gravity_gap` is reported separately: 'covered' only means O2 and pressure match.
    """
    p = PRESETS[preset] if isinstance(preset, str) else preset
    d = df.dropna(subset=["oxygen_pct", "pressure_kpa"]).copy()
    d["d_o2"] = d["oxygen_pct"] - p["o2"]
    d["d_kpa"] = d["pressure_kpa"] - p["kpa"]
    d["near"] = (d["d_o2"].abs() <= o2_tol) & (d["d_kpa"].abs() <= kpa_tol)
    d["dist"] = np.sqrt((d["d_o2"] / o2_scale) ** 2 + (d["d_kpa"] / kpa_scale) ** 2)

    rows = []
    for (pillar, mclass), g in d.groupby(["pillar", "material_class"]):
        best = g.loc[g["dist"].idxmin()]
        rows.append({
            "pillar": pillar,
            "material_class": mclass,
            "tests_total": len(g),
            "tests_near": int(g["near"].sum()),
            "status": "covered" if g["near"].any() else "GAP",
            "closest_test": f"{best.experiment} {best.test_id}",
            "closest_o2_pct": best.oxygen_pct,
            "closest_kpa": best.pressure_kpa,
            "closest_d_o2": round(best.d_o2, 2),      # test minus scenario, percentage points
            "closest_d_kpa": round(best.d_kpa, 2),    # test minus scenario, kPa
            "closest_relevance": round(1 / (1 + best.dist), 3),
            "o2_range_tested": _span(g.oxygen_pct, "%"),
            "kpa_range_tested": _span(g.pressure_kpa, " kPa"),
            "gravity_tested": ", ".join(sorted(g["gravity"].unique())),
            "gravity_gap": p["gravity"] not in set(g["gravity"]),
        })
    return pd.DataFrame(rows).sort_values(["pillar", "material_class"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load()
    pd.set_option("display.width", 220, "display.max_columns", 20)
    for name in ["Moon base (challenge brief)", "Exploration atmosphere"]:
        print(f"\n===== {name}: {PRESETS[name]}")
        print(rank(df, name, n=5)[["experiment", "test_id", "material", "relevance", "outcome_category", "why"]].to_string(index=False))
        print(gap(df, name)[["pillar", "material_class", "tests_near", "status", "closest_test", "o2_range_tested", "kpa_range_tested", "gravity_gap"]].to_string(index=False))