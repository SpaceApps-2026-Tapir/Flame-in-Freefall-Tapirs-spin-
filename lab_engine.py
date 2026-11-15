"""Engine behind "Explore the data visually": a view spec, a design matrix, and a registry of analyses.

Everything here is plain Python/pandas/scikit-learn (no Streamlit, no LLM) so the page, the agent and the
tests all drive the same functions:

    spec = default_spec("all")
    spec = apply_tool(spec, "set_design_matrix", {...})     # what goes into the analysis
    spec = apply_tool(spec, "set_view", {...})              # how the 3D chart looks
    spec, result = run_analysis(spec, "kmeans", {"k": 4})   # compute something and colour the chart by it
    fig = make_figure(spec)

The design matrix is a table with one row per variable: use it or not, its weight, and its transform.
Written with AI assistance (Claude); see README "Use of AI".
"""
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score

from evidence import PRESETS, gap, rank

ROOT = Path(__file__).resolve().parent
SEED = 42
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
OTHER = "#9a9a94"
TRANSFORMS = ["zscore", "log", "rank", "none"]

# Moon-habitat range spanned by the atlas presets (Moon base -> exploration atmosphere)
MOON = "Moon base (challenge brief)"
REGION = dict(o2=(PRESETS[MOON]["o2"], PRESETS["Exploration atmosphere"]["o2"]),
              kpa=(PRESETS["Exploration atmosphere"]["kpa"], PRESETS[MOON]["kpa"]))

OUTCOME_GROUPS = {
    "extinction": "Flame went out", "extinguished": "Flame went out", "blowoff": "Flame went out",
    "spread": "Kept burning", "burned_entire_sample": "Kept burning", "completion": "Kept burning",
    "disruption": "Droplet broke apart", "no_ignition": "Did not ignite",
    "smoke_measured": "Smoke measured", "see_notes": "Other", "not_reported": "Other",
}

# ------------------------------------------------------------------------------------------------ datasets
DATASETS = {
    "all": dict(
        title="All 678 tests (master table)",
        numeric={"oxygen_pct": "Oxygen (%)", "pressure_kpa": "Pressure (kPa)", "airflow_cms": "Airflow (cm/s)"},
        categorical={"material_class": "Fuel type", "flow_direction": "Flow direction", "pillar": "Pillar"},
        color={"experiment": "Experiment", "pillar": "Pillar", "outcome_group": "Outcome",
               "material_class": "Fuel type"},
        default_use={"oxygen_pct", "pressure_kpa", "airflow_cms", "material_class", "flow_direction"},
        o2="oxygen_pct", kpa="pressure_kpa", id="test_id",
        hover=["experiment", "test_id", "material", "oxygen_pct", "pressure_kpa", "airflow_cms", "outcome_category"],
        default_xyz=("oxygen_pct", "pressure_kpa", "airflow_cms"),
    ),
    "flex": dict(
        title="FLEX droplet burns (raw)",
        numeric={"o2_pct": "Oxygen (%)", "co2_pct": "CO₂ (%)", "he_pct": "Helium (%)", "n2_pct": "Nitrogen (%)",
                 "pressure_kpa": "Pressure (kPa)", "d0_mm": "Initial droplet diameter (mm)",
                 "d_ext_mm": "Extinction diameter (mm)", "burning_rate_mm2s": "Burning rate K (mm²/s)",
                 "burn_time_s": "Burn time (s)", "extinction_ratio": "Extinction / initial diameter"},
        categorical={"fuel": "Fuel", "diluent": "Diluent gas"},
        color={"test_end": "How the burn ended", "fuel": "Fuel", "diluent": "Diluent gas"},
        default_use={"o2_pct", "co2_pct", "he_pct", "pressure_kpa", "d0_mm", "fuel"},
        o2="o2_pct", kpa="pressure_kpa", id="test_id",
        hover=["test_id", "fuel", "o2_pct", "co2_pct", "he_pct", "pressure_kpa", "d0_mm", "burning_rate_mm2s", "test_end"],
        default_xyz=("o2_pct", "co2_pct", "burning_rate_mm2s"),
    ),
    "same": dict(
        title="SAME / SAME-R smoke tests (raw)",
        numeric={"count_mean_diam_um": "Count mean diameter (µm)", "avg_mass_diam_um": "Average-mass diameter (µm)",
                 "sigma_g": "Size spread σg", "mass_conc_mgm3": "Smoke mass conc. (mg/m³)",
                 "flow_cms": "Inlet flow (cm/s)", "aging_s": "Aging time (s)", "sample_temp_c": "Heater temp (°C)",
                 "iss_detector_v": "ISS detector (V)", "sts_detector_v": "Shuttle detector (V)"},
        categorical={"campaign": "Campaign"},
        color={"material": "Material", "campaign": "Campaign"},
        default_use={"count_mean_diam_um", "avg_mass_diam_um", "sigma_g", "mass_conc_mgm3"},
        o2=None, kpa=None, id="test_id",
        hover=["test_id", "material", "campaign", "description", "count_mean_diam_um", "mass_conc_mgm3", "iss_detector_v"],
        default_xyz=("count_mean_diam_um", "mass_conc_mgm3", "iss_detector_v"),
    ),
}

CHARTS = {
    "space": ("Conditions in 3D", "Each dot is a test, placed by the conditions it ran at. The red box is the "
                                  "Moon-habitat range."),
    "similarity": ("Look-alike map", "Tests that ran under similar conditions sit together. Each axis is named "
                                     "after the quantities it follows most; distances between far islands mean little."),
    "flows": ("From conditions to outcome", "Each band follows a group of tests from oxygen and pressure "
                                            "to fuel and to what the flame did. Wider band = more tests."),
    "paths": ("Every test as a line", "Each line is one test, crossing every measured quantity. Drag along an "
                                      "axis to keep only part of it."),
    "mixture": ("Gas mixture triangle", "Each dot is a droplet burn placed by its gas: oxygen, nitrogen and "
                                        "suppressant (CO₂ or helium). The star is Moon-base air."),
    "evidence": ("Where the evidence is", "Click a ring to zoom in: pillar → experiment → fuel → outcome. "
                                          "Size = number of tests."),
}
CHARTS_FOR = {"all": ["flows", "space", "evidence", "similarity", "paths"],
              "flex": ["mixture", "space", "similarity", "paths"],
              "same": ["space", "paths", "similarity"]}
ANALYSIS_TITLES = {"pca_loadings": "What drives the differences", "kmeans": "Natural groups",
                   "outliers": "Unusual tests", "nearest_to_scenario": "Closest real tests",
                   "coverage_gap": "Evidence near the scenario", "correlation": "What moves together",
                   "feature_importance": "What predicts it"}

_CACHE = {}


def load(dataset):
    if dataset not in _CACHE:
        if dataset == "all":
            d = pd.read_csv(ROOT / "master_tests.csv")
            d["outcome_group"] = d.outcome_category.map(OUTCOME_GROUPS).fillna("Other")
        elif dataset == "flex":
            d = pd.read_csv(ROOT / "data" / "raw" / "flex_droplets.csv")
            d = d[d.pressure_valid]
        else:
            d = pd.read_csv(ROOT / "data" / "raw" / "same_smoke.csv")
            d = d[d.material != "DBP"]
        _CACHE[dataset] = d.reset_index(drop=True)
    return _CACHE[dataset]


def labels(dataset):
    c = DATASETS[dataset]
    return {**c["numeric"], **c["categorical"], **c["color"], "cluster": "Cluster", "outlier": "Outlier"}


# ------------------------------------------------------------------------------------------------ spec
def default_matrix(dataset):
    c = DATASETS[dataset]
    rows = [dict(variable=k, label=v, kind="numeric", use=k in c["default_use"], weight=1.0,
                 transform="log" if k in ("mass_conc_mgm3", "aging_s") else "zscore") for k, v in c["numeric"].items()]
    rows += [dict(variable=k, label=v, kind="category", use=k in c["default_use"], weight=1.0, transform="one-hot")
             for k, v in c["categorical"].items()]
    return rows


def default_spec(dataset="all"):
    c = DATASETS[dataset]
    return dict(
        dataset=dataset,
        matrix=default_matrix(dataset),
        filters={},                       # {column: [allowed values]} or {column: [min, max]}
        chart={"all": "flows", "flex": "mixture", "same": "space"}[dataset],  # open on the most telling chart
        method="conditions",              # conditions (real axes) | tsne (look-alike map)
        xyz=list(c["default_xyz"]),
        color="outcome_group" if dataset == "all" else next(iter(c["color"])),
        camera=dict(azimuth=45.0, elevation=25.0, distance=2.0),
        perplexity=30,
        highlight=None,                   # {"label": str, "index": [row ids]}
        scenario=MOON if c["o2"] else None,
        extra={},                         # columns produced by analyses (cluster, outlier...)
        last_analysis=None,
        version=0,                        # bumps when the agent moves the camera (forces Plotly to apply it)
    )


def used(spec, kind=None):
    return [r for r in spec["matrix"] if r["use"] and (kind is None or r["kind"] == kind)]


# ------------------------------------------------------------------------------------------------ data + matrix
def filtered(spec):
    d = load(spec["dataset"]).copy()
    for col, val in spec["extra"].items():
        if len(val) == len(d):
            d[col] = val
    mask = pd.Series(True, index=d.index)
    for col, rule in spec["filters"].items():
        if col not in d:
            continue
        if isinstance(rule, (list, tuple)) and len(rule) == 2 and all(isinstance(v, (int, float)) for v in rule):
            mask &= d[col].between(*rule) | d[col].isna()
        else:
            mask &= d[col].astype(str).isin([str(v) for v in rule])
    return d[mask]


def design_matrix(spec, d=None):
    """Numeric matrix from the 'use' rows: transform each numeric column, one-hot categories, multiply by weight."""
    d = filtered(spec) if d is None else d
    blocks, names = [], []
    for r in used(spec, "numeric"):
        x = pd.to_numeric(d[r["variable"]], errors="coerce").astype(float)
        x = x.fillna(x.median() if x.notna().any() else 0)
        t = r["transform"]
        if t == "log":
            x = np.log10(x.clip(lower=x[x > 0].min() if (x > 0).any() else 1e-6))
        elif t == "rank":
            x = x.rank(pct=True)
        if t in ("zscore", "log", "rank"):
            x = (x - x.mean()) / (x.std() or 1)
        blocks.append((x * float(r["weight"])).to_numpy()[:, None])
        names.append(r["variable"])
    for r in used(spec, "category"):
        oh = pd.get_dummies(d[r["variable"]].fillna("unknown").astype(str), prefix=r["variable"], dtype=float)
        blocks.append(oh.to_numpy() * float(r["weight"]))
        names += list(oh.columns)
    if not blocks:
        raise ValueError("The design matrix is empty: mark at least one variable as 'use'.")
    return np.hstack(blocks), names


def axis_names(emb, d, numeric_labels, categorical=(), min_r=0.35):
    """Name each map axis after what it follows most (rank correlation), e.g. 'more oxygen →'.

    Candidates are the numeric quantities and, for categorical columns, one indicator per value
    ('more liquid fuel droplet tests'). Each axis is led by a different candidate when possible."""
    cands = {}
    for col, lab in numeric_labels.items():
        if col in d and d[col].notna().sum() > 10 and d[col].nunique() > 2:
            cands[lab.split(" (")[0]] = pd.to_numeric(d[col], errors="coerce")
    for col in categorical:
        if col in d:
            vals = list(d[col].dropna().astype(str).value_counts().index[:6])
            for val in vals[:1] if len(vals) == 2 else vals:  # a two-value category needs only one indicator
                if val.lower() in ("not stated", "unknown", "nan", "other", "none"):
                    continue
                cands["@" + re.sub(r"\s*\(.*?\)", "", val)] = (d[col].astype(str) == val).astype(float)
    names, used_first = [], set()
    for i in range(emb.shape[1]):
        axis = pd.Series(emb[:, i], index=d.index)
        r = {}
        for lab, x in cands.items():
            v = axis.corr(x, method="spearman")
            if pd.notna(v):
                r[lab] = v
        ranked = [(k, v) for k, v in sorted(r.items(), key=lambda kv: -abs(kv[1])) if abs(v) >= min_r]
        # lead each axis with a quantity no other axis leads with, so the three names differ
        fresh = [kv for kv in ranked if kv[0] not in used_first]
        top = fresh[:1] or ranked[:1]
        if top:
            used_first.add(top[0][0])
        if not top:
            names.append("mixed conditions")
            continue
        k, v = top[0]
        if k.startswith("@"):  # a category: point to the end of the axis where it sits
            k = "@" + (k[1:] if len(k) <= 22 else k[1:21].rsplit(" ", 1)[0] + "…")
            names.append(f"{k[1:]} →" if v > 0 else f"← {k[1:]}")
        else:
            names.append(f"{'more' if v > 0 else 'less'} {k.lower()} →")
    return names


def embedding(spec, d=None):
    d = filtered(spec) if d is None else d
    m = spec["method"]
    if m == "conditions":
        return d[spec["xyz"]].to_numpy(dtype=float), [labels(spec["dataset"]).get(v, v) for v in spec["xyz"]]
    X, names = design_matrix(spec, d)
    from sklearn.manifold import TSNE
    perp = float(spec.get("perplexity", 30))
    e = TSNE(n_components=3, perplexity=min(perp, (len(X) - 1) / 3), random_state=SEED, init="pca").fit_transform(X)
    cfg = DATASETS[spec["dataset"]]
    return e, axis_names(e, d, cfg["numeric"], [r["variable"] for r in used(spec, "category")])
    raise ValueError(f"Unknown method {m}")


# ------------------------------------------------------------------------------------------------ figure
def _camera(cam):
    az, el, r = np.radians(cam["azimuth"]), np.radians(cam["elevation"]), cam["distance"]
    return dict(eye=dict(x=r * np.cos(el) * np.cos(az), y=r * np.cos(el) * np.sin(az), z=r * np.sin(el)))


def _colors(values):
    cats = sorted(pd.Series(values).astype(str).unique())
    keep = cats[:len(PALETTE)]
    mapping = {c: PALETTE[i] for i, c in enumerate(keep)}
    folded = pd.Series(values).astype(str).where(pd.Series(values).astype(str).isin(keep), "Other")
    mapping["Other"] = OTHER
    return folded, mapping


def make_figure(spec, height=640):
    d = filtered(spec)
    if d.empty:
        raise ValueError("No tests pass the current filters.")
    chart = spec.get("chart", "space")
    if chart in ("flows", "paths", "mixture", "evidence"):
        fig = {"flows": _fig_flows, "paths": _fig_paths, "mixture": _fig_mixture, "evidence": _fig_evidence}[chart](spec, d)
        fig.update_layout(height=height, margin=dict(l=40, r=40, t=40, b=20))
        return fig
    e, axes = embedding(spec, d)
    lab = labels(spec["dataset"])
    cfg = DATASETS[spec["dataset"]]
    p = d.assign(_x=e[:, 0], _y=e[:, 1], _z=e[:, 2])
    col = spec["color"] if spec["color"] in p else next(iter(cfg["color"]))
    p[col], cmap = _colors(p[col])
    hover = {h: True for h in cfg["hover"] if h in p}
    hover.update({"_x": False, "_y": False, "_z": False})
    fig = px.scatter_3d(p, x="_x", y="_y", z="_z", color=col, color_discrete_map=cmap, hover_data=hover,
                        labels={**lab, "_x": axes[0], "_y": axes[1], "_z": axes[2]}, opacity=0.85)
    fig.update_traces(marker=dict(size=4, line=dict(width=0.4, color="rgba(255,255,255,.6)")))

    # Moon-habitat box, only when the real oxygen and pressure axes are on screen
    if spec["method"] == "conditions" and cfg["o2"] in spec["xyz"] and cfg["kpa"] in spec["xyz"]:
        fig.add_trace(_region_box(spec, p))

    hl = spec.get("highlight")
    if hl and hl.get("index"):
        h = p.loc[p.index.intersection(hl["index"])]
        fig.add_trace(go.Scatter3d(x=h._x, y=h._y, z=h._z, mode="markers", name=hl["label"],
                                   marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=4, color="#111")),
                                   hoverinfo="skip"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.02, title=None, itemsizing="constant"),
                      scene=dict(aspectmode="cube", camera=_camera(spec["camera"])),
                      uirevision=f"{spec['dataset']}-{spec['method']}-{spec['version']}")
    return fig


def _band(x, edges, names):
    return pd.cut(x, edges, labels=names).astype(str).replace("nan", "unknown")


def _discrete_scale(n):
    cols = (PALETTE + [OTHER] * 10)[:max(n, 1)]
    if n <= 1:
        return [[0, cols[0]], [1, cols[0]]]
    return [[i / (n - 1), c] for i, c in enumerate(cols)]


def _fig_flows(spec, d):
    cfg = DATASETS[spec["dataset"]]
    p = pd.DataFrame(index=d.index)
    if cfg["o2"]:
        p["Oxygen"] = _band(d[cfg["o2"]], [0, 18, 22, 28, 101], ["<18%", "18–22% (air)", "22–28%", "≥28% (Moon-like)"])
        p["Pressure"] = _band(d[cfg["kpa"]], [0, 80, 110, 1e4], ["reduced (<80 kPa)", "sea level", "high"])
    cats = [c for c in cfg["categorical"] if c in d][:1]
    for c in cats:
        p[labels(spec["dataset"])[c]] = d[c].fillna("unknown").astype(str)
    col = spec["color"] if spec["color"] in d else next(iter(cfg["color"]))
    folded, cmap = _colors(d[col])
    order = [k for k in cmap if k in set(folded)]
    p[labels(spec["dataset"]).get(col, col)] = folded.values
    codes = folded.map({k: i for i, k in enumerate(order)}).values
    fig = go.Figure(go.Parcats(
        dimensions=[dict(label=c, values=p[c]) for c in p.columns],
        line=dict(color=codes, colorscale=[[i / max(len(order) - 1, 1), cmap[k]] for i, k in enumerate(order)]
                  if len(order) > 1 else [[0, cmap[order[0]]], [1, cmap[order[0]]]], shape="hspline"),
        hoveron="color", arrangement="freeform", labelfont=dict(size=13)))
    return fig


def _fig_paths(spec, d):
    cfg, lab = DATASETS[spec["dataset"]], labels(spec["dataset"])
    nums = [r["variable"] for r in used(spec, "numeric")] or list(cfg["numeric"])
    nums = [c for c in nums if d[c].notna().sum() > 5][:7]
    col = spec["color"] if spec["color"] in d else next(iter(cfg["color"]))
    folded, cmap = _colors(d[col])
    order = [k for k in cmap if k in set(folded)]
    codes = folded.map({k: i for i, k in enumerate(order)}).values
    dims = [dict(label=lab.get(c, c), values=d[c].fillna(d[c].median())) for c in nums]
    fig = go.Figure(go.Parcoords(line=dict(color=codes, colorscale=_discrete_scale(len(order))
                                           if len(order) > 1 else [[0, cmap[order[0]]], [1, cmap[order[0]]]]),
                                 dimensions=dims, labelangle=-20))
    fig.add_annotation(x=0, y=-0.12, xref="paper", yref="paper", showarrow=False, xanchor="left",
                       text="  ".join(f"<span style='color:{cmap[k]}'>━</span> {k}" for k in order))
    return fig


def _fig_mixture(spec, d):
    lab = labels(spec["dataset"])
    p = d.assign(suppressant=d.co2_pct + d.he_pct)
    col = spec["color"] if spec["color"] in p else "test_end"
    p[col], cmap = _colors(p[col])
    fig = px.scatter_ternary(p, a="o2_pct", b="n2_pct", c="suppressant", color=col, color_discrete_map=cmap,
                             hover_data=["test_id", "fuel", "o2_pct", "co2_pct", "he_pct", "pressure_kpa", "test_end"],
                             labels={"o2_pct": "Oxygen", "n2_pct": "Nitrogen", "suppressant": "CO₂ + helium",
                                     **lab})
    fig.update_traces(marker=dict(size=8, line=dict(width=0.6, color="white")))
    fig.add_trace(go.Scatterternary(a=[REGION["o2"][0]], b=[100 - REGION["o2"][0]], c=[0], mode="markers+text",
                                    text=["Moon base"], textposition="top right", name="Moon-base air",
                                    marker=dict(symbol="star", size=18, color="#d03b3b", line=dict(color="white", width=1))))
    fig.update_layout(ternary=dict(aaxis_title="Oxygen %", baxis_title="Nitrogen %", caxis_title="CO₂ + He %"),
                      legend=dict(orientation="h", y=-0.08))
    return fig


def _fig_evidence(spec, d):
    p = d.assign(n=1)
    for c in ("pillar", "experiment", "material_class", "outcome_group"):
        p[c] = p[c].fillna("unknown").astype(str)
    fig = px.sunburst(p, path=["pillar", "experiment", "material_class", "outcome_group"], values="n",
                      color="pillar", color_discrete_map={"prevention": PALETTE[0], "detection": PALETTE[1],
                                                          "suppression": PALETTE[2]})
    fig.update_traces(insidetextorientation="radial", hovertemplate="%{label}: %{value} tests<extra></extra>")
    return fig


def _region_box(spec, p):
    cfg = DATASETS[spec["dataset"]]
    ranges = {}
    for axis, v in zip("xyz", spec["xyz"]):
        if v == cfg["o2"]:
            ranges[axis] = REGION["o2"]
        elif v == cfg["kpa"]:
            ranges[axis] = REGION["kpa"]
        else:
            col = p[f"_{axis}"]
            ranges[axis] = (float(col.min()), float(col.max()))
    (x0, x1), (y0, y1), (z0, z1) = ranges["x"], ranges["y"], ranges["z"]
    return go.Mesh3d(x=[x0, x1, x1, x0, x0, x1, x1, x0], y=[y0, y0, y1, y1, y0, y0, y1, y1],
                     z=[z0, z0, z0, z0, z1, z1, z1, z1],
                     i=[0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3], j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6],
                     k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7], color="#d03b3b", opacity=0.15,
                     name="Moon-habitat range", showlegend=True, hoverinfo="name")


# ------------------------------------------------------------------------------------------------ analyses
def _a_pca_loadings(spec, d, params):
    X, names = design_matrix(spec, d)
    p = PCA(n_components=min(3, X.shape[1]), random_state=SEED).fit(X)
    load_ = pd.DataFrame(p.components_.T, index=names, columns=[f"PC{i + 1}" for i in range(p.n_components_)])
    top = {c: load_[c].abs().sort_values(ascending=False).head(3).index.tolist() for c in load_}
    text = "; ".join(f"{c} ({v:.0%} of variance) is driven by {', '.join(top[c])}"
                     for c, v in zip(load_.columns, p.explained_variance_ratio_))
    return dict(summary=text, table=load_.round(3).reset_index(names="variable"))


def _a_kmeans(spec, d, params):
    k = int(params.get("k", 4))
    X, _ = design_matrix(spec, d)
    lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(X)
    full = pd.Series(np.nan, index=load(spec["dataset"]).index, dtype=object)
    full.loc[d.index] = [f"Cluster {i + 1}" for i in lab]
    spec["extra"]["cluster"] = full.tolist()
    spec["color"] = "cluster"
    by = next(iter(DATASETS[spec["dataset"]]["color"]))
    comp = pd.crosstab(full.loc[d.index], d[by])
    text = f"{k} clusters. " + "; ".join(
        f"{c}: mostly {r.idxmax()} ({r.max() / r.sum():.0%} of {int(r.sum())})" for c, r in comp.iterrows())
    return dict(summary=text, table=comp.reset_index(names="cluster"))


def _a_outliers(spec, d, params):
    X, _ = design_matrix(spec, d)
    frac = float(params.get("fraction", 0.05))
    score = -IsolationForest(contamination=frac, random_state=SEED).fit(X).score_samples(X)
    out = d.assign(anomaly_score=score.round(3)).sort_values("anomaly_score", ascending=False)
    n = max(1, int(round(frac * len(d))))
    spec["highlight"] = dict(label=f"{n} most unusual tests", index=out.index[:n].tolist())
    cols = [c for c in DATASETS[spec["dataset"]]["hover"] if c in out] + ["anomaly_score"]
    return dict(summary=f"The {n} tests least like the others, given the current design matrix.",
                table=out[cols].head(n))


def _a_nearest(spec, d, params):
    if params.get("o2") is not None and params.get("kpa") is not None:  # a flame from the simulator
        cfg = DATASETS[spec["dataset"]]
        if not cfg["o2"]:
            raise ValueError("This dataset has no oxygen/pressure.")
        o2, kpa = float(params["o2"]), float(params["kpa"])
        pool = d[d["fuel"] == params["fuel"]] if params.get("fuel") and "fuel" in d else d
        dist = np.hypot((pool[cfg["o2"]] - o2) / 3, (pool[cfg["kpa"]] - kpa) / 10)
        near = pool.assign(distance=dist.round(3)).sort_values("distance").head(int(params.get("n", 15)))
        spec["highlight"] = dict(label="Closest to your flame", index=near.index.tolist())
        cols = [c for c in cfg["hover"] if c in near] + ["distance"]
        return dict(summary=f"{len(near)} NASA burns closest to your flame ({o2:g}% O₂, {kpa:g} kPa).",
                    table=near[cols])
    scen = params.get("scenario") or spec.get("scenario") or MOON
    if spec["dataset"] != "all":
        cfg = DATASETS[spec["dataset"]]
        if not cfg["o2"]:
            raise ValueError("This dataset has no oxygen/pressure: switch to 'all' or 'flex'.")
        p = PRESETS[scen]
        dist = np.hypot((d[cfg["o2"]] - p["o2"]) / 5, (d[cfg["kpa"]] - p["kpa"]) / 20)
        near = d.assign(distance=dist.round(3)).sort_values("distance").head(int(params.get("n", 15)))
    else:
        near = rank(d, scen, n=int(params.get("n", 15)))
    spec["scenario"] = scen
    spec["highlight"] = dict(label=f"Closest to {scen.split(' (')[0]}", index=near.index.tolist())
    cols = [c for c in DATASETS[spec["dataset"]]["hover"] if c in near] + \
           [c for c in ("relevance", "distance", "why") if c in near]
    return dict(summary=f"{len(near)} tests closest to {scen} (by oxygen and pressure).", table=near[cols])


def _a_coverage(spec, d, params):
    scen = params.get("scenario") or spec.get("scenario") or MOON
    g = gap(load("all"), scen)
    by = g.groupby("pillar").agg(tests_near=("tests_near", "sum"), tests_total=("tests_total", "sum"),
                                 best_relevance=("closest_relevance", "max")).reset_index()
    text = "; ".join(f"{r.pillar}: {r.tests_near} of {r.tests_total} tests near {scen}" for r in by.itertuples())
    return dict(summary=text, table=g)


def _a_correlation(spec, d, params):
    cols = [r["variable"] for r in used(spec, "numeric")]
    if len(cols) < 2:
        raise ValueError("Mark at least two numeric variables as 'use'.")
    c = d[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    pairs = (c.where(np.triu(np.ones(c.shape, bool), 1)).stack().sort_values(key=abs, ascending=False).head(3))
    lab = labels(spec["dataset"])
    text = "Strongest links: " + "; ".join(f"{lab.get(a, a)} ↔ {lab.get(b, b)} ({v:+.2f})" for (a, b), v in pairs.items())
    fig = go.Figure(go.Heatmap(z=c.values, x=[lab.get(v, v) for v in cols], y=[lab.get(v, v) for v in cols],
                               zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                               text=c.round(2).values, texttemplate="%{text}"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    return dict(summary=text, figure=fig)


def _a_importance(spec, d, params):
    target = params.get("target") or spec["color"]
    if target not in d or target in ("cluster",):
        raise ValueError(f"Unknown target '{target}'. Use one of the colour columns.")
    y = d[target].astype(str).to_numpy(dtype=object)
    X, names = design_matrix(spec, d)
    if len(set(y)) < 2:
        raise ValueError("The target has a single value in the current selection.")
    cv = StratifiedKFold(min(5, pd.Series(y).value_counts().min()), shuffle=True, random_state=SEED) \
        if pd.Series(y).value_counts().min() >= 2 else 3
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, class_weight="balanced")
    acc = cross_val_score(rf, X, y, cv=cv).mean()
    rf.fit(X, y)
    imp = permutation_importance(rf, X, y, n_repeats=8, random_state=SEED).importances_mean
    s = pd.Series(imp, index=names)
    # fold one-hot columns back to their variable
    s = s.groupby([next((r["variable"] for r in used(spec) if n.startswith(r["variable"])), n) for n in names]).sum()
    s = s.sort_values(ascending=False)
    lab = labels(spec["dataset"])
    base = pd.Series(y).value_counts(normalize=True).max()
    text = (f"Predicting {lab.get(target, target)}: cross-validated accuracy {acc:.0%} (guessing: {base:.0%}). "
            f"Most informative: {', '.join(lab.get(v, v) for v in s.index[:3])}.")
    fig = go.Figure(go.Bar(x=s.values[::-1], y=[lab.get(v, v) for v in s.index[::-1]], orientation="h",
                           marker_color=PALETTE[0]))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Importance")
    return dict(summary=text, figure=fig)


ANALYSES = {
    "pca_loadings": (_a_pca_loadings, "Which variables drive the main directions of variation (PCA).", {}),
    "kmeans": (_a_kmeans, "Group the tests into k clusters using the design matrix; colours the chart by cluster.",
               {"k": "number of clusters, 2-8"}),
    "outliers": (_a_outliers, "Find the most unusual tests (isolation forest) and ring them on the chart.",
                 {"fraction": "share of tests to flag, e.g. 0.05"}),
    "nearest_to_scenario": (_a_nearest, "Ring the tests closest to a mission scenario (oxygen + pressure).",
                            {"scenario": "one of the scenario names", "n": "how many tests"}),
    "coverage_gap": (_a_coverage, "How many tests per pillar are near a scenario (evidence gap).",
                     {"scenario": "one of the scenario names"}),
    "correlation": (_a_correlation, "Spearman correlation between the numeric variables in use.", {}),
    "feature_importance": (_a_importance, "Which variables in the matrix best predict a category "
                                          "(cross-validated random forest).",
                           {"target": "a colour column, e.g. outcome_group, test_end, material"}),
}


def run_analysis(spec, name, params=None):
    if name not in ANALYSES:
        raise ValueError(f"Unknown analysis '{name}'. Available: {', '.join(ANALYSES)}")
    spec = deepcopy(spec)
    d = filtered(spec)
    result = ANALYSES[name][0](spec, d, params or {})
    result["name"] = name
    result["title"] = ANALYSIS_TITLES.get(name, name)
    spec["last_analysis"] = {k: v for k, v in result.items() if k in ("name", "summary")}
    return spec, result


# ------------------------------------------------------------------------------------------------ tools
def apply_tool(spec, tool, args):
    """Apply one agent/UI action to a copy of the spec and return it. Raises ValueError on bad input."""
    spec = deepcopy(spec)
    if tool == "set_design_matrix":
        if args.get("dataset") and args["dataset"] != spec["dataset"]:
            if args["dataset"] not in DATASETS:
                raise ValueError(f"dataset must be one of {list(DATASETS)}")
            spec = default_spec(args["dataset"])
        rows = {r["variable"]: r for r in spec["matrix"]}
        if args.get("only"):
            for r in spec["matrix"]:
                r["use"] = r["variable"] in args["only"]
        for ch in args.get("variables", []):
            v = ch.get("variable")
            if v not in rows:
                raise ValueError(f"Unknown variable '{v}'. Available: {', '.join(rows)}")
            if "use" in ch:
                rows[v]["use"] = bool(ch["use"])
            if "weight" in ch:
                rows[v]["weight"] = float(np.clip(float(ch["weight"]), 0, 10))
            if "transform" in ch and rows[v]["kind"] == "numeric":
                if ch["transform"] not in TRANSFORMS:
                    raise ValueError(f"transform must be one of {TRANSFORMS}")
                rows[v]["transform"] = ch["transform"]
        if "filters" in args:
            spec["filters"] = {k: v for k, v in (args["filters"] or {}).items() if v not in (None, [], "")}
        spec["extra"], spec["highlight"] = {}, None
    elif tool == "set_view":
        cfg = DATASETS[spec["dataset"]]
        if args.get("chart"):
            if args["chart"] not in CHARTS_FOR[spec["dataset"]]:
                raise ValueError(f"chart for this dataset must be one of {CHARTS_FOR[spec['dataset']]}")
            spec["chart"] = args["chart"]
            if args["chart"] == "flows" and spec["dataset"] == "all" and not args.get("color_by"):
                spec["color"] = "outcome_group"  # the last band should be what the flame did
            if args["chart"] == "space":
                spec["method"] = "conditions"
            elif args["chart"] == "similarity":
                spec["method"] = "tsne"
        if args.get("method"):
            if args["method"] not in ("conditions", "tsne"):
                raise ValueError("method must be conditions or tsne")
            spec["method"] = args["method"]
            if not args.get("chart"):
                spec["chart"] = "space" if args["method"] == "conditions" else "similarity"
        for i, k in enumerate(("x", "y", "z")):
            if args.get(k):
                if args[k] not in cfg["numeric"]:
                    raise ValueError(f"{k} must be a numeric variable: {', '.join(cfg['numeric'])}")
                spec["xyz"][i] = args[k]
        if args.get("color_by"):
            if args["color_by"] not in {**cfg["color"], "cluster": 1} or (
                    args["color_by"] == "cluster" and "cluster" not in spec["extra"]):
                raise ValueError(f"color_by must be one of {', '.join(cfg['color'])}")
            spec["color"] = args["color_by"]
        cam = spec["camera"]
        moved = False
        if args.get("rotate_degrees") is not None:
            cam["azimuth"] = (cam["azimuth"] + float(args["rotate_degrees"])) % 360
            moved = True
        if args.get("tilt_degrees") is not None:
            cam["elevation"] = float(np.clip(cam["elevation"] + float(args["tilt_degrees"]), -85, 85))
            moved = True
        if args.get("zoom") is not None:
            cam["distance"] = float(np.clip(cam["distance"] / float(args["zoom"]), 0.6, 5))
            moved = True
        if args.get("preset_view"):
            cam.update({"top": dict(azimuth=0, elevation=85), "front": dict(azimuth=-90, elevation=0),
                        "side": dict(azimuth=0, elevation=0),
                        "default": dict(azimuth=45, elevation=25, distance=2.0)}[args["preset_view"]])
            moved = True
        if moved:
            spec["version"] += 1
        if args.get("perplexity"):
            spec["perplexity"] = float(np.clip(args["perplexity"], 5, 50))
        if args.get("clear_highlight"):
            spec["highlight"] = None
    else:
        raise ValueError(f"Unknown tool {tool}")
    return spec


def describe(spec):
    """Compact state the agent sees: dataset, variables, filters, view and data summary."""
    d = filtered(spec)
    cfg = DATASETS[spec["dataset"]]
    num = d[[c for c in cfg["numeric"] if c in d]].describe().loc[["min", "max"]].round(2).to_dict()
    return dict(
        dataset=spec["dataset"], datasets_available={k: v["title"] for k, v in DATASETS.items()},
        rows_shown=len(d), rows_total=len(load(spec["dataset"])),
        design_matrix=[{k: r[k] for k in ("variable", "label", "kind", "use", "weight", "transform")}
                       for r in spec["matrix"]],
        filters=spec["filters"], chart=spec["chart"],
        charts_available={k: CHARTS[k][1] for k in CHARTS_FOR[spec["dataset"]]},
        method=spec["method"], axes=spec["xyz"], color_by=spec["color"],
        color_options=list(cfg["color"]) + (["cluster"] if "cluster" in spec["extra"] else []),
        numeric_ranges=num,
        category_values={c: sorted(d[c].dropna().astype(str).unique())[:12] for c in cfg["color"] if c in d},
        scenarios=list(PRESETS), last_analysis=spec.get("last_analysis"),
        analyses={k: {"what": v[1], "params": v[2]} for k, v in ANALYSES.items()},
    )
