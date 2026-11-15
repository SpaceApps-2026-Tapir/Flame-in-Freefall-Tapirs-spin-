"""Data analysis page: interactive 3D views and machine-learning models over the raw NASA measurements.

Three tabs:
  * Droplet flames (FLEX)      - 3D explorer + random-forest model of burning rate / extinction size,
                                 with a 3D response surface the user can slice.
  * Smoke and detectors (SAME) - 3D explorer + "can the smoke tell what is burning?" classifier,
                                 PCA / k-means view of smoke signatures.
  * All tests in 3D            - PCA or t-SNE embedding of all 678 tests, nearest tests to a mission scenario.

Data: data/raw/flex_droplets.csv and data/raw/same_smoke.csv (built by scripts/extract_raw_data.py)
and master_tests.csv. Every model reports cross-validated scores next to a naive baseline, so the
page never shows a fit without saying how well it generalises.
Written with AI assistance (Claude); see README "Use of AI".
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, confusion_matrix, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evidence import PRESETS, rank

RAW = Path(__file__).resolve().parent / "data" / "raw"
SEED = 42

# Categorical palette (validated for colour-vision deficiency, fixed order). A 6th+ category folds into "Other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
OTHER = "#9a9a94"
CHART_H = 620


# ---------------------------------------------------------------------------- data
@st.cache_data
def load_flex():
    d = pd.read_csv(RAW / "flex_droplets.csv")
    return d[d.pressure_valid].reset_index(drop=True)


@st.cache_data
def load_same():
    d = pd.read_csv(RAW / "same_smoke.csv")
    return d[d.material != "DBP"].reset_index(drop=True)  # DBP is the reference aerosol, not a spacecraft material


FLEX_LABELS = {
    "o2_pct": "Oxygen (%)", "co2_pct": "CO₂ (%)", "he_pct": "Helium (%)", "n2_pct": "Nitrogen (%)",
    "pressure_kpa": "Pressure (kPa)", "d0_mm": "Initial droplet diameter (mm)",
    "d_ext_mm": "Diameter at flame extinction (mm)", "burning_rate_mm2s": "Burning rate K (mm²/s)",
    "burn_time_s": "Burn time (s)", "extinction_ratio": "Extinction diameter / initial diameter",
}
FLEX_CATS = {"fuel": "Fuel", "diluent": "Diluent gas", "test_end": "How the burn ended"}
FLEX_FEATURES = ["o2_pct", "co2_pct", "he_pct", "pressure_kpa", "d0_mm", "is_heptane"]
FLEX_TARGETS = ["burning_rate_mm2s", "d_ext_mm", "extinction_ratio", "burn_time_s"]

SAME_LABELS = {
    "count_mean_diam_um": "Count mean diameter (µm)", "avg_mass_diam_um": "Average-mass diameter (µm)",
    "geo_mean_diam_um": "Geometric mean diameter (µm)", "sigma_g": "Size spread σg",
    "mass_conc_mgm3": "Smoke mass concentration (mg/m³)", "flow_cms": "Inlet flow (cm/s)",
    "aging_s": "Aging time (s)", "sample_temp_c": "Heater temperature (°C)",
    "iss_detector_v": "ISS smoke detector signal (V)", "sts_detector_v": "Shuttle smoke detector signal (V)",
    "mass_loss_mg": "Sample mass lost (mg)",
}
SAME_SMOKE = ["count_mean_diam_um", "avg_mass_diam_um", "sigma_g", "mass_conc_mgm3"]
SAME_CONDITIONS = ["flow_cms", "aging_s", "sample_temp_c"]


def color_map(values):
    """Fixed palette by sorted category; anything past the 5th becomes 'Other' (grey)."""
    cats = sorted(pd.Series(values).dropna().astype(str).unique())
    keep = cats[:len(PALETTE)]
    return {c: PALETTE[i] for i, c in enumerate(keep)}, set(keep)


def fold_other(s):
    _, keep = color_map(s)
    return s.astype(str).where(s.astype(str).isin(keep), "Other")


def style(fig, height=CHART_H, scene=True):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(orientation="h", y=-0.02, x=0, title=None, itemsizing="constant"))
    if scene:
        fig.update_layout(scene=dict(aspectmode="cube"))
    return fig


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value, help_text) in zip(cols, items):
        c.metric(label, value, help=help_text)


def method_note(text):
    with st.expander("Method and limits"):
        st.markdown(text)


def explorer_3d(df, labels, cats, defaults, key, hover):
    """Generic 3D scatter with axis / colour pickers."""
    num = [c for c in labels if c in df.columns and df[c].notna().sum() > 5]
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("X axis", num, num.index(defaults[0]), format_func=labels.get, key=f"{key}_x")
    y = c2.selectbox("Y axis", num, num.index(defaults[1]), format_func=labels.get, key=f"{key}_y")
    z = c3.selectbox("Z axis", num, num.index(defaults[2]), format_func=labels.get, key=f"{key}_z")
    color = c4.selectbox("Colour by", list(cats), format_func=cats.get, key=f"{key}_c")
    log = st.toggle("Log scale on all axes", False, key=f"{key}_log",
                    help="Useful when values span several orders of magnitude (smoke sizes and concentrations).")

    d = df.dropna(subset=[x, y, z]).copy()
    d[color] = fold_other(d[color])
    cmap, _ = color_map(d[color])
    cmap["Other"] = OTHER
    fig = px.scatter_3d(d, x=x, y=y, z=z, color=color, color_discrete_map=cmap,
                        labels={**labels, **cats}, hover_data={h: True for h in hover if h in d.columns},
                        log_x=log, log_y=log, log_z=log, opacity=0.85)
    fig.update_traces(marker=dict(size=4.5, line=dict(width=0.5, color="rgba(255,255,255,.6)")))
    st.plotly_chart(style(fig), theme="streamlit", width="stretch", key=f"{key}_fig")
    st.caption(f"{len(d)} of {len(df)} tests have all three values. Drag to rotate, scroll to zoom, "
               "double-click a legend entry to isolate it.")


# ---------------------------------------------------------------------------- FLEX models
@st.cache_resource
def flex_model(target):
    d = load_flex().dropna(subset=FLEX_FEATURES[:-1] + [target]).copy()
    d["is_heptane"] = (d.fuel == "Heptane").astype(int)
    X, y = d[FLEX_FEATURES].values, d[target].values
    cv = KFold(5, shuffle=True, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred_rf = cross_val_predict(rf, X, y, cv=cv)
    pred_lin = cross_val_predict(lin, X, y, cv=cv)

    # permutation importance measured on held-out folds only (no optimistic in-sample scores)
    imps = []
    for tr, te in cv.split(X):
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        imps.append(permutation_importance(m, X[te], y[te], n_repeats=10, random_state=SEED).importances_mean)
    rf.fit(X, y)
    return dict(model=rf, data=d, pred=pred_rf, r2_rf=r2_score(y, pred_rf), r2_lin=r2_score(y, pred_lin),
                importance=pd.Series(np.mean(imps, axis=0), index=FLEX_FEATURES).sort_values())


FEATURE_NAMES = {**FLEX_LABELS, "is_heptane": "Fuel (heptane vs methanol)"}


def flex_tab():
    d = load_flex()
    st.markdown(f"**{len(d)} droplet burns** from FLEX (PSI-69). Each row is one fuel droplet ignited in "
                "microgravity, with the gas mixture around it and what happened to the flame.")

    st.subheader("3D explorer")
    explorer_3d(d, FLEX_LABELS, FLEX_CATS, ("o2_pct", "co2_pct", "burning_rate_mm2s"), "flex",
                ["test_id", "fuel", "o2_pct", "co2_pct", "he_pct", "pressure_kpa", "d0_mm", "d_ext_mm",
                 "burning_rate_mm2s", "test_end"])

    st.subheader("What controls the flame? A machine-learning model")
    target = st.selectbox("Quantity to model", FLEX_TARGETS, format_func=FLEX_LABELS.get, key="flex_target")
    m = flex_model(target)
    metric_row([
        ("Random forest R²", f"{m['r2_rf']:.2f}", "Cross-validated (5 folds): each prediction comes from a model "
                                                  "that never saw that burn. 1 = perfect, 0 = no better than the mean."),
        ("Linear model R²", f"{m['r2_lin']:.2f}", "Same cross-validation with a straight-line (ridge) model, "
                                                  "as a baseline. A big gap means the effects are non-linear."),
        ("Burns used", f"{len(m['data'])}", "Burns with every input and the selected quantity recorded."),
    ])
    if m["r2_rf"] < 0.3:
        st.info("The model explains little of this quantity: read the surface below as a rough trend only.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        imp = m["importance"].rename(index=FEATURE_NAMES)
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PALETTE[0],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(height=330, margin=dict(l=0, r=10, t=30, b=0),
                          title=dict(text="Which inputs matter most", font=dict(size=14)),
                          xaxis_title="Drop in R² when the input is shuffled")
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="flex_imp")
    with right:
        dd = m["data"].assign(predicted=m["pred"])
        lim = [min(dd[target].min(), dd.predicted.min()), max(dd[target].max(), dd.predicted.max())]
        fig = px.scatter(dd, x=target, y="predicted", color="fuel",
                         color_discrete_map=color_map(dd.fuel)[0], labels={**FLEX_LABELS, "predicted": "Predicted"},
                         hover_data=["test_id", "o2_pct", "co2_pct", "he_pct", "pressure_kpa"])
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color=OTHER),
                                 name="Perfect prediction", hoverinfo="skip"))
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0),
                          title=dict(text="Measured vs predicted (held-out)", font=dict(size=14)),
                          legend=dict(orientation="h", y=-0.25, title=None))
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="flex_pred")

    st.markdown("#### Response surface")
    st.caption("Pick two inputs for the floor of the plot; the model predicts the height. Fix the others with the "
               "sliders. Dots are real burns; the surface far from any dot is extrapolation.")
    drivers = ["o2_pct", "co2_pct", "he_pct", "pressure_kpa", "d0_mm"]
    a, b, c = st.columns(3)
    sx = a.selectbox("Surface X", drivers, 0, format_func=FLEX_LABELS.get, key="surf_x")
    sy = b.selectbox("Surface Y", [v for v in drivers if v != sx], 0, format_func=FLEX_LABELS.get, key="surf_y")
    fuel = c.radio("Fuel", ["Methanol", "Heptane"], horizontal=True, key="surf_fuel")
    data = m["data"]
    fixed = {}
    others = [v for v in drivers if v not in (sx, sy)]
    for col, v in zip(st.columns(len(others)), others):
        lo, hi = float(data[v].min()), float(data[v].max())
        default = 0.0 if v in ("co2_pct", "he_pct") else float(data[v].median())
        if v == "pressure_kpa":
            col.caption("Moon base (challenge brief): ~70 kPa")
        fixed[v] = col.slider(FLEX_LABELS[v], lo, hi, min(max(default, lo), hi), key=f"fix_{v}")

    gx = np.linspace(data[sx].min(), data[sx].max(), 35)
    gy = np.linspace(data[sy].min(), data[sy].max(), 35)
    GX, GY = np.meshgrid(gx, gy)
    grid = pd.DataFrame({sx: GX.ravel(), sy: GY.ravel(), **{k: v for k, v in fixed.items()},
                         "is_heptane": int(fuel == "Heptane")})[FLEX_FEATURES]
    Z = m["model"].predict(grid.values).reshape(GX.shape)
    pts = data[data.fuel == fuel]
    fig = go.Figure([
        go.Surface(x=gx, y=gy, z=Z, colorscale="Blues", opacity=0.85, showscale=True,
                   colorbar=dict(title=dict(text=FLEX_LABELS[target].split(" (")[0], side="right"), len=0.6),
                   hovertemplate=f"{FLEX_LABELS[sx]}: %{{x:.1f}}<br>{FLEX_LABELS[sy]}: %{{y:.1f}}<br>"
                                 f"Predicted: %{{z:.2f}}<extra></extra>"),
        go.Scatter3d(x=pts[sx], y=pts[sy], z=pts[target], mode="markers", name=f"{fuel} burns (measured)",
                     marker=dict(size=3.5, color=PALETTE[1], line=dict(width=0.5, color="white")),
                     customdata=pts[["test_id", "pressure_kpa"]].values,
                     hovertemplate="%{customdata[0]} · %{customdata[1]:.0f} kPa<br>measured %{z:.2f}<extra></extra>"),
    ])
    fig.update_layout(scene=dict(xaxis_title=FLEX_LABELS[sx], yaxis_title=FLEX_LABELS[sy],
                                 zaxis_title=FLEX_LABELS[target], aspectmode="cube"),
                      height=CHART_H, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=0))
    st.plotly_chart(fig, theme="streamlit", width="stretch", key="flex_surface")

    method_note(f"""
* **Model:** random forest (400 trees, ≥3 burns per leaf) on oxygen, CO₂, helium, pressure, initial droplet
  size and fuel. Nitrogen is left out because it is simply the rest of the gas (it is 100 % minus the others).
* **Scores** are 5-fold cross-validated: every number comes from a model that did not see that burn.
  Importance is permutation importance measured on the held-out folds.
* **Limits:** {len(data)} burns, two fuels, droplets only. A random forest predicts in steps and cannot
  extrapolate: outside the tested ranges the surface goes flat. If the linear R² is as high as the forest's,
  the effect is close to a straight line and the simpler model is the better explanation. FLEX tells us about liquid fuels and gas suppressants, not about solid materials.
* **Data:** `data/raw/flex_droplets.csv`, built from `PSI-69_Experimental table_FLEX.csv`. Burns that record
  0 mmHg pressure are excluded.
""")


# ---------------------------------------------------------------------------- SAME models
@st.cache_resource
def same_classifier(features):
    d = load_same().dropna(subset=["count_mean_diam_um"]).copy()
    X = d[list(features)].copy()
    if "mass_conc_mgm3" in X:
        X["mass_conc_mgm3"] = np.log10(X["mass_conc_mgm3"])
    y = d.material.astype(str).to_numpy(dtype=object)
    X = X.to_numpy(dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=SEED,
                                              class_weight="balanced", n_jobs=-1))
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    p_rf = cross_val_predict(rf, X, y, cv=cv)
    p_lr = cross_val_predict(lr, X, y, cv=cv)
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y, p_rf, labels=labels, normalize="true")
    chance = pd.Series(y).value_counts(normalize=True).max()
    return dict(acc_rf=(p_rf == y).mean(), acc_lr=(p_lr == y).mean(), chance=chance, cm=cm, labels=labels, n=len(d))


@st.cache_data
def same_embedding(features, k):
    d = load_same().dropna(subset=["count_mean_diam_um"]).copy()
    X = d[list(features)].copy()
    if "mass_conc_mgm3" in X:
        X["mass_conc_mgm3"] = np.log10(X["mass_conc_mgm3"])
    Xs = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)
    pca = PCA(n_components=3, random_state=SEED).fit(Xs)
    emb = pca.transform(Xs)
    clusters = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xs)
    d[["pc1", "pc2", "pc3"]] = emb
    d["cluster"] = [f"Cluster {c + 1}" for c in clusters]
    return d, pca.explained_variance_ratio_, adjusted_rand_score(d.material.astype(str).to_numpy(dtype=object), clusters)


def same_tab():
    d = load_same()
    st.markdown(f"**{len(d)} smoke tests** from SAME and SAME-R (PSI-101/102). Spacecraft materials were heated "
                "until they smoked; instruments measured the particles and two smoke detectors (ISS and Shuttle) "
                "recorded their response. The DBP reference aerosol is left out.")

    st.subheader("3D explorer")
    explorer_3d(d, SAME_LABELS, {"material": "Material", "campaign": "Campaign"},
                ("flow_cms", "aging_s", "count_mean_diam_um"), "same",
                ["test_id", "material", "campaign", "description", "count_mean_diam_um", "mass_conc_mgm3",
                 "iss_detector_v"])

    st.subheader("Would a detector notice it?")
    st.caption("Detector signal against particle size. Bubble size = smoke mass concentration. "
               "Light-scattering detectors respond weakly to very small particles.")
    det = st.radio("Detector", ["iss_detector_v", "sts_detector_v"], format_func=SAME_LABELS.get,
                   horizontal=True, key="same_det")
    dd = d.dropna(subset=["count_mean_diam_um", det, "mass_conc_mgm3"])
    fig = px.scatter(dd, x="count_mean_diam_um", y=det, color="material", size="mass_conc_mgm3", size_max=28,
                     color_discrete_map=color_map(dd.material)[0], log_x=True, labels=SAME_LABELS,
                     hover_data=["test_id", "campaign", "description", "mass_conc_mgm3"])
    sat = dd[det].max()
    fig.add_hline(y=sat, line=dict(dash="dot", color=OTHER),
                  annotation_text="Signal ceiling: points here saturated the detector", annotation_position="bottom right")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2, title=None))
    st.plotly_chart(fig, theme="streamlit", width="stretch", key="same_det_fig")

    st.subheader("Can the smoke tell us what is burning?")
    use_cond = st.toggle("Also give the model the test settings (flow, aging time, heater temperature)", False,
                         key="same_cond",
                         help="Off: the model only sees what a sensor could measure (particle sizes, spread, "
                              "concentration). On: it also sees how the test was run, which makes it easier "
                              "but less realistic.")
    feats = tuple(SAME_SMOKE + (SAME_CONDITIONS if use_cond else []))
    c = same_classifier(feats)
    metric_row([
        ("Random forest accuracy", f"{c['acc_rf']:.0%}", "5-fold stratified cross-validation."),
        ("Logistic regression", f"{c['acc_lr']:.0%}", "Linear baseline, same cross-validation."),
        ("Guessing the most common", f"{c['chance']:.0%}", "What you get by always answering the most frequent material."),
        ("Tests used", f"{c['n']}", "Tests with a measured particle size."),
    ])

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        fig = go.Figure(go.Heatmap(z=c["cm"], x=c["labels"], y=c["labels"], colorscale="Blues", zmin=0, zmax=1,
                                   text=np.round(c["cm"] * 100).astype(int), texttemplate="%{text}%",
                                   hovertemplate="Real: %{y}<br>Predicted: %{x}<br>%{z:.0%}<extra></extra>",
                                   colorbar=dict(title="Share", tickformat=".0%")))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                          title=dict(text="Confusion matrix (held-out)", font=dict(size=14)),
                          xaxis_title="Predicted material", yaxis_title="Real material",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="same_cm")
    with right:
        k = st.slider("k-means clusters", 2, 8, 5, key="same_k",
                      help="Groups the tests by smoke signature alone, without being told the material.")
        e, var, ari = same_embedding(feats, k)
        by = st.radio("Colour by", ["material", "cluster"], horizontal=True, key="same_pca_c",
                      format_func={"material": "Real material", "cluster": "k-means cluster"}.get)
        fig = px.scatter_3d(e, x="pc1", y="pc2", z="pc3", color=by, color_discrete_map=color_map(e[by])[0],
                            labels={"pc1": f"PC1 ({var[0]:.0%})", "pc2": f"PC2 ({var[1]:.0%})",
                                    "pc3": f"PC3 ({var[2]:.0%})"},
                            hover_data=["test_id", "material", "cluster", "description"])
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(style(fig, height=420), theme="streamlit", width="stretch", key="same_pca")
        st.caption(f"PCA of the smoke signature ({sum(var):.0%} of the variance in 3 axes). "
                   f"Agreement between clusters and real materials (adjusted Rand index): {ari:.2f} "
                   "(0 = random, 1 = identical).")

    method_note("""
* **Classifier:** random forest (400 trees, balanced class weights) vs logistic regression, 5-fold stratified
  cross-validation. Mass concentration is log-transformed; missing values are filled with the median.
* **Reading it:** if accuracy is well above *guessing the most common*, smoke particles carry a fingerprint of
  the burning material, which a smarter detector could use. Confusions show which materials look alike.
* **Limits:** about 130 tests, five materials, all at 21 % O₂ and ~101 kPa. Some material groups come mostly
  from one campaign (SAME or SAME-R), so part of the signal may be the campaign, not the material.
* **Data:** `data/raw/same_smoke.csv`, from the sheet *Reduced Data with ISS Det* of
  `Combined SAMER Results 06 02 14.xlsx` (PSI-101).
""")


# ---------------------------------------------------------------------------- all tests
OUTCOME_GROUPS = {
    "extinction": "Flame went out", "extinguished": "Flame went out", "blowoff": "Flame went out",
    "spread": "Flame kept burning", "burned_entire_sample": "Flame kept burning", "completion": "Flame kept burning",
    "disruption": "Droplet broke apart", "no_ignition": "Did not ignite",
    "smoke_measured": "Smoke measured", "see_notes": "Other / see notes", "not_reported": "Other / see notes",
}
MASTER_COLOR = {"experiment": "Experiment", "pillar": "Pillar", "outcome_group": "Outcome",
                "material_class": "Fuel type"}


CAT_COLS = ["material_class", "flow_direction"]
NUM_COLS = ["oxygen_pct", "pressure_kpa", "airflow_cms"]


def _features(df):
    num = df[NUM_COLS].copy()
    med = num.median()
    scaler = StandardScaler().fit(num.fillna(med))
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[CAT_COLS].fillna("unknown"))

    def tf(d):
        return np.hstack([scaler.transform(d[NUM_COLS].fillna(med)) * 1.5,  # physical conditions weigh a bit more
                          enc.transform(d[CAT_COLS].fillna("unknown"))])
    return tf


@st.cache_resource
def umap_all(df, n_neighbors, min_dist):
    import umap  # heavy import (numba): only when this view is opened
    tf = _features(df)
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=SEED)
    emb = reducer.fit_transform(tf(df))
    return emb, reducer, tf


def neighbour_mixing(emb, labels, k=10):
    """Share of each experiment's k nearest neighbours (in the embedding) that belong to each other experiment."""
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]
    lab = np.asarray(labels)
    nb = pd.DataFrame({"own": np.repeat(lab, k), "other": lab[idx.ravel()]})
    return pd.crosstab(nb.own, nb.other, normalize="index")


@st.cache_data
def embed_all(df, method, perplexity):
    d = df.copy()
    X = _features(d)(d)
    if method == "PCA":
        model = PCA(n_components=3, random_state=SEED)
        emb = model.fit_transform(X)
        axes = [f"PC{i + 1} ({v:.0%})" for i, v in enumerate(model.explained_variance_ratio_)]
    else:
        emb = TSNE(n_components=3, perplexity=perplexity, random_state=SEED, init="pca").fit_transform(X)
        axes = ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
    return emb, axes


def all_tab(df):
    st.markdown(f"All **{len(df)} tests** in 3D: tests run under similar conditions (oxygen, pressure, airflow, "
                "fuel type, flow direction) end up close together. The experiment name is **not** an input, so "
                "if two experiments mix, they really tested the same conditions.")
    a, b, c, e = st.columns([1.2, 1.2, 1, 1.4])
    method = a.radio("Method", ["UMAP", "t-SNE", "PCA"], horizontal=True, key="all_m",
                     help="UMAP and t-SNE keep neighbourhoods: groups are meaningful, distances between far groups "
                          "are not. PCA is a plain rotation: axes keep their meaning.")
    if method == "UMAP":
        nn = b.slider("Neighbours", 5, 50, 15, key="all_nn",
                      help="Low = fine local detail, many islands. High = broader structure.")
        md = b.slider("Minimum distance", 0.0, 0.9, 0.3, 0.05, key="all_md", help="Low = tight clumps.")
    elif method == "t-SNE":
        perp = b.slider("Perplexity", 5, 50, 30, key="all_p")
    color = c.selectbox("Colour by", list(MASTER_COLOR), format_func=MASTER_COLOR.get, key="all_c")
    scen = e.selectbox("Place a mission scenario", ["(none)"] + list(PRESETS), index=2, key="all_s")

    base = df.reset_index(drop=True)
    d = base.copy()
    d["outcome_group"] = d.outcome_category.map(OUTCOME_GROUPS).fillna("Other / see notes")
    if method == "UMAP":
        emb, reducer, tf = umap_all(base, nn, md)
        axes = ["UMAP 1", "UMAP 2", "UMAP 3"]
    else:
        emb, axes = embed_all(base, method, perp if method == "t-SNE" else 30)
    d[["e1", "e2", "e3"]] = emb
    d[color] = fold_other(d[color])
    cmap, _ = color_map(d[color])
    cmap["Other"] = OTHER

    fig = px.scatter_3d(d, x="e1", y="e2", z="e3", color=color, color_discrete_map=cmap,
                        labels={"e1": axes[0], "e2": axes[1], "e3": axes[2], **MASTER_COLOR},
                        hover_data={"experiment": True, "test_id": True, "material": True, "oxygen_pct": True,
                                    "pressure_kpa": True, "outcome_category": True, "e1": False, "e2": False, "e3": False},
                        opacity=0.8)
    fig.update_traces(marker=dict(size=3.5))
    if scen != "(none)":
        p = PRESETS[scen]
        near = rank(base, scen, n=15)
        h = d.loc[near.index]
        fig.add_trace(go.Scatter3d(x=h.e1, y=h.e2, z=h.e3, mode="markers", name="15 closest tests",
                                   marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(width=3, color="#111")),
                                   customdata=np.c_[h.experiment, h.test_id, near.relevance],
                                   hovertemplate="%{customdata[0]} %{customdata[1]}<br>relevance %{customdata[2]}"
                                                 "<extra></extra>"))
        if method == "UMAP":
            # place the scenario itself: its O2 / pressure with the fuel type and flow of the tests closest to it
            row = pd.DataFrame([{"oxygen_pct": p["o2"], "pressure_kpa": p["kpa"], "airflow_cms": np.nan,
                                 **{c_: near[c_].mode().iat[0] for c_ in CAT_COLS}}])
            sp = reducer.transform(tf(row))[0]
            fig.add_trace(go.Scatter3d(x=[sp[0]], y=[sp[1]], z=[sp[2]], mode="markers+text", name=scen,
                                       text=[scen.split(" (")[0]], textposition="top center",
                                       marker=dict(size=11, symbol="diamond", color="#d03b3b",
                                                   line=dict(width=2, color="white")),
                                       hovertemplate=f"{scen}<br>{p['o2']:g}% O₂ · {p['kpa']:g} kPa<extra></extra>"))
    st.plotly_chart(style(fig, height=680), theme="streamlit", width="stretch", key="all_fig")

    mix = neighbour_mixing(emb, base.experiment)
    pairs = []
    for x_ in mix.index:
        for y_ in mix.columns:
            if x_ < y_ and (mix.at[x_, y_] + mix.get(x_, pd.Series()).get(y_, 0)) > 0:
                share = (mix.at[x_, y_] + (mix.at[y_, x_] if y_ in mix.index and x_ in mix.columns else 0)) / 2
                if share >= 0.05:
                    pairs.append((share, x_, y_))
    if pairs:
        st.markdown("**What the map shows:** " + " · ".join(
            f"{x_} and {y_} share **{s_:.0%}** of their nearest neighbours" for s_, x_, y_ in sorted(pairs, reverse=True))
            + ". Every other pair barely touches: each experiment explored its own island of conditions.")
    method_note("""
* **Inputs:** oxygen, pressure and airflow (standardised, weighted ×1.5) plus one-hot fuel type and flow
  direction. Missing airflow is filled with the median. Experiment and pillar are *not* inputs.
* **Neighbour sharing:** for each test, its 10 nearest neighbours in the 3D map; the share that belongs to the
  other experiment, averaged both ways.
* **Scenario diamond (UMAP only):** the scenario's oxygen and pressure, with the fuel type and flow direction most
  common among its 15 closest tests, projected into the trained map. **t-SNE** and **PCA** show only the rings.
* UMAP and t-SNE: the size of a group and the gap between far groups mean nothing.
""")


# ---------------------------------------------------------------------------- page
def page(df, header=True):
    st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
  .block-container {max-width: 1250px;}
  h1 {font-size: 2.2rem !important; border-bottom: 1px solid var(--atlas-border); padding-bottom: .2rem; margin-bottom: .2rem;}
  .tagline {color: var(--atlas-muted); font-size: .9rem; margin-bottom: 1rem;}
</style>""", unsafe_allow_html=True)
    if header:
        st.markdown("<a href='?' target='_self' style='text-decoration:none'>← Flame in Freefall</a>", unsafe_allow_html=True)
        st.markdown("<h1>Data analysis</h1>", unsafe_allow_html=True)
        st.markdown("<div class='tagline'>Interactive 3D views and machine-learning models over the raw NASA "
                    "measurements. Every model is scored on data it did not see.</div>", unsafe_allow_html=True)

    if not (RAW / "flex_droplets.csv").exists():
        st.error("Missing data/raw/*.csv. Run: python scripts/extract_raw_data.py \"<NASA RESOURCES>\" data/raw")
        return
    t1, t2, t3 = st.tabs(["Droplet flames (FLEX)", "Smoke and detectors (SAME)", "All tests in 3D"])
    with t1:
        flex_tab()
    with t2:
        same_tab()
    with t3:
        all_tab(df)
