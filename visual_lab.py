"""Explore the data visually: charts, editable design matrix and local analysis buttons.

Route: ?page=visual. Chart controls and analysis buttons edit the same lab_engine spec.
Written with AI assistance (Claude); see README "Use of AI".
"""

import pandas as pd
import streamlit as st

import lab_engine as L

METHODS = {"conditions": "Real axes", "tsne": "Look-alike map"}


def _spec():
    if "lab_spec" not in st.session_state:
        st.session_state.lab_spec = L.default_spec("all")
        st.session_state.lab_mver = 0
        st.session_state.lab_result = None
    # arriving from the landing page / guide: ?page=visual&data=flex&chart=mixture&q=...
    qp = st.query_params
    key = (qp.get("data"), qp.get("chart"), qp.get("q"), qp.get("o2"), qp.get("kpa"), qp.get("fuel"))
    if any(key) and st.session_state.get("lab_qp") != key:
        st.session_state.lab_qp = key
        spec = L.default_spec(qp.get("data")) if qp.get("data") in L.DATASETS else st.session_state.lab_spec
        if qp.get("chart") in L.CHARTS_FOR[spec["dataset"]]:
            spec = L.apply_tool(spec, "set_view", {"chart": qp["chart"]})
        _set(spec, matrix_changed=True)
        st.session_state.lab_result = None
        if qp.get("o2") and qp.get("kpa"):  # arriving from the simulator: ring the burns closest to that flame
            _analyse("nearest_to_scenario", {"o2": qp["o2"], "kpa": qp["kpa"], "fuel": qp.get("fuel")})
    return st.session_state.lab_spec


def _set(spec, matrix_changed=False):
    st.session_state.lab_spec = spec
    if matrix_changed:
        st.session_state.lab_mver += 1


def _do(tool, args):
    try:
        old = st.session_state.lab_spec
        new = L.apply_tool(old, tool, args)
        _set(new, matrix_changed=new["matrix"] != old["matrix"] or new["dataset"] != old["dataset"])
    except ValueError as e:
        st.toast(str(e))


def _analyse(name, params=None):
    try:
        spec, res = L.run_analysis(st.session_state.lab_spec, name, params or {})
        _set(spec)
        st.session_state.lab_result = res
    except ValueError as e:
        st.session_state.lab_result = {"name": name, "summary": f"Could not run: {e}"}


# ------------------------------------------------------------------------------------------------ pieces
DATA_NAMES = {"all": "All 678 tests", "flex": "Droplet burns (FLEX)", "same": "Smoke tests (SAME)"}


def controls(spec):
    cfg = L.DATASETS[spec["dataset"]]
    c1, c2, c3 = st.columns([1, 3.6, 1])
    ds = c1.selectbox("Data", list(L.DATASETS), list(L.DATASETS).index(spec["dataset"]), format_func=DATA_NAMES.get)
    if ds != spec["dataset"]:
        _do("set_design_matrix", {"dataset": ds})
        st.session_state.lab_result = None
        st.rerun()
    charts = L.CHARTS_FOR[spec["dataset"]]
    ch = c2.segmented_control("Chart", charts, default=spec["chart"], format_func=lambda k: L.CHARTS[k][0],
                              key=f"chart_{spec['dataset']}_{spec['chart']}")
    opts = list(cfg["color"]) + (["cluster"] if "cluster" in spec["extra"] else [])
    col = c3.selectbox("Colour", opts, opts.index(spec["color"]) if spec["color"] in opts else 0,
                       format_func=lambda k: L.labels(spec["dataset"]).get(k, k))
    changes = {}
    if ch and ch != spec["chart"]:
        changes["chart"] = ch
    if col != spec["color"]:
        changes["color_by"] = col
    if changes:
        _do("set_view", changes)
        st.rerun()


def camera_buttons(spec):
    if spec["chart"] not in ("space", "similarity"):
        return
    b = st.columns([1, 1, 1, 1, 8])
    if b[0].button(":material/rotate_right:", help="Rotate 45°"):
        _do("set_view", {"rotate_degrees": 45}); st.rerun()
    if b[1].button(":material/vertical_align_top:", help="Top view"):
        _do("set_view", {"preset_view": "top"}); st.rerun()
    if b[2].button(":material/crop_landscape:", help="Front view"):
        _do("set_view", {"preset_view": "front"}); st.rerun()
    if b[3].button(":material/restart_alt:", help="Reset view and clear rings"):
        _do("set_view", {"preset_view": "default", "clear_highlight": True}); st.rerun()


def advanced(spec):
    """Axes, map settings and the design matrix: everything technical, folded away."""
    cfg, lab = L.DATASETS[spec["dataset"]], L.labels(spec["dataset"])
    with st.expander("Advanced: axes, map settings and design matrix"):
        changes = {}
        if spec["chart"] == "space":
            a = st.columns(3)
            for i, (k, w) in enumerate(zip("xyz", a)):
                v = w.selectbox(f"{k.upper()} axis", list(cfg["numeric"]), list(cfg["numeric"]).index(spec["xyz"][i]),
                                format_func=lab.get)
                if v != spec["xyz"][i]:
                    changes[k] = v
        if spec["chart"] == "similarity":
            pp = st.slider("Neighbourhood size", 5, 50, int(spec.get("perplexity", 30)),
                           help="How many neighbours each test looks at when placing itself. Low = many small "
                                "islands; high = broader groups.")
            if pp != spec.get("perplexity", 30):
                changes["perplexity"] = pp
        if changes:
            _do("set_view", changes)
            st.rerun()
        matrix_editor(spec)


def matrix_editor(spec):
    st.markdown("##### Design matrix")
    st.caption("What goes into the look-alike map and every analysis. Tick a variable, give it more weight to make "
               "it count more, or change its transform.")
    df = pd.DataFrame(spec["matrix"])[["use", "label", "weight", "transform", "kind", "variable"]]
    edited = st.data_editor(
        df, key=f"dm_{spec['dataset']}_{st.session_state.lab_mver}", hide_index=True, width="stretch",
        disabled=["label", "kind", "variable"],
        column_config={
            "use": st.column_config.CheckboxColumn("Use", width="small"),
            "label": st.column_config.TextColumn("Variable"),
            "weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=10.0, step=0.5, width="small"),
            "transform": st.column_config.SelectboxColumn("Transform", options=L.TRANSFORMS + ["one-hot"],
                                                          width="small"),
            "kind": st.column_config.TextColumn("Type", width="small"),
            "variable": None})
    rows = edited.to_dict(orient="records")
    changes = [dict(variable=r["variable"], use=bool(r["use"]), weight=float(r["weight"] or 0),
                    **({"transform": r["transform"]} if r["kind"] == "numeric" and r["transform"] in L.TRANSFORMS else {}))
               for r, o in zip(rows, spec["matrix"])
               if (r["use"], r["weight"], r["transform"]) != (o["use"], o["weight"], o["transform"])]
    if changes:
        spec2 = L.apply_tool(spec, "set_design_matrix", {"variables": changes})
        _set(spec2)
        st.rerun()
    if spec["filters"]:
        st.caption("Filters: " + "; ".join(f"{k} = {v}" for k, v in spec["filters"].items()))
        if st.button("Clear filters"):
            _do("set_design_matrix", {"filters": {}}); st.rerun()


def analysis_buttons(spec):
    with st.expander("Look deeper: groups, unusual tests, what predicts what"):
        b = st.columns(3) + st.columns(3)
        k = st.session_state.get("lab_k", 4)
        if b[0].button("What drives it", width="stretch", help="Which variables drive the main directions of variation"):
            _analyse("pca_loadings")
        if b[1].button(f"Natural groups ({k})", width="stretch"):
            _analyse("kmeans", {"k": k})
        if b[2].button("Unusual tests", width="stretch"):
            _analyse("outliers", {"fraction": 0.05})
        if b[3].button("Closest to Moon base", width="stretch", disabled=L.DATASETS[spec["dataset"]]["o2"] is None):
            _analyse("nearest_to_scenario", {"scenario": L.MOON})
        if b[4].button("What moves together", width="stretch"):
            _analyse("correlation")
        if b[5].button("What predicts colour", width="stretch", help="Which variables best predict the colour category"):
            _analyse("feature_importance", {"target": spec["color"]})
        st.slider("Clusters (k)", 2, 8, 4, key="lab_k", label_visibility="collapsed")


def show_result():
    r = st.session_state.get("lab_result")
    if not r:
        return
    with st.container(border=True):
        st.markdown(f"**{r.get('title', r['name'])}** — {r['summary']}")
        if r.get("figure") is not None:
            st.plotly_chart(r["figure"], theme="streamlit", width="stretch", key="lab_res_fig")
        if isinstance(r.get("table"), pd.DataFrame):
            st.dataframe(r["table"], hide_index=True, width="stretch", height=min(320, 38 + 35 * len(r["table"])))


# ------------------------------------------------------------------------------------------------ page
def page(header=True):
    st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
  .block-container {max-width: 1400px;}
  h1 {font-size: 2.2rem !important; border-bottom: 1px solid var(--atlas-border); padding-bottom: .2rem; margin-bottom: .2rem;}
  .tagline {color: var(--atlas-muted); font-size: .9rem; margin-bottom: .8rem;}
</style>""", unsafe_allow_html=True)
    if header:
        st.markdown("<a href='?' target='_self' style='text-decoration:none'>← Flame in Freefall</a>", unsafe_allow_html=True)
        st.markdown("<h1>Explore the data visually</h1>", unsafe_allow_html=True)
        st.markdown("<div class='tagline'>Choose a dataset and chart, then explore with the controls below.</div>",
                    unsafe_allow_html=True)

    spec = _spec()
    controls(spec)
    spec = st.session_state.lab_spec
    try:
        with st.spinner("Drawing…"):
            fig = L.make_figure(spec)
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="lab_fig")
        hl = spec.get("highlight")
        st.caption(L.CHARTS[spec["chart"]][1] + f" · {len(L.filtered(spec))} tests"
                   + (f" · rings: {hl['label']}" if hl and spec["chart"] in ("space", "similarity") else ""))
    except ValueError as e:
        st.warning(str(e))
    camera_buttons(spec)
    analysis_buttons(spec)
    show_result()
    advanced(st.session_state.lab_spec)
