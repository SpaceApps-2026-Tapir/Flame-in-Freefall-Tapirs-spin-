from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evidence import KPA_SCALE, O2_SCALE, PRESETS, gap, load, rank

st.set_page_config(page_title="Flame in Freefall", page_icon="🔥", layout="wide")

SERIF = "'Linux Libertine', 'Georgia', 'Times New Roman', serif"
BASE_CSS = f"""
<style>
  h1, h2, .stMarkdown h1, .stMarkdown h2 {{font-family: {SERIF} !important; font-weight: 400 !important;}}
  a {{color: #3366cc;}}
  .placeholder {{border: 1px dashed #a2a9b1; background: #fbfbfb; color: #72777d; padding: 1.2rem; text-align: center;}}
  footer {{visibility: hidden;}}
</style>"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# Original emblem: a spherical microgravity flame (blue core, soot-orange shell)
EMBLEM = """
<svg width="190" height="190" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-label="Spherical flame">
  <defs>
    <radialGradient id="shell" cx="50%" cy="50%" r="50%">
      <stop offset="45%" stop-color="#ffe7c2" stop-opacity="0"/>
      <stop offset="70%" stop-color="#f6a04d" stop-opacity=".85"/>
      <stop offset="88%" stop-color="#e2611f" stop-opacity=".95"/>
      <stop offset="100%" stop-color="#b23a12" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="core" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="55%" stop-color="#9cc3ff"/>
      <stop offset="100%" stop-color="#3366cc" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="92" fill="url(#shell)"/>
  <circle cx="100" cy="100" r="58" fill="url(#core)"/>
  <circle cx="100" cy="100" r="9" fill="#2a2a2a"/>
</svg>"""


@st.cache_data
def get_data():
    return load()


df = get_data()
qp = st.query_params


def link(label, sub, **params):
    url = "?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    return f"<a class='portal' href='{url}' target='_self'><b>{label}</b><br><span>{sub}</span></a>"


# ====================================================================== HOME (portal)
def home():
    n = df["experiment"].value_counts()
    st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
  .block-container {max-width: 960px; padding-top: 3rem;}
  .title {text-align:center; font-family: """ + SERIF + """; font-size: 2.9rem; margin: .2rem 0 0; color:#202122;}
  .subtitle {text-align:center; font-family: """ + SERIF + """; font-size: 1.05rem; color:#54595d; margin-bottom: 1.4rem;}
  .portal, .portal * {text-decoration:none !important;}
  .portal:hover b {text-decoration: underline !important;}
  .portal {display:block; color:#3366cc !important; line-height:1.25; margin: .55rem 0;}
  .portal b {font-size: 1.15rem; font-weight: 700;}
  .portal span {color:#54595d; font-size: .85rem;}
  .left {text-align:right;} .right {text-align:left;}
  .emblem {display:flex; justify-content:center;}
  div[data-testid="stForm"] {border: none; padding: 0;}
  [data-testid="stFormSubmitButton"] button {background:#3366cc; color:white; border:none; border-radius:2px; width:100%;}
  [data-testid="stFormSubmitButton"] button:hover {background:#447ff5; color:white;}
  .divider {border-top: 1px solid #c8ccd1; margin: 1.6rem auto 1rem; width: 75%;}
  .langs {text-align:center; font-size: .92rem; line-height: 2;}
  .langs a {margin: 0 .6rem; text-decoration:none;}
  .note {text-align:center; color:#72777d; font-size:.78rem; margin-top: 1.5rem;}
</style>""", unsafe_allow_html=True)

    st.markdown("<div class='title'>Flame in Freefall</div>"
                "<div class='subtitle'>The Fire Safety Evidence Atlas</div>", unsafe_allow_html=True)

    # emblem with portal links around it: pillars on the left, every mission scenario on the right
    left, mid, right = st.columns([1.2, 1, 1.2], vertical_alignment="center")
    with left:
        st.markdown("<div class='left'>"
                    + link("Prevention", f"BASS · BASS-II · {n.get('BASS', 0) + n.get('BASS-II', 0)} tests", page="article", pillar="prevention")
                    + link("Detection", f"SAME · SAME-R · {n.get('SAME', 0) + n.get('SAME-R', 0)} tests", page="article", pillar="detection")
                    + link("Suppression", f"FLEX · {n.get('FLEX', 0)} tests", page="article", pillar="suppression")
                    + "</div>", unsafe_allow_html=True)
    with mid:
        st.markdown(f"<div class='emblem'>{EMBLEM}</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='right'>"
                    + "".join(link(name.replace(" (challenge brief)", ""),
                                   f"{s['o2']:g}% O₂ · {s['kpa'] / 6.895:.1f} psi", page="article", scenario=name)
                              for name, s in PRESETS.items())
                    + "</div>", unsafe_allow_html=True)

    # search bar: one question box + blue button (the scenario is chosen in the article's sidebar)
    _, c, _ = st.columns([0.6, 3, 0.6])
    with c:
        with st.form("search", border=False):
            s1, s2 = st.columns([3, 0.55], vertical_alignment="bottom")
            q = s1.text_input("Question", placeholder="Ask the evidence, e.g. Do flames go out on their own at 30% oxygen?",
                              label_visibility="collapsed")
            go = s2.form_submit_button("🔍")
        if go:
            st.query_params.update({"page": "article", "q": q})
            st.rerun()
        st.markdown("<div class='langs'><a href='?page=explore' target='_self'>Or filter all "
                    f"{len(df)} tests in the data explorer →</a></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='note'>{len(df)} microgravity fire tests from NASA Physical Sciences Informatics · "
                "Built for NASA Space Apps Challenge 2026 · Not affiliated with or endorsed by NASA</div>",
                unsafe_allow_html=True)


# ====================================================================== ARTICLE (Wikipedia-style page)
SECTIONS = [
    ("overview", "Overview"), ("scenario", "Mission scenario"), ("evidence", "Closest NASA evidence"),
    ("gaps", "Evidence gaps"), ("map", "Test conditions map"), ("experiments", "Experiments"),
    ("ask", "Ask the evidence"), ("earth", "Applications on Earth"),
    ("method", "Methodology and use of AI"), ("references", "References"),
]


def heading(anchor):
    st.markdown(f'<h2 id="{anchor}">{dict(SECTIONS)[anchor]}</h2>', unsafe_allow_html=True)


def placeholder(text):
    st.markdown(f'<div class="placeholder">{text}</div>', unsafe_allow_html=True)


# Chart colours: one fixed hue per pillar (colour-blind-safe set), shape per experiment.
PILLAR_COLORS = {"prevention": "#2a78d6", "detection": "#eb6834", "suppression": "#1baf7a"}
EXPERIMENT_SYMBOLS = {"BASS": "circle", "BASS-II": "diamond", "SAME": "circle", "SAME-R": "diamond", "FLEX": "circle"}
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def evidence_table(ranked, p):
    """Closest NASA tests to the scenario, straight from rank()."""
    if p["gravity"] not in set(df["gravity"]):
        st.warning(f"No test in this atlas was run at **{p['gravity']}** gravity "
                   f"(all {len(df)} are {', '.join(sorted(df['gravity'].unique()))}gravity). "
                   "Tests below are ranked on oxygen and pressure only.", icon="⚠️")
    if ranked.empty:
        st.info("No tests with oxygen and pressure values for this pillar.")
        return
    cols = ["experiment", "test_id", "material", "oxygen_pct", "pressure_kpa", "airflow_cms", "suppressant",
            "outcome_category", "relevance", "why", "source_location"]
    st.dataframe(ranked[cols], hide_index=True, width="stretch", column_config={
        "experiment": "Experiment",
        "test_id": "Test",
        "material": "Material",
        "oxygen_pct": st.column_config.NumberColumn("O₂ %", format="%.1f"),
        "pressure_kpa": st.column_config.NumberColumn("kPa", format="%.1f"),
        "airflow_cms": st.column_config.NumberColumn("Airflow cm/s", format="%.1f"),
        "suppressant": "Suppressant",
        "outcome_category": "Outcome",
        "relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1, format="%.2f",
                                                     help="1 / (1 + distance); 1.00 = identical O₂ and pressure"),
        "why": "Why it ranks here",
        "source_location": "Source",
    })
    st.caption("Relevance compares oxygen and pressure only, not material, airflow or gravity. "
               "Pick a test below to see the original NASA note and full source.")

    labels = [f"#{i} · {r.experiment} {r.test_id} · {r.material} · {r.oxygen_pct:g}% O₂, {r.pressure_kpa:g} kPa"
              for i, r in enumerate(ranked.itertuples(), 1)]
    pick = st.selectbox("Show details for test", range(len(ranked)), format_func=lambda i: labels[i])
    test_details(ranked.iloc[pick])


def test_details(r):
    """Detail panel for one test: the NASA note behind the outcome category, and where it comes from."""
    def val(x, unit=""):
        return "not recorded" if pd.isna(x) else (f"{x:g}{unit}" if isinstance(x, float) else str(x))

    with st.container(border=True):
        st.markdown(f"#### {r.experiment} {r.test_id}: {r.material}")
        a, b = st.columns(2)
        a.markdown(f"**Pillar:** {r.pillar}  \n**Fuel type:** {r.material_class}  \n"
                   f"**Oxygen:** {val(r.oxygen_pct, '%')}  \n**Pressure:** {val(r.pressure_kpa, ' kPa')}  \n"
                   f"**Gravity:** {r.gravity}")
        b.markdown(f"**Airflow:** {val(r.airflow_cms, ' cm/s')} ({val(r.flow_direction)})  \n"
                   f"**Suppressant:** {val(r.suppressant)}  \n**Outcome (our category):** {r.outcome_category}  \n"
                   f"**Relevance:** {r.relevance:.2f}")
        st.markdown(f"**Original NASA note:** {val(r.outcome_raw)}")
        st.markdown(f"**Test conditions as logged:** {val(r.condition_detail)}")
        if not pd.isna(r.assumed_fields):
            st.warning(f"Nominal value filled in for: {r.assumed_fields} (not logged per test).", icon="ℹ️")
        st.caption(f"Source: {r.source_file} · {r.source_location}")


def gap_banners(gaps, o2_tol, kpa_tol):
    """One banner per pillar x fuel type: covered, covered except gravity, or GAP."""
    def verdict(r):
        return "gap" if r.status == "GAP" else ("gravity" if r.gravity_gap else "covered")

    gaps = gaps.assign(verdict=[verdict(r) for r in gaps.itertuples()])
    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Covered", int((gaps.verdict == "covered").sum()))
    m2.metric("⚠️ Covered except gravity", int((gaps.verdict == "gravity").sum()))
    m3.metric("❌ Gap", int((gaps.verdict == "gap").sum()))
    st.markdown(f"<div class='hatnote'>A pillar × fuel type is covered when at least one test is within "
                f"±{o2_tol:g} percentage points of O₂ <b>and</b> ±{kpa_tol:g} kPa of the scenario.</div>",
                unsafe_allow_html=True)

    for pillar_name, g in gaps.groupby("pillar", sort=False):
        st.markdown(f"**{pillar_name.capitalize()}**")
        for r in g.itertuples():
            msg = (f"**{r.material_class}**: {r.tests_near} of {r.tests_total} tests within tolerance. "
                   f"Closest: {r.closest_test} at {r.closest_o2_pct:g}% O₂, {r.closest_kpa:g} kPa "
                   f"(**{r.closest_d_o2:+.1f} pp O₂, {r.closest_d_kpa:+.1f} kPa** from the scenario). "
                   f"Tested: {r.o2_range_tested} O₂, {r.kpa_range_tested}, gravity: {r.gravity_tested}.")
            if r.verdict == "gap":
                over_o2 = max(0.0, abs(r.closest_d_o2) - o2_tol)
                over_kpa = max(0.0, abs(r.closest_d_kpa) - kpa_tol)
                outside = " and ".join(t for t in [f"{over_o2:.1f} pp O₂" if over_o2 else "",
                                                   f"{over_kpa:.1f} kPa" if over_kpa else ""] if t)
                st.error(msg + f" It lies outside the tolerance box by {outside}.", icon="❌")
            elif r.verdict == "gravity":
                st.warning(msg + " Oxygen and pressure match, gravity does not.", icon="⚠️")
            else:
                st.success(msg, icon="✅")


def conditions_map(pts, preset, p, o2_tol, kpa_tol):
    """Every test as a dot (O₂ × pressure), the scenario as a star inside its tolerance box."""
    pts = pts.dropna(subset=["oxygen_pct", "pressure_kpa"])
    fig = px.scatter(
        pts, x="oxygen_pct", y="pressure_kpa", color="pillar", symbol="experiment",
        color_discrete_map=PILLAR_COLORS, symbol_map=EXPERIMENT_SYMBOLS,
        hover_name="test_id",
        hover_data={"experiment": True, "material": True, "outcome_category": True, "airflow_cms": ":.1f",
                    "oxygen_pct": ":.1f", "pressure_kpa": ":.1f", "pillar": False},
        labels={"oxygen_pct": "Oxygen (%)", "pressure_kpa": "Pressure (kPa)", "outcome_category": "outcome",
                "airflow_cms": "airflow (cm/s)", "pillar": "Pillar", "experiment": "Experiment"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.7, line=dict(width=1.5, color=SURFACE)))
    if p is not None:  # no scenario chosen yet: just the tests, no star or tolerance box
        fig.add_shape(type="rect", x0=p["o2"] - o2_tol, x1=p["o2"] + o2_tol,
                      y0=p["kpa"] - kpa_tol, y1=p["kpa"] + kpa_tol,
                      line=dict(color=INK_2, width=1.5, dash="dash"), fillcolor="rgba(11,11,11,0.04)", layer="below")
        fig.add_trace(go.Scatter(
            x=[p["o2"]], y=[p["kpa"]], mode="markers+text", name=f"★ {preset}",
            marker=dict(symbol="star", size=20, color=INK, line=dict(width=2, color=SURFACE)),
            text=[preset], textposition="top center", textfont=dict(color=INK, size=13),
            hovertemplate=f"<b>{preset}</b><br>{p['o2']:g}% O₂, {p['kpa']:g} kPa<br>{p['source']}<extra></extra>",
        ))
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_2), title_font=dict(color=INK_2))
    fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2), height=480,
                      margin=dict(l=10, r=10, t=30, b=10), xaxis=axis, yaxis=axis,
                      legend=dict(title_text="", bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, theme=None, width="stretch")

    stacked = pts.groupby(["oxygen_pct", "pressure_kpa"]).size()
    top = stacked.idxmax()
    box = " The dashed box is the tolerance set above." if p is not None else " Choose a scenario to place it on the map."
    st.caption(f"{len(pts)} tests. Points overlap where conditions were identical: {stacked.max()} tests sit at "
               f"{top[0]:g}% O₂, {top[1]:g} kPa. Values are exact (no jitter).{box}")


def article():
    st.markdown("""
<style>
  .block-container {max-width: 1150px; padding-top: 2rem;}
  h1 {font-size: 2.2rem !important; border-bottom: 1px solid #a2a9b1; padding-bottom: .2rem; margin-bottom: .2rem;}
  h2 {font-size: 1.6rem !important; border-bottom: 1px solid #a2a9b1; padding-bottom: .15rem; margin-top: 1.6rem;}
  .tagline {color: #54595d; font-size: .9rem; margin-bottom: 1rem;}
  .infobox {border: 1px solid #a2a9b1; background: #f8f9fa; padding: .6rem .8rem; font-size: .88rem;}
  .infobox .ib-title {font-weight: 700; text-align: center; font-size: 1rem; padding-bottom: .4rem; display: block;}
  .infobox table {width: 100%; border-collapse: collapse;}
  .infobox th, .infobox td {border: none !important; border-top: 1px solid #eaecf0 !important; background: transparent !important; padding: .2rem .3rem;}
  .infobox th {text-align: left; vertical-align: top; width: 45%;}
  .hatnote {font-style: italic; color: #54595d; padding-left: 1.6em; margin-bottom: .6rem;}
  .toc a {text-decoration: none;}
</style>""", unsafe_allow_html=True)

    presets = list(PRESETS)
    default = presets.index(qp["scenario"]) if qp.get("scenario") in PRESETS else None  # nothing pre-selected
    pillars = ["All", "prevention", "detection", "suppression"]
    default_p = pillars.index(qp["pillar"]) if qp.get("pillar") in pillars else 0

    with st.sidebar:
        st.markdown("<a href='?' target='_self' style='text-decoration:none;font-family:Georgia,serif;font-size:1.3rem;color:#202122'>"
                    "🔥 Flame in Freefall</a>", unsafe_allow_html=True)
        st.markdown("### Contents")
        st.markdown("<div class='toc'>" + "".join(
            f"{i}&nbsp; <a href='#{a}'>{t}</a><br>" for i, (a, t) in enumerate(SECTIONS, 1)) + "</div>",
            unsafe_allow_html=True)
        st.markdown("<a href='?page=explore' target='_self'>Data explorer: filter all tests →</a>",
                    unsafe_allow_html=True)
        st.divider()
        preset = st.selectbox("Mission scenario", presets, index=default, placeholder="Choose a scenario…")
        if preset:
            st.caption(PRESETS[preset]["source"])
        pillar = st.radio("Fire safety pillar", pillars, index=default_p)

    st.markdown("<h1>Flame in Freefall: Fire Safety Evidence Atlas</h1>", unsafe_allow_html=True)
    st.markdown("<div class='tagline'>From NASA's microgravity combustion experiments (Physical Sciences Informatics)</div>",
                unsafe_allow_html=True)

    main, side = st.columns([2.3, 1], gap="large")
    with side:
        p, counts = PRESETS.get(preset), df["experiment"].value_counts()
        scenario_box = f"""
  <span class="ib-title">{preset}</span>
  <table>
    <tr><th>Oxygen</th><td>{p['o2']:g}%</td></tr>
    <tr><th>Pressure</th><td>{p['kpa']:g} kPa ({p['kpa'] / 6.895:.1f} psi)</td></tr>
    <tr><th>Gravity</th><td>{p['gravity']}</td></tr>
    <tr><th>Source</th><td>{p['source']}</td></tr>
  </table>
  <hr style="margin:.5rem 0">""" if p else ""
        st.markdown(f"""
<div class="infobox">{scenario_box}
  <span class="ib-title">NASA data in this atlas</span>
  <table>
    <tr><th>Tests</th><td>{len(df)}</td></tr>
    {''.join(f"<tr><th>{e}</th><td>{n}</td></tr>" for e, n in counts.items())}
    <tr><th>Gravity tested</th><td>{', '.join(sorted(df.gravity.unique()))}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    with main:
        heading("overview")
        st.markdown(
            "**Flame in Freefall** gathers NASA's microgravity fire experiments into one table, ranks them "
            "against a mission scenario, and shows where the evidence stops. It covers the three pillars of "
            "spacecraft fire safety: **prevention** (BASS, BASS-II), **detection** (SAME, SAME-R) and "
            "**suppression** (FLEX).")
        placeholder("TODO [Pitch]: 2-3 sentence summary + key finding")

        heading("scenario")
        pil = None if pillar == "All" else pillar
        pts = df if pil is None else df[df["pillar"] == pil]
        o2_tol = kpa_tol = None

        if p is None:
            st.info("**Choose a mission scenario in the sidebar** (ISS cabin, Moon base, exploration atmosphere, "
                    "spacesuit or Mars habitat). The atlas will then rank NASA tests against its oxygen, pressure "
                    "and gravity, and show where the evidence stops.", icon="👈")
        else:
            st.markdown("<div class='hatnote'>All numbers below are computed from NASA tables, not by AI.</div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            o2_tol = c1.slider("Oxygen tolerance (± percentage points)", 0.5, 15.0, 3.0, 0.5)
            kpa_tol = c2.slider("Pressure tolerance (± kPa)", 1.0, 60.0, 10.0, 1.0)
            with st.expander("Ranking weights"):
                w1, w2, w3 = st.columns(3)
                w_o2 = w1.slider("Oxygen weight", 0.0, 3.0, 1.0, 0.1)
                w_kpa = w2.slider("Pressure weight", 0.0, 3.0, 1.0, 0.1)
                n_rank = w3.slider("Tests to show", 5, 50, 10, 5)
                st.caption(f"distance = √((w_O₂ · ΔO₂ / {O2_SCALE:g})² + (w_P · ΔP / {KPA_SCALE:g})²); "
                           f"relevance = 1 / (1 + distance). {O2_SCALE:g} O₂ points count the same as {KPA_SCALE:g} kPa.")
            st.caption(f"Tolerance box: {p['o2'] - o2_tol:g}–{p['o2'] + o2_tol:g}% O₂ and "
                       f"{p['kpa'] - kpa_tol:g}–{p['kpa'] + kpa_tol:g} kPa.")

        heading("evidence")
        if p is None:
            st.caption("Choose a scenario to rank the tests.")
        else:
            evidence_table(rank(df, preset, pillar=pil, n=n_rank, w_o2=w_o2, w_kpa=w_kpa), p)

        heading("gaps")
        if p is None:
            st.caption("Choose a scenario to see which pillars and fuel types have tests near it.")
        else:
            gaps = gap(df, preset, o2_tol=o2_tol, kpa_tol=kpa_tol)
            gap_banners(gaps if pil is None else gaps[gaps["pillar"] == pil], o2_tol, kpa_tol)

        heading("map")
        conditions_map(pts, preset, p, o2_tol, kpa_tol)

        heading("experiments")
        for name, pillar_label, desc in [
            ("BASS / BASS-II", "Prevention", "How solid materials (PMMA, fabric, Nomex) ignite, spread and go out in low-speed airflow."),
            ("FLEX", "Suppression", "Fuel droplets burned in different O₂, pressure and suppressant (CO₂, He) mixtures."),
            ("SAME / SAME-R", "Detection", "Smoke particle sizes from spacecraft materials and how ISS/Shuttle detectors respond."),
        ]:
            with st.expander(f"{name}: {pillar_label}"):
                st.write(desc)
                placeholder("TODO: summary card, key findings with page citations, small chart")

        heading("ask")
        st.markdown("<div class='hatnote'>Answers use only NASA documents and the tests ranked above, with citations. "
                    "If there is no evidence, the assistant says so.</div>", unsafe_allow_html=True)
        question = st.text_input("Ask a question about fire safety evidence", value=qp.get("q", ""),
                                 placeholder="e.g. In the high-oxygen FLEX tests, did flames go out on their own?")
        placeholder("TODO [RAG 2]: answer(question, rank(...), gap(...)) → text with citations")

        heading("earth")
        placeholder("TODO [Pitch]: NASA spinoffs: firefighter air tanks, smoke detector, protective clothing")

        heading("method")
        st.markdown(
            "- **Data:** 678 tests from PSI-25, PSI-26, PSI-69, PSI-101, PSI-102, built with `build_master.py`.\n"
            "- **Ranking:** distance on O₂ and pressure (scales and tolerances shown above).\n"
            "- **Outcome categories** for BASS/BASS-II are our reading of crew notes; the original note is always shown.\n"
            "- **Use of AI:** TODO list tools used (code assistance, Gemini for answers).")

        heading("references")
        placeholder("TODO [Pitch]: numbered list of PSI datasets (with DOI), NTRS reports, papers")

    st.markdown("<hr><div class='tagline'>Built for NASA Space Apps Challenge 2026, Flame in Freefall. "
                "Data: NASA Physical Sciences Informatics. Not affiliated with or endorsed by NASA.</div>",
                unsafe_allow_html=True)


# ====================================================================== DATA EXPLORER (pick columns → filter → results)
LABELS = {
    "experiment": "Experiment", "psi_id": "PSI dataset", "pillar": "Pillar", "test_id": "Test ID",
    "material": "Material", "material_class": "Fuel type", "oxygen_pct": "Oxygen (%)",
    "pressure_kpa": "Pressure (kPa)", "airflow_cms": "Airflow (cm/s)", "airflow_min_cms": "Minimum airflow (cm/s)",
    "flow_direction": "Flow direction", "gravity": "Gravity", "suppressant": "Suppressant",
    "condition_detail": "Condition details", "outcome_category": "Outcome", "outcome_raw": "Original note",
    "assumed_fields": "Nominal values used for", "source_file": "Source file", "source_location": "Source location",
}
NUMERIC = ["oxygen_pct", "pressure_kpa", "airflow_cms", "airflow_min_cms"]  # range sliders
TEXT = ["test_id", "condition_detail", "outcome_raw"]                      # "contains" search
NOT_FILTERABLE = ["assumed_fields", "source_file", "source_location"]      # always shown, never filtered
FILTERABLE = [c for c in LABELS if c not in NOT_FILTERABLE]                # the rest: multi-select dropdowns
NONE = "(blank in table)"


def column_filter(c):
    """Draw the right widget for column c and return a boolean mask over df."""
    s, label = df[c], LABELS[c]
    n_missing = int(s.isna().sum())

    if c in NUMERIC:
        lo, hi = float(s.min()), float(s.max())
        if lo == hi:
            st.caption(f"{label}: every test is at {lo:g}")
            return pd.Series(True, index=df.index)
        a, b = st.slider(label, lo, hi, (lo, hi), step=0.1, key=f"f_{c}")
        keep = st.checkbox(f"Keep the {n_missing} tests with no {label.lower()}", True, key=f"na_{c}") if n_missing else False
        return s.between(a, b) | (s.isna() & keep)

    if c in TEXT:
        q = st.text_input(f"{label} contains", key=f"f_{c}", placeholder="type to search")
        return s.fillna("").str.contains(q, case=False, regex=False) if q else pd.Series(True, index=df.index)

    options = sorted(s.dropna().unique()) + ([NONE] if n_missing else [])
    pick = st.multiselect(label, options, key=f"f_{c}", placeholder=f"All {len(options)} values")
    if not pick:
        return pd.Series(True, index=df.index)
    return s.isin(pick) | (s.isna() & (NONE in pick))


def explore():
    st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
  .block-container {max-width: 1150px; padding-top: 2rem;}
  h1 {font-size: 2.2rem !important; border-bottom: 1px solid #a2a9b1; padding-bottom: .2rem; margin-bottom: .2rem;}
  .tagline {color: #54595d; font-size: .9rem; margin-bottom: 1rem;}
</style>""", unsafe_allow_html=True)
    st.markdown("<a href='?' target='_self' style='text-decoration:none'>← Flame in Freefall</a>", unsafe_allow_html=True)
    st.markdown("<h1>Data explorer</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='tagline'>Filter all {len(df)} NASA microgravity fire tests by any column. "
                "Every row keeps its source file and page.</div>", unsafe_allow_html=True)

    # 1) pick columns, confirm with Done (form = no reload on every pick)
    with st.form("pick_columns", border=False):
        f1, f2 = st.columns([5, 1], vertical_alignment="bottom")
        chosen = f1.multiselect("Columns to filter on", FILTERABLE, format_func=LABELS.get,
                                default=st.session_state.get("explore_cols", []),
                                placeholder="Choose the columns you need")
        if f2.form_submit_button("Done", type="primary", width="stretch"):
            st.session_state["explore_cols"] = chosen
    cols = st.session_state.get("explore_cols", [])

    # 2) one slider / dropdown / search box per chosen column, two per row
    mask = pd.Series(True, index=df.index)
    if cols:
        st.markdown("#### Filters")
        grid = st.columns(2, gap="large")
        for i, c in enumerate(cols):
            with grid[i % 2]:
                mask &= column_filter(c)
    hide_nominal = st.checkbox("Hide tests where a nominal (assumed) value was used",
                               help="Where NASA did not log a value per test, the master table fills in a nominal one "
                                    "and names it in the 'Nominal values used for' column")
    if hide_nominal:
        mask &= df["assumed_fields"].isna()

    # 3) results
    res = df[mask]
    st.markdown(f"#### Results: {len(res)} of {len(df)} tests match")
    if res.empty:
        st.error("No NASA test matches this combination of filters: this is an evidence gap.", icon="❌")
        return
    st.dataframe(res, hide_index=True, width="stretch", column_config=LABELS)
    st.download_button("Download these tests (CSV)", res.to_csv(index=False).encode("utf-8"),
                       file_name="flame_in_freefall_tests.csv", mime="text/csv")


# ====================================================================== router
if qp.get("page") == "article":
    article()
elif qp.get("page") == "explore":
    explore()
else:
    home()