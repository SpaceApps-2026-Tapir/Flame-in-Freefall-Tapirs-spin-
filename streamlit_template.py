from html import escape
from pathlib import Path
import os
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evidence import KPA_SCALE, O2_SCALE, PRESETS, gap, load, rank
from glossary import COLUMN_HELP, term, tip
from rag import answer, retrieve

st.set_page_config(page_title="Flame in Freefall", page_icon=":material/local_fire_department:", layout="wide")

SERIF = "'Linux Libertine', 'Georgia', 'Times New Roman', serif"
BASE_CSS = f"""
<style>
  /* light-dark() follows Streamlit's color-scheme, including changes from the app menu. */
  [data-testid="stApp"] {{
    --atlas-text: light-dark(#202122, #f1f5f9);
    --atlas-muted: light-dark(#54595d, #c5cbd4);
    --atlas-note: light-dark(#60666d, #bbc4d1);
    --atlas-link: light-dark(#3366cc, #9cc3ff);
    --atlas-border: light-dark(#a2a9b1, #5c687a);
    --atlas-divider: light-dark(#c8ccd1, #465268);
    --atlas-panel: light-dark(#f8f9fa, #202735);
    --atlas-placeholder: light-dark(#fbfbfb, #1d2532);
  }}
  h1, h2, .stMarkdown h1, .stMarkdown h2 {{font-family: {SERIF} !important; font-weight: 400 !important; color: var(--atlas-text);}}
  a {{color: var(--atlas-link);}}
  .brand-link {{text-decoration: none; font-family: Georgia, serif; font-size: 1.3rem; color: var(--atlas-text);}}
  .home-link {{display: inline-block; margin-left: .6rem; padding: .15rem .5rem; border: 1px solid var(--atlas-border);
               border-radius: .35rem; background: var(--atlas-panel); font-size: .75rem; text-decoration: none !important;}}
  .home-link:hover {{background: var(--atlas-placeholder); text-decoration: underline !important;}}
  .home-link:focus-visible {{outline: 2px solid var(--atlas-link); outline-offset: 3px;}}
  .placeholder {{border: 1px dashed var(--atlas-border); background: var(--atlas-placeholder); color: var(--atlas-note); padding: 1.2rem; text-align: center;}}
  footer {{visibility: hidden;}}
  abbr.gl {{text-decoration: underline dotted; text-underline-offset: 3px; cursor: help;}}
  .alert {{border-radius: .35rem; border-left: 5px solid; padding: .7rem 1rem; margin: .45rem 0; line-height: 1.55;}}
  .alert.ok {{background: light-dark(#e8f5e9, #123124); color: light-dark(#14532d, #c0f5d5); border-color: light-dark(#0ca30c, #4ade80);}}
  .alert.warn {{background: light-dark(#fff7e0, #342b16); color: light-dark(#5c4400, #ffe6a3); border-color: light-dark(#e0a100, #e7b74a);}}
  .alert.bad {{background: light-dark(#fdecec, #381c24); color: light-dark(#7f1d1d, #ffc7ce); border-color: light-dark(#d03b3b, #ff8091);}}
  .alert.info {{background: light-dark(#eaf2fd, #172a43); color: light-dark(#1c3f6e, #c7ddff); border-color: light-dark(#3366cc, #83b4ff);}}
  .tag {{display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
         padding: .05rem .45rem; margin-right: .5rem; border-radius: .25rem; background: light-dark(rgba(0,0,0,.08), rgba(255,255,255,.08));}}
  .st-key-conditions_map :is(.xtick text, .ytick text, .xtitle, .ytitle, .legendtext, .textpoint text) {{
    fill: var(--atlas-text) !important;
  }}
</style>"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


def alert(kind, html, label=None):
    """Coloured box like st.success/warning/error, but HTML is allowed so glossary terms can show on hover.
    Status is carried by the colour bar AND a text label, never by colour or a symbol alone."""
    tag = f"<span class='tag'>{label}</span>" if label else ""
    st.markdown(f"<div class='alert {kind}'>{tag}{html}</div>", unsafe_allow_html=True)

# Transparent vector emblem, kept in its own file for editing.
# Keep it in one Markdown HTML block so blank lines do not split the SVG.
EMBLEM = (Path(__file__).parent / "assets" / "logo.svg").read_text(encoding="utf-8").replace("\n", " ")


@st.cache_data
def get_data():
    return load()


df = get_data()
qp = st.query_params


def link(label, sub, hint=None, **params):
    url = "?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    title = f" title='{escape(hint, quote=True)}'" if hint else ""
    return f"<a class='portal' href='{url}' target='_self'{title}><b>{label}</b><br><span>{sub}</span></a>"


# ====================================================================== HOME (portal)
def home():
    n = df["experiment"].value_counts()
    st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
  .block-container {max-width: 960px; padding-top: 3rem;}
  .title {text-align:center; font-family: """ + SERIF + """; font-size: 2.9rem; margin: .2rem 0 0; color:var(--atlas-text);}
  .subtitle {text-align:center; font-family: """ + SERIF + """; font-size: 1.05rem; color:var(--atlas-muted); margin-bottom: 1.4rem;}
  .portal, .portal * {text-decoration:none !important;}
  .portal-grid {display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.9rem; margin:1.5rem 0 1rem;}
  .portal:hover {border-color:var(--atlas-link); background:var(--atlas-placeholder);}
  .portal:hover b {text-decoration: underline !important;}
  .portal {display:flex; flex-direction:column; justify-content:center; gap:.45rem; color:var(--atlas-link) !important;
           line-height:1.3; padding:1rem; min-height:6.5rem; border:1px solid var(--atlas-border);
           border-radius:.4rem; background:var(--atlas-panel); overflow-wrap:anywhere;}
  .portal br {display:none;}
  .portal b {font-size: 1.05rem; font-weight: 700;}
  .portal span {color:var(--atlas-muted); font-size: .85rem;}
  .portal:focus-visible {outline:2px solid var(--atlas-link); outline-offset:4px; border-radius:.25rem;}
  .emblem {display:flex; justify-content:center; margin-bottom:1rem;}
  @media (max-width:800px) {.portal-grid {grid-template-columns:repeat(2, minmax(0, 1fr));}}
  @media (max-width:480px) {.portal-grid {grid-template-columns:1fr;}}
  div[data-testid="stForm"] {border: none; padding: 0;}
  [data-testid="stFormSubmitButton"] button {background:#3366cc; color:white; border:none; border-radius:2px; width:100%;}
  [data-testid="stFormSubmitButton"] button:hover {background:#2b57ae; color:white;}
  .divider {border-top: 1px solid var(--atlas-divider); margin: 1.6rem auto 1rem; width: 75%;}
  .langs {text-align:center; font-size: .92rem; line-height: 2;}
  .langs a {margin: 0 .6rem; text-decoration:none;}
  .note {text-align:center; color:var(--atlas-note); font-size:.78rem; margin-top: 1.5rem;}
</style>""", unsafe_allow_html=True)

    st.markdown("<div class='title'>Flame in Freefall</div>"
                "<div class='subtitle'>The Fire Safety Evidence Atlas</div>", unsafe_allow_html=True)

    scenario_links = [link(name.replace(" (challenge brief)", ""),
                           f"{s['o2']:g}% O₂ · {s['kpa'] / 6.895:.1f} psi",
                           hint=f"{s['o2']:g}% oxygen at {s['kpa']:g} kPa ({s['kpa'] / 6.895:.1f} psi), "
                                f"{s['gravity']} gravity. Source: {s['source']}",
                           page="article", scenario=name)
                      for name, s in PRESETS.items()]

    pillar_links = [
        link("Prevention", f"BASS · BASS-II · {n.get('BASS', 0) + n.get('BASS-II', 0)} tests",
             hint=tip("prevention"), page="article", pillar="prevention"),
        link("Detection", f"SAME · SAME-R · {n.get('SAME', 0) + n.get('SAME-R', 0)} tests",
             hint=tip("detection"), page="article", pillar="detection"),
        link("Suppression", f"FLEX · {n.get('FLEX', 0)} tests",
             hint=tip("suppression"), page="article", pillar="suppression"),
    ]
    st.markdown(f"<div class='emblem'>{EMBLEM}</div>", unsafe_allow_html=True)

    # search bar: one question box + blue button (the scenario is chosen in the article's sidebar)
    _, c, _ = st.columns([0.6, 3, 0.6])
    with c:
        with st.form("search", border=False):
            s1, s2 = st.columns([3, 0.55], vertical_alignment="bottom")
            q = s1.text_input("Question", placeholder="Ask the evidence, e.g. Do flames go out on their own at 30% oxygen?",
                              label_visibility="collapsed")
            go = s2.form_submit_button("Search")
        if go:
            st.query_params.update({"page": "article", "q": q})
            st.rerun()
        st.markdown("<div class='langs'><a href='?page=explore' target='_self'>Or filter all "
                    f"{len(df)} tests in the data explorer →</a></div>", unsafe_allow_html=True)

    st.markdown("<nav class='portal-grid' aria-label='Fire safety topics and mission scenarios'>"
                + "".join(pillar_links + scenario_links) + "</nav>", unsafe_allow_html=True)

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


def evidence_table(ranked, p):
    """Closest NASA tests to the scenario, straight from rank()."""
    if p["gravity"] not in set(df["gravity"]):
        alert("warn", f"No test in this atlas was run at <b>{term(p['gravity'] + ' gravity', p['gravity'])}</b> "
                      f"(all {len(df)} are in {term('microgravity')}). "
                      "Tests below are ranked on oxygen and pressure only.", label="Gravity mismatch")
    if ranked.empty:
        alert("info", "No tests with oxygen and pressure values for this pillar.")
        return
    cols = ["experiment", "test_id", "material", "oxygen_pct", "pressure_kpa", "airflow_cms", "suppressant",
            "outcome_category", "relevance", "why", "source_location"]
    h = lambda c: tip(COLUMN_HELP[c])  # noqa: E731  header tooltip from the glossary
    st.dataframe(ranked[cols], hide_index=True, width="stretch", column_config={
        "experiment": st.column_config.TextColumn("Experiment", help=h("experiment")),
        "test_id": st.column_config.TextColumn("Test", help=h("test_id")),
        "material": st.column_config.TextColumn("Material", help=h("material")),
        "oxygen_pct": st.column_config.NumberColumn("O₂ %", format="%.1f", help=h("oxygen_pct")),
        "pressure_kpa": st.column_config.NumberColumn("kPa", format="%.1f", help=h("pressure_kpa")),
        "airflow_cms": st.column_config.NumberColumn("Airflow cm/s", format="%.1f", help=h("airflow_cms")),
        "suppressant": st.column_config.TextColumn("Suppressant", help=h("suppressant")),
        "outcome_category": st.column_config.TextColumn("Outcome", help=h("outcome_category")),
        "relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1, format="%.2f",
                                                     help=h("relevance")),
        "why": st.column_config.TextColumn("Why it ranks here", help=h("why")),
        "source_location": st.column_config.TextColumn("Source", help=h("source_location")),
    })
    outcomes = " · ".join(term(o) for o in ranked["outcome_category"].dropna().unique())
    st.caption(f"Hover a column header or a dotted word for its meaning. Outcomes in this table: {outcomes}. "
               f"{term('Relevance', 'relevance')} compares oxygen and pressure only, not material, airflow or gravity. "
               "Pick a test below to see the original NASA note and full source.", unsafe_allow_html=True)

    labels = [f"#{i} · {r.experiment} {r.test_id} · {r.material} · {r.oxygen_pct:g}% O₂, {r.pressure_kpa:g} kPa"
              for i, r in enumerate(ranked.itertuples(), 1)]
    pick = st.selectbox("Show details for test", range(len(ranked)), format_func=lambda i: labels[i])
    test_details(ranked.iloc[pick])


def test_details(r):
    """Detail panel for one test: the NASA note behind the outcome category, and where it comes from."""
    def val(x, unit=""):
        return "not recorded" if pd.isna(x) else (f"{x:g}{unit}" if isinstance(x, float) else str(x))

    with st.container(border=True):
        st.markdown(f"#### {term(r.experiment)} {escape(str(r.test_id))}: {escape(str(r.material))}",
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown(f"**{term('Pillar', 'pillar')}:** {term(r.pillar)}  \n"
                   f"**{term('Fuel type', 'fuel type')}:** {term(r.material_class)}  \n"
                   f"**{term('Oxygen', 'o2')}:** {val(r.oxygen_pct, '%')}  \n"
                   f"**Pressure:** {val(r.pressure_kpa) if pd.isna(r.pressure_kpa) else f'{r.pressure_kpa:g} ' + term('kPa', 'kpa')}  \n"
                   f"**{term('Gravity', 'gravity')}:** {term(r.gravity)}", unsafe_allow_html=True)
        b.markdown(f"**{term('Airflow', 'airflow')}:** {val(r.airflow_cms, ' cm/s')} ({term(val(r.flow_direction))})  \n"
                   f"**{term('Suppressant', 'suppressant')}:** {escape(val(r.suppressant))}  \n"
                   f"**{term('Outcome', 'outcome')} (our category):** {term(r.outcome_category)}  \n"
                   f"**{term('Relevance', 'relevance')}:** {r.relevance:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Original NASA note:** {val(r.outcome_raw)}")
        st.markdown(f"**Test conditions as logged:** {val(r.condition_detail)}")
        if not pd.isna(r.assumed_fields):
            alert("info", f"{term('Nominal value', 'nominal')} filled in for: {escape(str(r.assumed_fields))} "
                          "(not logged per test).", label="Note")
        st.caption(f"Source: {r.source_file} · {r.source_location}")


def gap_banners(gaps, o2_tol, kpa_tol):
    """One banner per pillar x fuel type: covered, covered except gravity, or GAP."""
    def verdict(r):
        return "gap" if r.status == "GAP" else ("gravity" if r.gravity_gap else "covered")

    gaps = gaps.assign(verdict=[verdict(r) for r in gaps.itertuples()])
    m1, m2, m3 = st.columns(3)
    m1.metric("Covered", int((gaps.verdict == "covered").sum()), help=tip("covered"))
    m2.metric("Gravity gap", int((gaps.verdict == "gravity").sum()), help=tip("covered except gravity"))
    m3.metric("Gap", int((gaps.verdict == "gap").sum()), help=tip("gap"))
    st.markdown(f"<div class='hatnote'>A {term('pillar')} × {term('fuel type')} is covered when at least one test "
                f"is within ±{o2_tol:g} {term('percentage points', 'pp')} of {term('O₂', 'o2')} <b>and</b> "
                f"±{kpa_tol:g} {term('kPa', 'kpa')} of the scenario.</div>", unsafe_allow_html=True)

    pp, o2, kpa, tol = term("pp"), term("O₂", "o2"), term("kPa", "kpa"), term("tolerance")
    for pillar_name, g in gaps.groupby("pillar", sort=False):
        st.markdown(f"**{term(pillar_name.capitalize(), pillar_name)}**", unsafe_allow_html=True)
        for r in g.itertuples():
            msg = (f"<b>{term(r.material_class)}</b>: {r.tests_near} of {r.tests_total} tests within {tol}. "
                   f"Closest: {r.closest_test} at {r.closest_o2_pct:g}% {o2}, {r.closest_kpa:g} {kpa} "
                   f"(<b>{r.closest_d_o2:+.1f} {pp} O₂, {r.closest_d_kpa:+.1f} kPa</b> from the scenario). "
                   f"Tested: {r.o2_range_tested} O₂, {r.kpa_range_tested}, gravity: {term(r.gravity_tested)}.")
            if r.verdict == "gap":
                over_o2 = max(0.0, abs(r.closest_d_o2) - o2_tol)
                over_kpa = max(0.0, abs(r.closest_d_kpa) - kpa_tol)
                outside = " and ".join(t for t in [f"{over_o2:.1f} {pp} O₂" if over_o2 else "",
                                                   f"{over_kpa:.1f} kPa" if over_kpa else ""] if t)
                alert("bad", f"{msg} It lies outside the {tol} box by {outside}.", label=term("Gap", "gap"))
            elif r.verdict == "gravity":
                alert("warn", f"{msg} Oxygen and pressure match, gravity does not.",
                      label=term("Gravity gap", "covered except gravity"))
            else:
                alert("ok", msg, label=term("Covered", "covered"))


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
    fig.update_traces(marker=dict(size=9, opacity=0.7, line=dict(width=1.5, color="rgba(128,128,128,0.45)")))
    if p is not None:  # no scenario chosen yet: just the tests, no star or tolerance box
        fig.add_shape(type="rect", x0=p["o2"] - o2_tol, x1=p["o2"] + o2_tol,
                      y0=p["kpa"] - kpa_tol, y1=p["kpa"] + kpa_tol,
                      line=dict(color="#8a9099", width=1.5, dash="dash"), fillcolor="rgba(128,128,128,0.08)", layer="below")
        fig.add_trace(go.Scatter(
            x=[p["o2"]], y=[p["kpa"]], mode="markers+text", name=f"Scenario: {preset}",
            marker=dict(symbol="star", size=20, color="#ffffff", line=dict(width=2, color="#0b0b0b")),
            text=[preset], textposition="top center", textfont=dict(size=13),
            hovertemplate=f"<b>{preset}</b><br>{p['o2']:g}% O₂, {p['kpa']:g} kPa<br>{p['source']}<extra></extra>",
        ))
    axis = dict(zeroline=False)
    fig.update_layout(height=480,
                      margin=dict(l=10, r=10, t=30, b=10), xaxis=axis, yaxis=axis,
                      legend=dict(title_text="", bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, theme="streamlit", key="conditions_map", width="stretch")

    stacked = pts.groupby(["oxygen_pct", "pressure_kpa"]).size()
    top = stacked.idxmax()
    box = (f" The dashed box is the {term('tolerance')} set above." if p is not None
           else " Choose a scenario to place it on the map.")
    pillars = ", ".join(term(k) for k in PILLAR_COLORS)
    st.caption(f"{len(pts)} tests. Colour = {term('pillar')} ({pillars}); shape = experiment "
               f"({' · '.join(term(e) for e in EXPERIMENT_SYMBOLS)}). Points overlap where conditions were identical: "
               f"{stacked.max()} tests sit at {top[0]:g}% O₂, {top[1]:g} kPa. Values are exact (no jitter).{box}",
               unsafe_allow_html=True)


def article():
    st.markdown("""
<style>
  .block-container {max-width: 1150px; padding-top: 2rem;}
  h1 {font-size: 2.2rem !important; border-bottom: 1px solid var(--atlas-border); padding-bottom: .2rem; margin-bottom: .2rem;}
  h2 {font-size: 1.6rem !important; border-bottom: 1px solid var(--atlas-border); padding-bottom: .15rem; margin-top: 1.6rem;}
  .tagline {color: var(--atlas-muted); font-size: .9rem; margin-bottom: 1rem;}
  .infobox {border: 1px solid var(--atlas-border); background: var(--atlas-panel); color: var(--atlas-text); padding: .6rem .8rem; font-size: .88rem;}
  .infobox .ib-title {font-weight: 700; text-align: center; font-size: 1rem; padding-bottom: .4rem; display: block;}
  .infobox table {width: 100%; border-collapse: collapse;}
  .infobox th, .infobox td {border: none !important; border-top: 1px solid var(--atlas-divider) !important; background: transparent !important; color: inherit; padding: .2rem .3rem;}
  .infobox th {text-align: left; vertical-align: top; width: 45%;}
  .hatnote {font-style: italic; color: var(--atlas-muted); padding-left: 1.6em; margin-bottom: .6rem;}
  .toc a {text-decoration: none;}
</style>""", unsafe_allow_html=True)

    presets = list(PRESETS)
    default = presets.index(qp["scenario"]) if qp.get("scenario") in PRESETS else None  # nothing pre-selected
    pillars = ["All", "prevention", "detection", "suppression"]
    default_p = pillars.index(qp["pillar"]) if qp.get("pillar") in pillars else 0

    with st.sidebar:
        st.markdown("<a href='?' target='_self' class='brand-link'>"
                    "Flame in Freefall</a>", unsafe_allow_html=True)
        st.markdown("### Contents")
        st.markdown("<div class='toc'>" + "".join(
            f"{i}&nbsp; <a href='#{a}'>{t}</a><br>" for i, (a, t) in enumerate(SECTIONS, 1)) + "</div>",
            unsafe_allow_html=True)
        st.markdown("<a href='?page=explore' target='_self'>Data explorer: filter all tests →</a>",
                    unsafe_allow_html=True)
        st.divider()
        preset = st.selectbox("Mission scenario", presets, index=default, placeholder="Choose a scenario…",
                              help=tip("scenario"))
        if preset:
            st.caption(PRESETS[preset]["source"])
        pillar = st.radio("Fire safety pillar", pillars, index=default_p, help=tip("pillar"),
                          captions=["Every test", "Stop fires starting", "Spot fires early", "Put fires out"])

    st.markdown("<h1>Flame in Freefall: Fire Safety Evidence Atlas</h1>", unsafe_allow_html=True)
    st.markdown("<div class='tagline'>From NASA's microgravity combustion experiments (Physical Sciences Informatics)</div>",
                unsafe_allow_html=True)

    main, side = st.columns([2.3, 1], gap="large")
    with side:
        p, counts = PRESETS.get(preset), df["experiment"].value_counts()
        scenario_box = f"""
  <span class="ib-title">{preset}</span>
  <table>
    <tr><th>{term('Oxygen', 'o2')}</th><td>{p['o2']:g}%</td></tr>
    <tr><th>Pressure</th><td>{p['kpa']:g} {term('kPa', 'kpa')} ({p['kpa'] / 6.895:.1f} {term('psi')})</td></tr>
    <tr><th>{term('Gravity', 'gravity')}</th><td>{term(p['gravity'])}</td></tr>
    <tr><th>Source</th><td>{escape(p['source'])}</td></tr>
  </table>
  <hr style="margin:.5rem 0">""" if p else ""
        st.markdown(f"""
<div class="infobox">{scenario_box}
  <span class="ib-title">NASA data in this atlas</span>
  <table>
    <tr><th>Tests</th><td>{len(df)}</td></tr>
    {''.join(f"<tr><th>{term(e)}</th><td>{n}</td></tr>" for e, n in counts.items())}
    <tr><th>Gravity tested</th><td>{', '.join(term(g) for g in sorted(df.gravity.unique()))}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    with main:
        heading("overview")
        st.markdown(
            f"**Flame in Freefall** gathers NASA's {term('microgravity')} fire experiments into one table, ranks them "
            f"against a mission {term('scenario')}, and shows where the evidence stops. It covers the three "
            f"{term('pillars', 'pillar')} of spacecraft fire safety: "
            f"**{term('prevention')}** ({term('BASS')}, {term('BASS-II')}), "
            f"**{term('detection')}** ({term('SAME')}, {term('SAME-R')}) and **{term('suppression')}** ({term('FLEX')}). "
            "Words with a dotted underline show a short definition when you hover over them.",
            unsafe_allow_html=True)
        placeholder("TODO [Pitch]: 2-3 sentence summary + key finding")

        heading("scenario")
        pil = None if pillar == "All" else pillar
        pts = df if pil is None else df[df["pillar"] == pil]
        o2_tol = kpa_tol = None

        if p is None:
            alert("info", "<b>Choose a mission scenario in the sidebar</b> (ISS cabin, Moon base, exploration "
                          "atmosphere, spacesuit or Mars habitat). The atlas will then rank NASA tests against its "
                          "oxygen, pressure and gravity, and show where the evidence stops.", label="Start here")
        else:
            st.markdown("<div class='hatnote'>All numbers below are computed from NASA tables, not by AI.</div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            o2_tol = c1.slider("Oxygen tolerance (± percentage points)", 0.5, 15.0, 3.0, 0.5,
                               help=f"{tip('tolerance')} {tip('pp')}")
            kpa_tol = c2.slider("Pressure tolerance (± kPa)", 1.0, 60.0, 10.0, 1.0,
                                help=f"{tip('tolerance')} {tip('kpa')}")
            with st.expander("Ranking weights"):
                w1, w2, w3 = st.columns(3)
                w_o2 = w1.slider("Oxygen weight", 0.0, 3.0, 1.0, 0.1, help=tip("weight"))
                w_kpa = w2.slider("Pressure weight", 0.0, 3.0, 1.0, 0.1, help=tip("weight"))
                n_rank = w3.slider("Tests to show", 5, 50, 10, 5, help="How many of the closest tests to list.")
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
            ("BASS / BASS-II", "Prevention",
             f"{term('BASS')} / {term('BASS-II')}: how solid materials ({term('PMMA', 'pmma')} plastic, fabric such as "
             f"{term('Nomex', 'nomex')}, wax) ignite, spread and go out in low-speed {term('airflow')}. "
             f"Run in the {term('MSG', 'msg')}."),
            ("FLEX", "Suppression",
             f"{term('FLEX')}: {term('fuel droplets', 'liquid fuel droplet')} burned in different {term('O₂', 'o2')}, "
             f"pressure and {term('suppressant')} (CO₂, helium) mixtures. Run in the {term('CIR', 'cir')} with the "
             f"{term('MDCA', 'mdca')}."),
            ("SAME / SAME-R", "Detection",
             f"{term('SAME')} / {term('SAME-R')}: smoke particle sizes from heated spacecraft materials and how "
             f"smoke detectors respond. Includes a {term('reference aerosol')}."),
        ]:
            with st.expander(f"{name}: {pillar_label}"):
                st.markdown(desc, unsafe_allow_html=True)
                placeholder("TODO: summary card, key findings with page citations, small chart")

        heading("ask")
        st.markdown("<div class='hatnote'>Answers use only NASA documents and the tests ranked above, with citations. "
                    "If there is no evidence, the assistant says so.</div>", unsafe_allow_html=True)
        question = st.text_input("Ask a question about fire safety evidence", value=qp.get("q", ""),
                                 placeholder="e.g. In the high-oxygen FLEX tests, did flames go out on their own?")
        if st.button("Find evidence and answer", disabled=not question.strip()):
            try:
                passages = retrieve(question)
                if not passages:
                    st.info("No indexed PDF passages. Add documents to documents/manifest.csv and run the ingestion script.")
                else:
                    for i, passage in enumerate(passages, 1):
                        with st.expander(f"[S{i}] {passage['title']} — PDF page {passage['page']}"):
                            st.write(passage["text"])
                            source = passage.get("source_page_url") or passage["pdf_url"]
                            st.link_button("Open source PDF", source.split("#")[0] + f"#page={passage['page']}")
                    api_key = os.environ.get("GROQ_API_KEY", "")
                    model = os.environ.get("GROQ_MODEL", "")
                    if not api_key or not model:
                        st.info("Passage search is ready. Set GROQ_API_KEY and GROQ_MODEL to enable generated answers.")
                    else:
                        context = {"scenario": PRESETS[preset] if preset else None}
                        if preset:
                            context["ranked_tests"] = rank(df, preset, pillar=pil, n=5).to_json(orient="records")
                            context["gaps"] = gap(df, preset).to_json(orient="records")
                        with st.spinner("Reading the retrieved evidence…"):
                            st.write(answer(question, passages, context, api_key, model))
                        st.caption("AI-generated answer. Check the passages above; citation IDs do not guarantee factual accuracy.")
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(str(exc))

        heading("earth")
        placeholder("TODO [Pitch]: NASA spinoffs: firefighter air tanks, smoke detector, protective clothing")

        heading("method")
        st.markdown(
            f"- **Data:** 678 tests from {term('PSI', 'psi_db')}-25, PSI-26, PSI-69, PSI-101, PSI-102, "
            "built with `build_master.py`.\n"
            f"- **Ranking:** distance on O₂ and pressure ({term('relevance')}, {term('weights', 'weight')} and "
            f"{term('tolerances', 'tolerance')} shown above).\n"
            f"- **{term('Outcome', 'outcome')} categories** for BASS/BASS-II are our reading of crew notes; "
            "the original note is always shown.\n"
            "- **Use of AI:** TODO list tools used (code assistance, Groq for answers).", unsafe_allow_html=True)

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
    s, label, help_text = df[c], LABELS[c], tip(COLUMN_HELP.get(c, c))
    n_missing = int(s.isna().sum())

    if c in NUMERIC:
        lo, hi = float(s.min()), float(s.max())
        if lo == hi:
            st.caption(f"{label}: every test is at {lo:g}")
            return pd.Series(True, index=df.index)
        a, b = st.slider(label, lo, hi, (lo, hi), step=0.1, key=f"f_{c}", help=help_text)
        keep = st.checkbox(f"Keep the {n_missing} tests with no {label.lower()}", True, key=f"na_{c}") if n_missing else False
        return s.between(a, b) | (s.isna() & keep)

    if c in TEXT:
        q = st.text_input(f"{label} contains", key=f"f_{c}", placeholder="type to search", help=help_text)
        return s.fillna("").str.contains(q, case=False, regex=False) if q else pd.Series(True, index=df.index)

    options = sorted(s.dropna().unique()) + ([NONE] if n_missing else [])
    pick = st.multiselect(label, options, key=f"f_{c}", placeholder=f"All {len(options)} values", help=help_text)
    defined = [o for o in options if tip(o)]  # values with a glossary entry (outcomes, flow directions, fuel types...)
    if defined:
        st.caption("Hover for meaning: " + " · ".join(term(o) for o in defined), unsafe_allow_html=True)
    if not pick:
        return pd.Series(True, index=df.index)
    return s.isin(pick) | (s.isna() & (NONE in pick))


def explore():
    st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
  .block-container {max-width: 1150px; padding-top: 2rem;}
  h1 {font-size: 2.2rem !important; border-bottom: 1px solid var(--atlas-border); padding-bottom: .2rem; margin-bottom: .2rem;}
  .tagline {color: var(--atlas-muted); font-size: .9rem; margin-bottom: 1rem;}
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
                                placeholder="Choose the columns you need",
                                help="After you press Done, each column gets a slider, dropdown or search box. "
                                     "Hover the ? next to each filter, or a results column header, for its meaning.")
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
                               help=tip("nominal"))
    if hide_nominal:
        mask &= df["assumed_fields"].isna()

    # 3) results
    res = df[mask]
    st.markdown(f"#### Results: {len(res)} of {len(df)} tests match")
    if res.empty:
        alert("bad", "No NASA test matches this combination of filters: this is an evidence gap.", label="No matches")
        return
    st.dataframe(res, hide_index=True, width="stretch", column_config={
        c: st.column_config.Column(LABELS[c], help=tip(COLUMN_HELP.get(c, c))) for c in res.columns})
    st.download_button("Download these tests (CSV)", res.to_csv(index=False).encode("utf-8"),
                       file_name="flame_in_freefall_tests.csv", mime="text/csv")


# ====================================================================== router
if qp.get("page") == "article":
    article()
elif qp.get("page") == "explore":
    explore()
else:
    home()
