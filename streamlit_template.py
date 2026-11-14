from urllib.parse import quote

import streamlit as st

from evidence import PRESETS, load

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

    # emblem with portal links around it (like the language ring)
    left, mid, right = st.columns([1.2, 1, 1.2])
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
                    + link("Moon base", "30% O₂ · 10.2 psi", page="article", scenario="Moon base (challenge brief)")
                    + link("ISS cabin", "21% O₂ · 14.7 psi", page="article", scenario="ISS cabin")
                    + link("Evidence gaps", "where NASA data stops", page="article", scenario="Exploration atmosphere")
                    + "</div>", unsafe_allow_html=True)

    # search bar: scenario dropdown (like the language picker) + question + blue button
    _, c, _ = st.columns([0.6, 3, 0.6])
    with c:
        with st.form("search", border=False):
            s1, s2, s3 = st.columns([1.35, 3, 0.55], vertical_alignment="bottom")
            scen = s1.selectbox("Scenario", list(PRESETS), index=1, label_visibility="collapsed")
            q = s2.text_input("Question", placeholder="Ask the evidence, e.g. Do flames go out on their own at 30% oxygen?",
                              label_visibility="collapsed")
            go = s3.form_submit_button("🔍")
        if go:
            st.query_params.update({"page": "article", "scenario": scen, "q": q})
            st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='langs'>Read the atlas for: "
                + " · ".join(f"<a href='?page=article&scenario={quote(k)}' target='_self'>{k}</a>" for k in PRESETS)
                + "</div>", unsafe_allow_html=True)
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
    default = presets.index(qp["scenario"]) if qp.get("scenario") in PRESETS else 1
    pillars = ["All", "prevention", "detection", "suppression"]
    default_p = pillars.index(qp["pillar"]) if qp.get("pillar") in pillars else 0

    with st.sidebar:
        st.markdown("<a href='?' target='_self' style='text-decoration:none;font-family:Georgia,serif;font-size:1.3rem;color:#202122'>"
                    "🔥 Flame in Freefall</a>", unsafe_allow_html=True)
        st.markdown("### Contents")
        st.markdown("<div class='toc'>" + "".join(
            f"{i}&nbsp; <a href='#{a}'>{t}</a><br>" for i, (a, t) in enumerate(SECTIONS, 1)) + "</div>",
            unsafe_allow_html=True)
        st.divider()
        preset = st.selectbox("Mission scenario", presets, index=default)
        st.caption(PRESETS[preset]["source"])
        pillar = st.radio("Fire safety pillar", pillars, index=default_p)

    st.markdown("<h1>Flame in Freefall: Fire Safety Evidence Atlas</h1>", unsafe_allow_html=True)
    st.markdown("<div class='tagline'>From NASA's microgravity combustion experiments (Physical Sciences Informatics)</div>",
                unsafe_allow_html=True)

    main, side = st.columns([2.3, 1], gap="large")
    with side:
        p, counts = PRESETS[preset], df["experiment"].value_counts()
        st.markdown(f"""
<div class="infobox">
  <span class="ib-title">{preset}</span>
  <table>
    <tr><th>Oxygen</th><td>{p['o2']:g}%</td></tr>
    <tr><th>Pressure</th><td>{p['kpa']:g} kPa ({p['kpa'] / 6.895:.1f} psi)</td></tr>
    <tr><th>Gravity</th><td>{p['gravity']}</td></tr>
    <tr><th>Source</th><td>{p['source']}</td></tr>
  </table>
  <hr style="margin:.5rem 0">
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
        st.markdown("<div class='hatnote'>Pick a scenario in the sidebar. All numbers below are computed from NASA tables, not by AI.</div>",
                    unsafe_allow_html=True)
        placeholder("TODO [Dashboard]: tolerance sliders (O₂ ±, kPa ±) and ranking weights")

        heading("evidence")
        placeholder("TODO [Datos]: rank(df, preset, pillar) → table with relevance, outcome, 'why' + notes and source")

        heading("gaps")
        placeholder("TODO [Datos]: gap(df, preset) → ✅ covered / ❌ GAP banner per pillar, closest test and distance")

        heading("map")
        placeholder("TODO [Dashboard]: Plotly scatter, O₂ % × pressure kPa, one dot per test, star = scenario")

        heading("experiments")
        for name, pil, desc in [
            ("BASS / BASS-II", "Prevention", "How solid materials (PMMA, fabric, Nomex) ignite, spread and go out in low-speed airflow."),
            ("FLEX", "Suppression", "Fuel droplets burned in different O₂, pressure and suppressant (CO₂, He) mixtures."),
            ("SAME / SAME-R", "Detection", "Smoke particle sizes from spacecraft materials and how ISS/Shuttle detectors respond."),
        ]:
            with st.expander(f"{name}: {pil}"):
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


# ====================================================================== router
if qp.get("page") == "article":
    article()
else:
    home()