from html import escape
import json
from pathlib import Path
import os
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evidence import KPA_SCALE, O2_SCALE, PRESETS, gap, load, rank
from glossary import COLUMN_HELP, term, tip
from rag import NO_EVIDENCE, NO_PASSAGES, answer, retrieve
from references import dataset_reference, references_for
from security import csv_bytes, safe_http_url, safe_model_markdown, validate_question
from vector_store import index_inventory

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
  /* Reserve a row below form inputs for Streamlit's absolutely positioned Enter hint. */
  [data-testid="stForm"] [data-testid="stTextInput"] {{padding-bottom: 1.5rem;}}
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
                           page="scenario", scenario=name)
                      for name, s in PRESETS.items()]

    pillar_links = [
        link("Prevention", f"BASS · BASS-II · {n.get('BASS', 0) + n.get('BASS-II', 0)} tests",
             hint=tip("prevention"), page="scenario", pillar="prevention"),
        link("Detection", f"SAME · SAME-R · {n.get('SAME', 0) + n.get('SAME-R', 0)} tests",
             hint=tip("detection"), page="scenario", pillar="detection"),
        link("Suppression", f"FLEX · {n.get('FLEX', 0)} tests",
             hint=tip("suppression"), page="scenario", pillar="suppression"),
    ]
    st.markdown(f"<div class='emblem'>{EMBLEM}</div>", unsafe_allow_html=True)

    # search bar: one question box + blue button; it opens the Ask page with the answer
    _, c, _ = st.columns([0.6, 3, 0.6])
    with c:
        with st.form("search", border=False):
            s1, s2 = st.columns([3, 0.55], vertical_alignment="top")
            q = s1.text_input("Question", placeholder="Ask the evidence, e.g. Do flames go out on their own at 30% oxygen?",
                              label_visibility="collapsed")
            go = s2.form_submit_button("Search")
        if go:
            st.query_params.from_dict({"page": "ask", "q": q.strip()} if q.strip() else {"page": "ask"})
            st.rerun()
        st.markdown("<div class='langs'><a href='?page=scenario' target='_self'>Scenario report</a> · "
                    f"<a href='?page=explore' target='_self'>Filter all {len(df)} tests</a> · "
                    "<a href='?page=about' target='_self'>About this atlas</a></div>", unsafe_allow_html=True)

    st.markdown("<nav class='portal-grid' aria-label='Fire safety topics and mission scenarios'>"
                + "".join(pillar_links + scenario_links) + "</nav>", unsafe_allow_html=True)

    st.markdown(f"<div class='note'>{len(df)} microgravity fire tests from NASA Physical Sciences Informatics · "
                "Built for NASA Space Apps Challenge 2026 · Not affiliated with or endorsed by NASA</div>",
                unsafe_allow_html=True)


# ====================================================================== shared page frame (every page except home)
PAGES = [("ask", "Ask the evidence"), ("scenario", "Scenario report"), ("explore", "Data explorer"), ("about", "About")]
PAGE_CSS = """
<style>
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
  .block-container {max-width: 1150px; padding-top: 1.5rem;}
  h1 {font-size: 2.2rem !important; border-bottom: 1px solid var(--atlas-border); padding-bottom: .2rem; margin-bottom: .2rem;}
  h2 {font-size: 1.6rem !important; border-bottom: 1px solid var(--atlas-border); padding-bottom: .15rem; margin-top: 1.6rem;}
  .tagline {color: var(--atlas-muted); font-size: .95rem; margin-bottom: 1rem;}
  .topnav {display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: .4rem 1.5rem;
           border-bottom: 1px solid var(--atlas-divider); padding-bottom: .5rem; margin-bottom: 1.4rem;}
  .topnav nav {display: flex; flex-wrap: wrap; gap: .3rem 1.3rem; font-size: .95rem;}
  .topnav nav a {text-decoration: none;}
  .topnav nav b {color: var(--atlas-text); border-bottom: 2px solid var(--atlas-link);}
  .facts {font-size: 1rem; margin: .2rem 0 .6rem;}
  .facts .src {color: var(--atlas-muted); font-size: .88rem;}
  .infobox {border: 1px solid var(--atlas-border); background: var(--atlas-panel); color: var(--atlas-text); padding: .6rem .8rem; font-size: .88rem;}
  .infobox .ib-title {font-weight: 700; text-align: center; font-size: 1rem; padding-bottom: .4rem; display: block;}
  .infobox table {width: 100%; border-collapse: collapse;}
  .infobox th, .infobox td {border: none !important; border-top: 1px solid var(--atlas-divider) !important; background: transparent !important; color: inherit; padding: .2rem .3rem;}
  .infobox th {text-align: left; vertical-align: top; width: 45%;}
  .index-table {width: 100%; border-collapse: collapse; margin: .5rem 0 1rem; font-size: .92rem;}
  .index-table th, .index-table td {padding: .42rem .55rem; border-bottom: 1px solid var(--atlas-divider); text-align: left;}
  .index-table th {color: var(--atlas-muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;}
  .index-table .num {text-align: right; font-variant-numeric: tabular-nums;}
  .index-table tr.total td {font-weight: 700; border-top: 2px solid var(--atlas-border);}
  .index-table a {text-decoration: none;}
  .index-table a:hover {text-decoration: underline;}
  .ref-item {margin: .65rem 0; line-height: 1.5;}
  .ref-item .ref-meta, .ref-item .ref-authors {color: var(--atlas-muted); font-size: .85rem; margin-top: .1rem;}
  .hatnote {font-style: italic; color: var(--atlas-muted); margin-bottom: .6rem;}
</style>"""

PILLARS = ["All", "prevention", "detection", "suppression"]
PILLAR_LABELS = {"All": "All pillars", "prevention": "Prevention: stop fires starting",
                 "detection": "Detection: spot fires early", "suppression": "Suppression: put fires out"}
O2_TOL, KPA_TOL = 3.0, 10.0  # default tolerance box: ±3 percentage points O₂, ±10 kPa


def carry(**params):
    """Link that keeps the chosen scenario and pillar when moving between pages."""
    keep = {k: qp[k] for k in ("scenario", "pillar") if qp.get(k)}
    keep.update(params)
    return "?" + "&".join(f"{k}={quote(str(v))}" for k, v in keep.items())


def set_param(name, value):
    if value:
        st.query_params[name] = value
    else:
        st.query_params.pop(name, None)


def qfloat(name, default):
    try:
        return float(qp.get(name, default))
    except (TypeError, ValueError):
        return default


def pick_index(options, value):
    return options.index(value) if value in options else None


def page_header(current, title, tagline):
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    links = "".join(f"<b>{label}</b>" if key == current else f"<a href='{carry(page=key)}' target='_self'>{label}</a>"
                    for key, label in PAGES)
    st.markdown(f"<div class='topnav'><a href='?' target='_self' class='brand-link'>Flame in Freefall</a>"
                f"<nav aria-label='Pages'>{links}</nav></div>", unsafe_allow_html=True)
    st.markdown(f"<h1>{title}</h1><div class='tagline'>{tagline}</div>", unsafe_allow_html=True)


def h2(text):
    st.markdown(f"<h2>{text}</h2>", unsafe_allow_html=True)


def placeholder(text):
    st.markdown(f'<div class="placeholder">{text}</div>', unsafe_allow_html=True)


def site_footer():
    st.markdown("<hr><div class='tagline'>Built for NASA Space Apps Challenge 2026, Flame in Freefall. "
                "Data: NASA Physical Sciences Informatics. Not affiliated with or endorsed by NASA.</div>",
                unsafe_allow_html=True)


# ====================================================================== ASK THE EVIDENCE (RAG + Groq)
# Only these columns go to the model, to keep each request small (Groq free tier: about 8,000 tokens per minute).
LLM_TEST_COLUMNS = ["experiment", "test_id", "material", "material_class", "oxygen_pct", "pressure_kpa",
                    "airflow_cms", "flow_direction", "gravity", "suppressant", "outcome_category", "relevance", "why"]
LLM_GAP_COLUMNS = ["pillar", "material_class", "tests_total", "tests_near", "status", "closest_test",
                   "closest_d_o2", "closest_d_kpa", "o2_range_tested", "kpa_range_tested", "gravity_gap"]
EXAMPLES = ["What fabrics were tested in BASS?",
            "How does airflow speed affect flame extinction of fabrics?",
            "Which suppressants did FLEX test, and how did they affect the flames?"]


def secret(name):
    """Environment variable first, then .streamlit/secrets.toml (local) or the Secrets box (Streamlit Cloud)."""
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:  # no secrets file
        return ""


def records(frame, columns):
    return json.loads(frame[columns].to_json(orient="records"))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_retrieve(question):
    return retrieve(question)


@st.cache_data(ttl=300, show_spinner=False)
def cached_index_inventory():
    return index_inventory()


# Streamlit reruns the whole page on every click; caching stops each rerun from calling Groq again.
@st.cache_data(ttl=3600, show_spinner=False)
def cached_answer(question, passages, context, model, _api_key):
    return answer(question, passages, context, _api_key, model)


def scenario_context(preset, pillar, o2_tol, kpa_tol, w_o2, w_kpa):
    """The same closest tests and gaps the scenario report shows, as data for the model."""
    if preset is None:
        return {"scenario": None}
    pil = None if pillar == "All" else pillar
    gaps = gap(df, preset, o2_tol=o2_tol, kpa_tol=kpa_tol)
    return {"scenario": {"name": preset, **PRESETS[preset]}, "pillar": pillar,
            "tolerance": {"o2_pp": o2_tol, "kpa": kpa_tol},
            "ranked_tests": records(rank(df, preset, pillar=pil, n=5, w_o2=w_o2, w_kpa=w_kpa), LLM_TEST_COLUMNS),
            "gaps": records(gaps if pil is None else gaps[gaps["pillar"] == pil], LLM_GAP_COLUMNS)}


def run_question(question, context):
    result = {"question": question, "context": context, "passages": [], "text": None, "note": None, "error": None}
    question_error = validate_question(question)
    if question_error:
        result["error"] = question_error
        return result
    try:
        with st.spinner("Searching NASA documents…"):
            result["passages"] = cached_retrieve(question)
        if not result["passages"]:
            result["note"] = "No NASA documents are indexed yet. Run python scripts/ingest_documents.py first."
            return result
        api_key, model = secret("GROQ_API_KEY"), secret("GROQ_MODEL")
        if not api_key or not model:
            result["note"] = ("Showing the matching passages only. Set GROQ_API_KEY and GROQ_MODEL "
                              "to get a written answer.")
            return result
        with st.spinner("Reading the evidence and writing an answer…"):
            result["text"] = cached_answer(question, result["passages"], context, model, api_key)
    except (ValueError, RuntimeError, OSError) as exc:
        result["error"] = str(exc)
    return result


def context_line(context):
    if context["scenario"] is None:
        return ("Answered from NASA documents only. Choose a mission scenario above to also give it the closest "
                "NASA tests and the evidence gaps for that scenario.")
    tol = context["tolerance"]
    return (f"Also given the 5 closest tests and the evidence gaps for {context['scenario']['name']} "
            f"({PILLAR_LABELS[context['pillar']].split(':')[0].lower()}, tolerance ±{tol['o2_pp']:g} pp O₂ and "
            f"±{tol['kpa']:g} kPa), computed from the data table as in the scenario report.")


def show_answer(last):
    if last["error"]:
        alert("bad", escape(last["error"]), label="Error")
    elif last["text"] in (NO_EVIDENCE, NO_PASSAGES):
        alert("info", "<b>The NASA documents in this atlas do not answer this question.</b> The atlas only knows "
                      "what NASA's microgravity fire reports say, so it cannot answer general questions "
                      "(like <i>what is fire?</i>) or compare things the reports never compare.<br>"
                      "Try naming an experiment, a material or a condition, for example: <i>Which materials "
                      "had the highest flame spread rate in BASS-II?</i> or <i>How did oxygen concentration "
                      "change flame spread over PMMA?</i>", label="No answer")
    elif last["text"]:
        with st.container(border=True):
            st.markdown(safe_model_markdown(last["text"]))
            st.caption("AI-generated from the sources below. Check them: a citation does not guarantee the claim is right.")
    if last["note"]:
        alert("info", escape(last["note"]), label="Note")
    if last["passages"]:
        st.markdown("**Sources** (open one to read the passage)")
        for i, p in enumerate(last["passages"], 1):
            with st.expander(f"[S{i}] {p['title']}, PDF page {p['page']}"):
                st.write(p["text"])
                source = safe_http_url(p.get("source_page_url") or p.get("pdf_url"))
                if source:
                    st.link_button("Open source PDF", source.split("#")[0] + f"#page={p['page']}")
                else:
                    st.caption("Source link unavailable: the PDF URL is not a valid http(s) address.")
    st.caption(context_line(last["context"]))


def select_example(question):
    """Update the question before Streamlit renders the form again."""
    st.session_state["ask_question"] = question
    st.query_params["q"] = question


def ask_page():
    page_header("ask", "Ask the evidence",
                "Answers come only from NASA documents, with citations. "
                "If the documents do not cover your question, the assistant says so.")
    presets = list(PRESETS)
    # Seed shared links once; URL updates must not reset the question widget.
    if "ask_question" not in st.session_state:
        st.session_state["ask_question"] = qp.get("q", "")
    with st.form("ask", border=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="top")
        question = c1.text_input("Question", key="ask_question", label_visibility="collapsed",
                                 placeholder="e.g. In the high-oxygen FLEX tests, did flames go out on their own?")
        asked = c2.form_submit_button("Ask", type="primary", width="stretch")
        s1, s2 = st.columns(2)
        preset = s1.selectbox("Mission scenario (optional)", presets, index=pick_index(presets, qp.get("scenario")),
                              placeholder="None: answer from documents only",
                              help="If you choose one, the answer also gets the 5 closest NASA tests and the evidence "
                                   "gaps for that scenario, computed from the data table (not by AI).")
        pillar = s2.selectbox("Fire safety pillar", PILLARS, index=pick_index(PILLARS, qp.get("pillar")) or 0,
                              format_func=PILLAR_LABELS.get, help=f"{tip('pillar')} Only used with a scenario.")
    question = question.strip()
    if asked:  # keep everything in the URL so a reload or a shared link shows the same answer
        params = {"page": "ask", "q": question, "scenario": preset, "pillar": None if pillar == "All" else pillar}
        params.update({k: qp.get(k) for k in ("o2_tol", "kpa_tol", "w_o2", "w_kpa")})
        st.query_params.from_dict({k: v for k, v in params.items() if v})

    context = scenario_context(preset, pillar, qfloat("o2_tol", O2_TOL), qfloat("kpa_tol", KPA_TOL),
                               qfloat("w_o2", 1.0), qfloat("w_kpa", 1.0))
    last = st.session_state.get("ask_result")
    if question and (asked or last is None or last["question"] != question or last["context"] != context):
        last = run_question(question, context)
        st.session_state["ask_result"] = last

    if not question or last is None or last["question"] != question:
        st.markdown("**Not sure what to ask? Try one of these:**")
        for i, example in enumerate(EXAMPLES):
            st.button(example, key=f"example_{i}", on_click=select_example, args=(example,))
        return

    show_answer(last)
    if preset:
        st.markdown(f"<a href='{carry(page='scenario')}' target='_self'>See the full scenario report for "
                    f"{escape(preset)} →</a>", unsafe_allow_html=True)


def _ieee_citation_html(ref):
    """Escape an IEEE citation and make its DOI or source URL clickable."""
    citation = escape(ref.get("citation", ""))
    target = (f"https://doi.org/{ref['doi']}" if ref.get("doi")
              else safe_http_url(ref.get("url")))
    if not target:
        return citation
    url = escape(target, quote=True)
    label = escape(target)
    return citation.replace(
        label,
        f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{label}</a>",
        1,
    )


def reference_line(ref, show_citation=False, number=None):
    """One reference: compact metadata, or the full IEEE citation on the About page."""
    prefix = f"[{number}] " if number is not None else ""
    title = escape(ref.get("title", "Untitled reference"))
    year = f" ({escape(str(ref['year']))})" if ref.get("year") else ""
    ident = f" <code>{escape(ref['identifier'])}</code>" if ref.get("identifier") else ""
    url = safe_http_url(ref.get("url"))
    if ref.get("doi"):
        doi = escape(ref["doi"], quote=True)
        link = (f"<a href='https://doi.org/{doi}' target='_blank' rel='noopener noreferrer'>"
                f"DOI {doi}</a>")
    elif url:
        link = (f"<a href='{escape(url, quote=True)}' target='_blank' rel='noopener noreferrer'>"
                "source</a>")
    else:
        link = "no link recorded"
    role = f" · {escape(ref['role'])}" if ref.get("role") else ""
    authors = (f"<div class='ref-authors'>{escape(ref['authors'])}</div>"
               if ref.get("authors") else "")
    if show_citation and ref.get("citation"):
        meta = " · ".join(part for part in [
            f"<code>{escape(ref['identifier'])}</code>" if ref.get("identifier") else "",
            escape(ref["role"]) if ref.get("role") else "",
            f"License: {escape(ref['license'])}" if ref.get("license") else "",
        ] if part)
        meta = f"<div class='ref-meta'>{meta}</div>" if meta else ""
        st.markdown(f"<div class='ref-item'>{prefix}{_ieee_citation_html(ref)}{meta}</div>",
                    unsafe_allow_html=True)
        return
    st.markdown(f"<div class='ref-item'><b>{title}</b>{year}{ident} — {link}{role}{authors}</div>",
                unsafe_allow_html=True)


def render_references(experiment, show_citation=False, start=1):
    """Render one experiment's references and return the next IEEE list number."""
    refs = references_for(experiment)
    if not refs:
        st.caption("No references recorded for this experiment.")
        return start
    number = start
    for ref in refs:
        reference_line(ref, show_citation=show_citation,
                       number=number if show_citation else None)
        number += 1
    return number


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
        st.markdown(f"**Original NASA note:** {escape(val(r.outcome_raw))}")
        st.markdown(f"**Test conditions as logged:** {escape(val(r.condition_detail))}")
        if not pd.isna(r.assumed_fields):
            alert("info", f"{term('Nominal value', 'nominal')} filled in for: {escape(str(r.assumed_fields))} "
                          "(not logged per test).", label="Note")
        source = f"Source: {escape(str(r.source_file))} · {escape(str(r.source_location))}"
        ref = dataset_reference(r.experiment)
        if ref:
            label = escape(ref.get("identifier") or "PSI dataset")
            target = (f"https://doi.org/{ref['doi']}" if ref.get("doi")
                      else safe_http_url(ref.get("url")))
            if target:
                source += f" · Dataset [{label}]({target})"
        st.caption(source)


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
            msg = (f"<b>{term(r.material_class)}</b>: {int(r.tests_near)} of {int(r.tests_total)} tests "
                   f"within {tol}. Closest: {escape(str(r.closest_test))} at {r.closest_o2_pct:g}% {o2}, "
                   f"{r.closest_kpa:g} {kpa} (<b>{r.closest_d_o2:+.1f} {pp} O₂, "
                   f"{r.closest_d_kpa:+.1f} kPa</b> from the scenario). "
                   f"Tested: {escape(str(r.o2_range_tested))} O₂, {escape(str(r.kpa_range_tested))}, "
                   f"gravity: {term(r.gravity_tested)}.")
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
    box = (f" The dashed box is what counts as close ({term('tolerance')})." if p is not None
           else " Choose a scenario to place it on the map.")
    pillars = ", ".join(term(k) for k in PILLAR_COLORS)
    st.caption(f"{len(pts)} tests. Colour = {term('pillar')} ({pillars}); shape = experiment "
               f"({' · '.join(term(e) for e in EXPERIMENT_SYMBOLS)}). Points overlap where conditions were identical: "
               f"{stacked.max()} tests sit at {top[0]:g}% O₂, {top[1]:g} kPa. Values are exact (no jitter).{box}",
               unsafe_allow_html=True)


def key_finding(gaps, preset, p, o2_tol, kpa_tol):
    """One plain-language summary of the gap table: which fuel groups have tests near the scenario, which do not."""
    def names(rows):
        return ", ".join(f"{term(r.material_class)} ({escape(str(r.pillar))})" for r in rows)

    near = [r for r in gaps.itertuples() if r.status != "GAP"]
    far = [r for r in gaps.itertuples() if r.status == "GAP"]
    box = f"±{o2_tol:g} {term('pp')} {term('O₂', 'o2')} and ±{kpa_tol:g} {term('kPa', 'kpa')}"
    if near:
        text = (f"<b>{len(near)} of {len(gaps)}</b> fuel groups have NASA tests within {box} of "
                f"{escape(preset)}: {names(near)}.")
        if far:
            text += f" <b>No tests near these conditions</b> for {names(far)}."
    else:
        text = f"<b>No NASA test</b> in this atlas comes within {box} of {escape(preset)}, for any fuel group."
    if p["gravity"] not in set(df["gravity"]):
        text += (f" And none ran at {term(p['gravity'] + ' gravity', p['gravity'])}: "
                 f"every test is from {term('microgravity')}.")
    alert("info", text, label="Key finding")


def scenario_page():
    page_header("scenario", "Scenario report",
                f"How close NASA's {len(df)} microgravity fire tests come to the atmosphere a crew would breathe. "
                "Every number on this page is computed from NASA tables, not by AI.")
    presets = list(PRESETS)
    c1, c2 = st.columns([1.2, 1])
    preset = c1.selectbox("Mission scenario", presets, index=pick_index(presets, qp.get("scenario")),
                          placeholder="Choose a scenario…", help=tip("scenario"), key="sc_preset")
    pillar = c2.selectbox("Fire safety pillar", PILLARS, index=pick_index(PILLARS, qp.get("pillar")) or 0,
                          format_func=PILLAR_LABELS.get, help=tip("pillar"), key="sc_pillar")
    pil = None if pillar == "All" else pillar
    set_param("scenario", preset)  # so the page links and a reload keep the choice
    set_param("pillar", pil)
    p = PRESETS.get(preset)
    pts = df if pil is None else df[df["pillar"] == pil]

    if p is None:
        alert("info", "<b>Choose a mission scenario above</b> (ISS cabin, Moon base, exploration atmosphere, "
                      "spacesuit or Mars habitat). This report then shows:<br>"
                      "1. A <b>key finding</b> in one sentence,<br>"
                      "2. <b>Where the evidence stops</b>: which materials have no NASA tests near the scenario,<br>"
                      "3. The <b>closest NASA tests</b>, with the original NASA notes,<br>"
                      "4. A <b>map</b> of every test's oxygen and pressure, with the scenario on it.", label="Start here")
        h2("Test conditions map")
        conditions_map(pts, None, None, None, None)
        return

    st.markdown(f"<div class='facts'><b>{escape(preset)}</b>: {p['o2']:g}% {term('O₂', 'o2')} at {p['kpa']:g} "
                f"{term('kPa', 'kpa')} ({p['kpa'] / 6.895:.1f} {term('psi')}), {term(p['gravity'])} gravity. "
                f"<span class='src'>Source: {escape(p['source'])}</span></div>", unsafe_allow_html=True)

    with st.expander("Adjust what counts as close (advanced)"):
        st.markdown("A test counts as **close** when it is inside **both** tolerances. Wider tolerances turn more "
                    "gaps into covered. The weights only change the order of the closest-tests table.")
        t1, t2 = st.columns(2)
        o2_tol = t1.slider("Oxygen tolerance (± percentage points)", 0.5, 15.0, O2_TOL, 0.5, key="o2_tol",
                           help=f"{tip('tolerance')} {tip('pp')}")
        kpa_tol = t2.slider("Pressure tolerance (± kPa)", 1.0, 60.0, KPA_TOL, 1.0, key="kpa_tol",
                            help=f"{tip('tolerance')} {tip('kpa')}")
        w1, w2, w3 = st.columns(3)
        w_o2 = w1.slider("Oxygen weight", 0.0, 3.0, 1.0, 0.1, key="w_o2", help=tip("weight"))
        w_kpa = w2.slider("Pressure weight", 0.0, 3.0, 1.0, 0.1, key="w_kpa", help=tip("weight"))
        n_rank = w3.slider("Tests to show", 5, 50, 10, 5, key="n_rank", help="How many of the closest tests to list.")
        st.caption(f"distance = √((w_O₂ · ΔO₂ / {O2_SCALE:g})² + (w_P · ΔP / {KPA_SCALE:g})²); "
                   f"relevance = 1 / (1 + distance). {O2_SCALE:g} O₂ points count the same as {KPA_SCALE:g} kPa.")
    st.caption(f"Close means {p['o2'] - o2_tol:g}–{p['o2'] + o2_tol:g}% O₂ and "
               f"{p['kpa'] - kpa_tol:g}–{p['kpa'] + kpa_tol:g} kPa.")

    ranked = rank(df, preset, pillar=pil, n=n_rank, w_o2=w_o2, w_kpa=w_kpa)
    gaps = gap(df, preset, o2_tol=o2_tol, kpa_tol=kpa_tol)
    gaps = gaps if pil is None else gaps[gaps["pillar"] == pil]

    key_finding(gaps, preset, p, o2_tol, kpa_tol)
    h2("Where the evidence stops")
    gap_banners(gaps, o2_tol, kpa_tol)
    h2("Closest NASA tests")
    evidence_table(ranked, p)
    h2("Test conditions map")
    conditions_map(pts, preset, p, o2_tol, kpa_tol)

    h2("Ask about this scenario")
    st.markdown("<div class='hatnote'>The answer uses NASA documents plus the closest tests and gaps on this page, "
                "with your settings.</div>", unsafe_allow_html=True)
    with st.form("ask_scenario", border=False):
        f1, f2 = st.columns([5, 1], vertical_alignment="top")
        q = f1.text_input("Question about this scenario", label_visibility="collapsed",
                          placeholder=f"e.g. Which tests are closest to {preset}, and what happened in them?")
        if f2.form_submit_button("Ask", type="primary", width="stretch") and q.strip():
            params = {"page": "ask", "q": q.strip(), "scenario": preset, "pillar": pil}
            for name, value, default in (("o2_tol", o2_tol, O2_TOL), ("kpa_tol", kpa_tol, KPA_TOL),
                                         ("w_o2", w_o2, 1.0), ("w_kpa", w_kpa, 1.0)):
                if value != default:
                    params[name] = f"{value:g}"
            st.query_params.from_dict({k: v for k, v in params.items() if v})
            st.rerun()


INDEX_TOPIC = {"BASS": "Prevention", "BASS-II": "Prevention", "FLEX": "Suppression",
               "SAME": "Detection", "SAME-R": "Detection"}
INDEX_TOPIC_ORDER = ("Prevention", "Detection", "Suppression", "Background / cross-cutting")


def index_tables(inventory):
    """Group the active PDF index by atlas topic, plus one row per PDF."""
    rows = []
    for item in inventory:
        topic = INDEX_TOPIC.get(item.get("experiment", ""), INDEX_TOPIC_ORDER[-1])
        rows.append({"PDF": item.get("title", "Untitled PDF"), "Topic": topic,
                     "Embeddings": item.get("passages", 0), "URL": item.get("url", "")})
    grouped = {}
    for row in rows:
        bucket = grouped.setdefault(row["Topic"], {"Topic": row["Topic"], "PDFs": 0, "Embeddings": 0})
        bucket["PDFs"] += 1
        bucket["Embeddings"] += row["Embeddings"]

    def topic_order(topic):
        return INDEX_TOPIC_ORDER.index(topic) if topic in INDEX_TOPIC_ORDER else len(INDEX_TOPIC_ORDER)

    summary = sorted(grouped.values(), key=lambda row: (topic_order(row["Topic"]), row["Topic"]))
    summary.append({"Topic": "Total", "PDFs": len(rows),
                    "Embeddings": sum(row["Embeddings"] for row in rows)})
    rows.sort(key=lambda row: (topic_order(row["Topic"]), row["PDF"].casefold()))
    return summary, rows


def index_summary_html(summary):
    body = "".join(
        f"<tr{' class=total' if row['Topic'] == 'Total' else ''}>"
        f"<td>{escape(row['Topic'])}</td>"
        f"<td class='num'>{row['PDFs']:,}</td>"
        f"<td class='num'>{row['Embeddings']:,}</td></tr>"
        for row in summary)
    return ("<table class='index-table' aria-label='PDFs and embeddings by topic'>"
            "<thead><tr><th scope='col'>Topic</th><th scope='col' class='num'>PDFs</th>"
            "<th scope='col' class='num'>Embeddings</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def index_pdf_html(rows):
    def pdf_link(row):
        title = escape(row["PDF"])
        url = safe_http_url(row.get("URL", ""))
        if not url:
            return title
        return (f"<a href='{escape(url, quote=True)}' target='_blank' rel='noopener noreferrer' "
                f"title='Open PDF in a new tab'>{title} ↗</a>")

    body = "".join(
        f"<tr><td>{pdf_link(row)}</td><td>{escape(row['Topic'])}</td>"
        f"<td class='num'>{row['Embeddings']:,}</td></tr>"
        for row in rows)
    return ("<table class='index-table' aria-label='Indexed PDFs, topics and embeddings'>"
            "<thead><tr><th scope='col'>PDF</th><th scope='col'>Topic</th>"
            "<th scope='col' class='num'>Embeddings</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def about_page():
    page_header("about", "About this atlas",
                "What Flame in Freefall does, how to read it, and where its data comes from.")
    main, side = st.columns([2.3, 1], gap="large")
    with side:
        counts = df["experiment"].value_counts()
        st.markdown(f"""
<div class="infobox">
  <span class="ib-title">NASA data in this atlas</span>
  <table>
    <tr><th>Tests</th><td>{len(df)}</td></tr>
    {''.join(f"<tr><th>{term(e)}</th><td>{n}</td></tr>" for e, n in counts.items())}
    <tr><th>Gravity tested</th><td>{', '.join(term(g) for g in sorted(df.gravity.unique()))}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    with main:
        h2("What it does")
        st.markdown(
            f"**Flame in Freefall** gathers NASA's {term('microgravity')} fire experiments into one table, compares "
            f"them with the atmosphere a crew would breathe (a mission {term('scenario')}), and shows where the "
            f"evidence stops. It covers the three {term('pillars', 'pillar')} of spacecraft fire safety: "
            f"**{term('prevention')}** ({term('BASS')}, {term('BASS-II')}), "
            f"**{term('detection')}** ({term('SAME')}, {term('SAME-R')}) and **{term('suppression')}** ({term('FLEX')}). "
            "Words with a dotted underline show a short definition when you hover over them.",
            unsafe_allow_html=True)
        placeholder("TODO [Pitch]: 2-3 sentence summary + key finding")

        h2("How to use it")
        st.markdown(
            "<ul>"
            "<li><a href='?page=ask' target='_self'><b>Ask the evidence</b></a>: type a question and get an answer "
            "written from NASA documents, with the passages it used.</li>"
            "<li><a href='?page=scenario' target='_self'><b>Scenario report</b></a>: pick a mission atmosphere "
            "(for example a Moon base) and see which NASA tests come close and where there are none.</li>"
            "<li><a href='?page=explore' target='_self'><b>Data explorer</b></a>: filter all tests by any column "
            "and download them.</li></ul>", unsafe_allow_html=True)

        h2("Reading a scenario report")
        st.markdown(
            f"- **Mission scenario**: the oxygen %, pressure and gravity a crew would live in.\n"
            f"- **Fire safety pillar**: show only prevention, detection or suppression tests.\n"
            f"- **Key finding**: one sentence summarising the evidence gaps below it.\n"
            f"- **Where the evidence stops**: for each pillar and {term('fuel type')}, is any test close to the "
            f"scenario? {term('Covered', 'covered')}, {term('Gravity gap', 'covered except gravity')} or "
            f"{term('Gap', 'gap')}.\n"
            f"- **Closest NASA tests**: every test sorted by {term('relevance')} (1.00 = same oxygen and pressure). "
            "Pick one to read NASA's original note.\n"
            "- **Test conditions map**: every test as a dot; the star is the scenario and the dashed box is "
            "what counts as close.\n"
            f"- **Adjust what counts as close**: the {term('tolerances', 'tolerance')} decide covered versus gap; "
            f"the {term('weights', 'weight')} only change the order of the closest tests.",
            unsafe_allow_html=True)

        h2("How the AI answer works")
        st.markdown(
            "1. A local search (no AI) finds the 5 passages in NASA's PDF reports that best match the question.\n"
            "2. A language model hosted by Groq writes an answer from those passages only. If you chose a "
            "scenario, it also gets the closest tests and gaps from the data table.\n"
            "3. Every claim must cite a passage, [S1] to [S5]. Answers that cite anything else are rejected, and if "
            "the passages do not cover the question, it says so.\n\n"
            "Numbers come from the passages or the data table, never from the model's memory. "
            "Check the passages: a citation does not prove the claim is right.")

        h2("NASA documents indexed")
        inventory = cached_index_inventory()
        if not inventory:
            st.caption("No PDF index is available yet. Run `python scripts/ingest_documents.py` to build it.")
        else:
            summary, rows = index_tables(inventory)
            st.markdown(f"The local search index contains **{len(rows)} PDFs** converted into "
                        f"**{summary[-1]['Embeddings']:,} embedded passages**. Each passage is a 100-word "
                        "overlapping chunk; the table groups them by the atlas topic they support.")
            st.markdown(index_summary_html(summary), unsafe_allow_html=True)
            with st.expander("See the PDFs in the index"):
                st.markdown(index_pdf_html(rows), unsafe_allow_html=True)
                st.caption("Click a PDF title to open it in a new tab.")
            st.caption("Topics come from the experiment recorded in `documents/manifest.csv`; PDFs without "
                       "a single experiment are grouped as background / cross-cutting.")

        h2("Experiments")
        for name, pillar_label, desc, experiments in [
            ("BASS / BASS-II", "Prevention",
             f"{term('BASS')} / {term('BASS-II')}: how solid materials ({term('PMMA', 'pmma')} plastic, fabric such as "
             f"{term('Nomex', 'nomex')}, wax) ignite, spread and go out in low-speed {term('airflow')}. "
             f"Run in the {term('MSG', 'msg')}.", ["BASS", "BASS-II"]),
            ("FLEX", "Suppression",
             f"{term('FLEX')}: {term('fuel droplets', 'liquid fuel droplet')} burned in different {term('O₂', 'o2')}, "
             f"pressure and {term('suppressant')} (CO₂, helium) mixtures. Run in the {term('CIR', 'cir')} with the "
             f"{term('MDCA', 'mdca')}.", ["FLEX"]),
            ("SAME / SAME-R", "Detection",
             f"{term('SAME')} / {term('SAME-R')}: smoke particle sizes from heated spacecraft materials and how "
             f"smoke detectors respond. Includes a {term('reference aerosol')}.", ["SAME", "SAME-R"]),
        ]:
            with st.expander(f"{name}: {pillar_label}"):
                st.markdown(desc, unsafe_allow_html=True)
                for experiment in experiments:
                    if len(experiments) > 1:
                        st.markdown(f"**{term(experiment)}**", unsafe_allow_html=True)
                    render_references(experiment)
                placeholder("TODO: summary card, key findings with page citations, small chart")

        h2("Applications on Earth")
        placeholder("TODO [Pitch]: NASA spinoffs: firefighter air tanks, smoke detector, protective clothing")

        h2("Methodology and use of AI")
        st.markdown(
            f"- **Data:** 678 tests from {term('PSI', 'psi_db')}-25, PSI-26, PSI-69, PSI-101, PSI-102, "
            "built with `build_master.py`.\n"
            f"- **Ranking:** distance on O₂ and pressure ({term('relevance')}, {term('weights', 'weight')} and "
            f"{term('tolerances', 'tolerance')} in the scenario report).\n"
            f"- **{term('Outcome', 'outcome')} categories** for BASS/BASS-II are our reading of crew notes; "
            "the original note is always shown.\n"
            "- **Use of AI:** TODO list tools used (code assistance, Groq for answers).", unsafe_allow_html=True)

        h2("References")
        st.markdown("Datasets and reports used by this atlas, in IEEE style. Dataset DOIs resolve to the canonical "
                    "NASA PSI landing page, where the experimental tables and files are published. "
                    "Licenses are shown for each source.")
        number = 1
        for experiment in ["BASS", "BASS-II", "FLEX", "SAME", "SAME-R"]:
            if references_for(experiment):
                st.markdown(f"**{term(experiment)}**", unsafe_allow_html=True)
                number = render_references(experiment, show_citation=True, start=number)
        st.caption("Reference metadata is maintained in `references.csv` and linked to experiments in "
                   "`experiment_references.csv`. See `THIRD_PARTY_NOTICES.md` for the full license audit. "
                   "Row-level provenance remains visible in the test details.")


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
    page_header("explore", "Data explorer", f"Filter all {len(df)} NASA microgravity fire tests by any column. "
                "Every row keeps its source file and page.")

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
    st.download_button("Download these tests (.csv)", csv_bytes(res),
                       file_name="flame_in_freefall_tests.csv", mime="text/csv")


# ====================================================================== router
page = qp.get("page")
if page == "article":  # old links: the article page was split into Ask and Scenario report
    page = "ask" if qp.get("q") else "scenario"
    st.query_params["page"] = page
if page in ("ask", "scenario", "explore", "about"):
    {"ask": ask_page, "scenario": scenario_page, "explore": explore, "about": about_page}[page]()
    site_footer()
else:
    home()
