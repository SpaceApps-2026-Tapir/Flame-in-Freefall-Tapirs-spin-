"""Home-page overview: what NASA has tested, where the gap is, and which experiment would close it.

Three blocks, one message each, almost no text:
  1. What we know   - pillar x scenario matrix of tests near each condition
  2. How far        - how close the nearest NASA test is to the Moon base, per pillar
  3. Test next      - the condition inside the Moon-habitat range that one new experiment
                      would cover best, with coverage before -> after

All numbers come from evidence.gap() / the master table, so they match the rest of the atlas.
Written with AI assistance (Claude); see README "Use of AI".
"""
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evidence import PRESETS, gap

PILLARS = [("prevention", "Prevention", "Will it catch fire?"),
           ("detection", "Detection", "Will we notice?"),
           ("suppression", "Suppression", "Can we put it out?")]
COLUMNS = [("ISS cabin", "ISS today"), ("Moon base (challenge brief)", "Moon base"),
           ("Exploration atmosphere", "Exploration air"), ("Mars habitat (assumed)", "Mars habitat"),
           ("Spacesuit", "Spacesuit")]
MOON = "Moon base (challenge brief)"
TOL_O2, TOL_KPA = 3.0, 10.0  # same default "near" box as the article's gap section

# Moon-habitat range spanned by the atlas presets (Moon base -> exploration atmosphere)
REGION = dict(o2=(PRESETS[MOON]["o2"], PRESETS["Exploration atmosphere"]["o2"]),
              kpa=(PRESETS["Exploration atmosphere"]["kpa"], PRESETS[MOON]["kpa"]))
WHAT_TO_BURN = {"prevention": "Solid materials (PMMA, fabrics)",
                "detection": "Smoke from Kapton and Teflon",
                "suppression": "Fuel droplets with CO₂ or helium"}

CSS = """
<style>
  /* one visual system for the whole landing: same radius, border, spacing and type everywhere */
  .ov {--r: .6rem; --gap: .75rem; margin: 3.2rem 0 0;}
  .ov h3 {font-family: Georgia, serif; font-weight: 400; font-size: 1.55rem; margin: 0 0 .25rem; color: var(--atlas-text);
          display: flex; align-items: center; gap: .6rem;}
  .ov .step {margin-right: .15rem; display: inline-grid; place-items: center; width: 1.7rem; height: 1.7rem; flex: none; border-radius: 50%;
             background: var(--atlas-link); color: var(--atlas-panel); font: 700 .9rem system-ui, sans-serif;}
  .ov .lead {color: var(--atlas-muted); font-size: .95rem; margin: 0 0 1rem 2.3rem; max-width: 46rem;}
  .ov .foot {font-size: .78rem; color: var(--atlas-note); margin-top: .7rem;}
  .ov .box, .ov a.hcard, .ov a.branch, .ov .card {border: 1px solid var(--atlas-border); border-radius: var(--r);
          background: var(--atlas-panel); text-decoration: none !important; color: var(--atlas-text);}
  .ov a.hcard:hover, .ov a.branch:hover {border-color: var(--atlas-link); box-shadow: 0 0 0 1px var(--atlas-link);}
  .ov a:focus-visible {outline: 2px solid var(--atlas-link); outline-offset: 3px;}
  .ov.tight {margin: .9rem 0 .8rem;}
  .ov .verdict {font-size: 1.05rem; line-height: 1.6; padding: .8rem 1rem; border-radius: var(--r);
                border-left: 4px solid var(--atlas-link); background: var(--atlas-panel); color: var(--atlas-text);}
  .ov .card .p, .ov .kicker {font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--atlas-muted);}

  /* map: scenarios x pillars */
  .ov .mapwrap {overflow-x: auto;}
  /* evidence table: hairlines like the rest of the atlas, colour only in the numbers */
  .ov table.et {width: 100%; min-width: 680px; border-collapse: collapse !important; border: none !important; table-layout: fixed;}
  .ov table.et :is(th, td) {border: none !important; border-bottom: 1px solid var(--atlas-divider) !important;
          background: transparent !important; padding: .55rem .6rem; vertical-align: middle;}
  .ov table.et thead th {border-bottom: 2px solid var(--atlas-border) !important; text-align: left; vertical-align: bottom;}
  .ov table.et th.rowh {text-align: left; width: 22%;}
  .ov table.et th a {font-weight: 700; font-size: .95rem; text-decoration: none;}
  .ov table.et th a:hover {text-decoration: underline;}
  .ov table.et th span {display: block; font-weight: 400; font-size: .76rem; color: var(--atlas-muted); margin-top: .1rem;}
  .ov table.et th small {display: block; font-weight: 400; font-size: .7rem; color: var(--atlas-note);}
  .ov table.et th svg.teaser {margin: .4rem 0 .35rem; max-width: 9rem;}
  .ov table.et th.corner {font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
          color: var(--atlas-muted); vertical-align: bottom;}
  .ov table.et td {text-align: center; padding: 0;}
  .ov table.et td a {display: block; padding: .8rem .3rem; text-decoration: none !important; border-radius: .3rem;}
  .ov table.et td a:hover {background: var(--atlas-placeholder);}
  .ov table.et td b {display: block; font-size: 1.4rem; font-weight: 600; line-height: 1.1;}
  .ov table.et td span {font-size: .74rem; color: var(--atlas-muted);}
  .ov table.et td span::before {content: "●"; margin-right: .3rem; font-size: .7rem;}
  .ov table.et td.ok b, .ov table.et td.ok span::before {color: light-dark(#0b8a0b, #4ade80);}
  .ov table.et td.warn b, .ov table.et td.warn span::before {color: light-dark(#b07d00, #e7b74a);}
  .ov table.et td.bad b, .ov table.et td.bad span::before {color: light-dark(#c03535, #ff8091);}
  .ov .mapgrid {display: grid; grid-template-columns: 1.35fr repeat(5, minmax(6.4rem, 1fr)); gap: var(--gap); min-width: 720px;}
  .ov a.hcard {display: flex; flex-direction: column; gap: .2rem; padding: .6rem .65rem; line-height: 1.2;}
  .ov a.hcard b {color: var(--atlas-link); font-size: .95rem;}
  .ov a.hcard span {color: var(--atlas-muted); font-size: .74rem;}
  .ov a.hcard em {font-style: normal; font-size: .68rem; color: var(--atlas-note); margin-top: auto;}
  .ov a.cell {display: flex; flex-direction: column; justify-content: center; align-items: center; gap: .15rem;
              border-radius: var(--r); border: 1px solid transparent; border-left-width: 5px; text-decoration: none !important;}
  .ov a.cell.ok {border-left-color: light-dark(#0ca30c, #4ade80);}
  .ov a.cell.warn {border-left-color: light-dark(#e0a100, #e7b74a);}
  .ov a.cell.bad {border-left-color: light-dark(#d03b3b, #ff8091);}
  .ov a.cell:hover {box-shadow: 0 0 0 1px currentColor;}
  .ov .n {font-size: 1.6rem; font-weight: 700; line-height: 1;}
  .ov .s {font-size: .66rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;}
  .ov .ok   {background: light-dark(#e8f5e9, #123124); color: light-dark(#14532d, #c0f5d5); border-color: light-dark(#0ca30c, #4ade80);}
  .ov .warn {background: light-dark(#fff7e0, #342b16); color: light-dark(#5c4400, #ffe6a3); border-color: light-dark(#e0a100, #e7b74a);}
  .ov .bad  {background: light-dark(#fdecec, #381c24); color: light-dark(#7f1d1d, #ffc7ce); border-color: light-dark(#d03b3b, #ff8091);}

  /* cards (findings, simulator, next tests) */
  .ov .cards {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--gap);}
  .ov .card {padding: .85rem 1rem;}
  .ov .card .cond {font-size: 1.45rem; font-weight: 700; margin: .3rem 0 .15rem;}
  .ov .card .what {font-size: .85rem; color: var(--atlas-muted);}
  .ov .card .cov {margin-top: .6rem; font-size: .85rem; color: var(--atlas-muted);}
  .ov .card .cov b {color: var(--atlas-text); font-size: 1.05rem;}

  /* branches */
  .ov .branches {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--gap);}
  .ov a.branch {display: flex; flex-direction: column; gap: .25rem; padding: .75rem .85rem;}
  .ov a.branch b {color: var(--atlas-link); font-size: 1rem;}
  .ov a.branch span {color: var(--atlas-muted); font-size: .8rem;}

  /* mini maps */
  .ov svg.teaser {width: 100%; height: auto; margin: .3rem 0; display: block;}
  .ov svg .tbg {fill: light-dark(#ffffff, #151b26); stroke: var(--atlas-divider);}
  .ov svg .tbox {fill: rgba(208,59,59,.14); stroke: #d03b3b; stroke-width: 1.2;}
  .ov svg .tpt {fill: light-dark(#b9bcc2, #5b6576);}
  .ov svg .tpt.on {fill: var(--atlas-link);}
  .ov svg .tsc {fill: #d03b3b; stroke: white; stroke-width: 1.5;}
  .ov svg .tlab {font: 700 9px system-ui, sans-serif; fill: #d03b3b;}
  .ov svg .trow {fill: light-dark(#e3e6ea, #2a3342);}

  /* hero thesis */
  .thesis {text-align: center; margin: .4rem auto 1.2rem; max-width: 40rem;}
  .thesis p {font-family: Georgia, serif; font-size: 1.3rem; line-height: 1.45; color: var(--atlas-text); margin: 0 0 .9rem;}
  .thesis p b {color: #d03b3b; font-weight: 400;}
  .chips {display: flex; justify-content: center; flex-wrap: wrap; gap: .5rem;}
  .chip {border: 1px solid var(--atlas-border); border-radius: 2rem; padding: .3rem .8rem; font-size: .82rem;
         color: var(--atlas-muted); background: var(--atlas-panel);}
  .chip b {color: var(--atlas-text);}
  .chip.red b {color: #d03b3b;}

  @media (max-width: 760px) {.ov .cards, .ov .branches {grid-template-columns: 1fr;} .ov .lead {margin-left: 0;}}
</style>"""


def section(n, title, lead):
    badge = f"<span class='step'>{n}</span>" if n else ""
    return (f"<h3>{badge}{title}</h3>" + (f"<div class='lead'>{lead}</div>" if lead else ""))


def tidy(fig, height):
    """Same type, margins and transparent background for every landing chart."""
    fig.update_layout(height=height, font=dict(family="Source Sans Pro, system-ui, sans-serif", size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=10, b=0))
    return fig


def hero(df):
    """The theme that ties the page together, right under the title."""
    gaps = gap_counts(df)
    solids_moon = gaps["solids_near_moon"]
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"""
<div class='thesis'>
  <p>A Moon base will breathe <b>30–34% oxygen</b>. In that air, a fire may no longer put itself out —
  and NASA has never tested how solid materials burn there.</p>
  <div class='chips'>
    <span class='chip'><b>{len(df)}</b> NASA fire tests on the ISS</span>
    <span class='chip red'><b>{solids_moon}</b> with solids in Moon-base air</span>
    <span class='chip red'><b>0</b> in Moon gravity</span>
  </div>
</div>""", unsafe_allow_html=True)


FLEX_RAW = Path(__file__).resolve().parent / "data" / "raw" / "flex_droplets.csv"
FUELS = [("Methanol", "#eb6834"), ("Heptane", "#2a78d6")]
FEATS = ["o2_pct", "co2_pct", "he_pct", "pressure_kpa", "heptane", "d0_mm"]


@st.cache_data
def self_extinction():
    """Logistic model of P(flame goes out on its own) from the FLEX droplet burns, with bootstrap bands.

    Returns the fitted curves, the oxygen at which half the flames self-extinguish (with a 90% interval),
    binned observations, and the burns run in a Moon-base-like atmosphere.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    d = pd.read_csv(FLEX_RAW)
    d = d[d.pressure_valid & d.d0_mm.notna()].copy()
    d["y"] = d.test_end.eq("extinction").astype(int)
    d["heptane"] = (d.fuel == "Heptane").astype(int)
    model = lambda: LogisticRegression(C=10, max_iter=5000)
    auc = cross_val_score(model(), d[FEATS], d.y, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()

    o2 = np.linspace(10, 36, 105)
    d0 = d.d0_mm.median()
    rng = np.random.default_rng(0)
    curves = {f: [] for f, _ in FUELS}
    o50 = {f: [] for f, _ in FUELS}
    for i in range(400):  # bootstrap
        s = d if i == 0 else d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        m = model().fit(s[FEATS], s.y)
        c, b = dict(zip(FEATS, m.coef_[0])), m.intercept_[0]
        for fuel, _ in FUELS:
            base = b + c["pressure_kpa"] * 101.3 + c["heptane"] * (fuel == "Heptane") + c["d0_mm"] * d0
            curves[fuel].append(1 / (1 + np.exp(-(base + c["o2_pct"] * o2))))
            o50[fuel].append(-base / c["o2_pct"])
    out = {"o2": o2, "auc": auc, "n": len(d)}
    for fuel, _ in FUELS:
        arr = np.array(curves[fuel])
        out[fuel] = dict(fit=arr[0], lo=np.percentile(arr, 5, axis=0), hi=np.percentile(arr, 95, axis=0),
                         o50=np.percentile(o50[fuel], [5, 50, 95]))
    nd = d[d.diluent == "none"].copy()
    nd["bin"] = pd.cut(nd.o2_pct, [0, 15, 18, 21.5, 26, 36])
    out["obs"] = (nd.groupby(["fuel", "bin"], observed=True)
                    .agg(o2=("o2_pct", "mean"), share=("y", "mean"), n=("y", "size")).reset_index())
    moon = d[(d.o2_pct >= REGION["o2"][0]) & (d.pressure_kpa <= REGION["kpa"][1] + 5)]
    out["moon_n"], out["moon_ext"] = len(moon), int(moon.y.sum())
    return out


@st.cache_data(show_spinner=False)
def _simulate(fuel, o2, co2, he, kpa, d0):
    import sim
    return sim.simulate(fuel, o2, co2, he, kpa, d0)


PRESET_AIR = {"moon": dict(o2=30.0, kpa=70.3, co2=0.0, g="Moon (0.16 g)"),
              "iss": dict(o2=21.0, kpa=101.3, co2=0.0, g="Microgravity (ISS)")}


def _set_air(which):
    for k, v in PRESET_AIR[which].items():
        st.session_state[f"fl_{k}"] = v


OUTCOME_COLORS = {"went out by itself": "#1baf7a", "kept burning to the end": "#eda100", "droplet burst": "#eb6834"}
END_LABEL = {"extinction": "went out by itself", "completion": "kept burning to the end", "disruption": "droplet burst"}


def _burn_flow_fig(sim, match):
    d = sim.data()
    on = d.index.isin(match.index)
    dims = {
        "Oxygen": pd.cut(d.o2_pct, [0, 18, 22, 28, 101], labels=["<18%", "18–22% (air)", "22–28%", "≥28% (Moon-like)"]),
        "Pressure": pd.cut(d.pressure_kpa, [0, 80, 110, 1e4], labels=["reduced (<80 kPa)", "sea level", "high"]),
        "Suppressant": d.diluent.map({"none": "none", "CO2": "CO₂", "He": "helium"}),
        "Fuel": d.fuel,
        "The flame": d.test_end.map(END_LABEL),
    }
    fig = go.Figure(go.Parcats(
        dimensions=[dict(label=k, values=v.astype(str)) for k, v in dims.items()],
        line=dict(color=on.astype(int), colorscale=[[0, "rgba(160,160,160,.35)"], [1, "#2a78d6"]], shape="hspline"),
        hoveron="color", arrangement="freeform", labelfont=dict(size=13)))
    return tidy(fig, 360).update_layout(margin=dict(l=110, r=150, t=30, b=10))


def _burn_map_fig(sim, s, match):
    d = sim.data()
    emb, names = sim.burn_map()
    star = sim.place_on_map(s["fuel"], s["o2"], s["co2"], s["he"], s["kpa"], s["d0"])
    fig = go.Figure(go.Scatter3d(x=emb[:, 0], y=emb[:, 1], z=emb[:, 2], mode="markers", name="other NASA burns",
                                 marker=dict(size=3.5, color="#a3a39c", opacity=.7), hoverinfo="skip"))
    m = d.loc[match.index]
    for end, col in OUTCOME_COLORS.items():
        g = m[m.test_end.map(END_LABEL) == end]
        if len(g):
            fig.add_trace(go.Scatter3d(x=emb[g.index, 0], y=emb[g.index, 1], z=emb[g.index, 2], mode="markers",
                                       name=f"like yours: {end}", marker=dict(size=7, color=col),
                                       customdata=g[["test_id", "o2_pct", "pressure_kpa"]].values,
                                       hovertemplate="%{customdata[0]} · %{customdata[1]}% O₂ · "
                                                     "%{customdata[2]:.0f} kPa<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[star[0]], y=[star[1]], z=[star[2]], mode="markers+text", name="your flame",
                               text=["your flame"], textposition="top center",
                               marker=dict(size=10, color="#d03b3b", symbol="diamond", line=dict(color="white", width=2))))
    tidy(fig, 460)
    fig.update_layout(scene=dict(xaxis_title=names[0], yaxis_title=names[1], zaxis_title=names[2], aspectmode="cube",
                                 xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                                 zaxis=dict(showticklabels=False)),
                      legend=dict(orientation="h", y=0))
    return fig


def fire_lab_block():
    """One interactive block: change the air and the fuel, see the chance the flame puts itself out, watch it burn."""
    import sim
    if not sim.RAW.exists():
        return
    for k, v in dict(fuel="Methanol", o2=30.0, kpa=70.3, co2=0.0, d0=3.0, g="Moon (0.16 g)").items():
        st.session_state.setdefault(f"fl_{k}", v)

    st.markdown("<div class='ov'>" + section(2, "Light a fire in space",
                "Change the air, the fuel and the gravity. Will the flame put itself out?") + "</div>",
                unsafe_allow_html=True)
    b1, b2, g_col = st.columns([1.3, 1, 3], vertical_alignment="bottom")
    b1.button("Try it in a Moon base", on_click=_set_air, args=("moon",), type="primary", width="stretch")
    b2.button("ISS air", on_click=_set_air, args=("iss",), width="stretch")

    grav = g_col.segmented_control("Gravity", list(sim.GRAVITY), key="fl_g", label_visibility="collapsed") \
        or "Microgravity (ISS)"
    c = st.columns([1, 1.4, 1.4, 1.4, 1.4])
    fuel = c[0].radio("Fuel", ["Methanol", "Heptane"], key="fl_fuel")
    o2 = c[1].slider("Oxygen (%)", 12.0, 34.0, step=0.5, key="fl_o2")
    kpa = c[2].slider("Pressure (kPa)", 50.0, 110.0, step=0.5, key="fl_kpa")
    co2 = c[3].slider("CO₂ added (%)", 0.0, 60.0, step=1.0, key="fl_co2", help="A suppressant gas mixed into the air.")
    d0 = c[4].slider("Droplet size (mm)", 1.5, 4.5, step=0.1, key="fl_d0")

    s = _simulate(fuel, o2, co2, 0.0, kpa, d0)
    r = self_extinction()
    color = dict(FUELS)[fuel]
    grid = np.linspace(10, 36, 105)

    fig = go.Figure()
    lo_o2, hi_o2 = REGION["o2"]
    fig.add_vrect(x0=lo_o2, x1=hi_o2, fillcolor="rgba(127,127,127,.15)", line_width=0)
    fig.add_annotation(x=(lo_o2 + hi_o2) / 2, y=100, yanchor="bottom", text="Moon base", showarrow=False)
    other, ocol = [f for f in FUELS if f[0] != fuel][0]
    fig.add_trace(go.Scatter(x=grid, y=sim.p_curve(other, co2, 0.0, kpa, d0, grid) * 100, mode="lines",
                             name=other, line=dict(color=ocol, width=1.5, dash="dot"), opacity=.6,
                             hovertemplate=f"{other}: %{{y:.0f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=sim.p_curve(fuel, co2, 0.0, kpa, d0, grid) * 100, mode="lines", name=fuel,
                             line=dict(color=color, width=3),
                             hovertemplate=f"{fuel}: %{{y:.0f}}% at %{{x:.1f}}% O₂<extra></extra>"))
    ob = r["obs"][r["obs"].fuel == fuel]
    fig.add_trace(go.Scatter(x=ob.o2, y=ob.share * 100, mode="markers", showlegend=False,
                             marker=dict(color=color, size=np.clip(ob.n * 1.2, 7, 20), opacity=.45),
                             customdata=ob.n, hovertemplate="NASA: %{y:.0f}% of %{customdata} burns<extra></extra>"))
    fig.add_trace(go.Scatter(x=[o2], y=[s["p_self_ext"] * 100], mode="markers+text", name="Your flame",
                             text=[f"{s['p_self_ext']:.0%}"], textposition="top center",
                             textfont=dict(size=14, color=color),
                             marker=dict(size=17, color=color, symbol="star", line=dict(color="white", width=2)),
                             hoverinfo="skip"))
    tidy(fig, 330)
    fig.update_layout(xaxis=dict(title="Oxygen (%)", range=[10, 36]),
                      yaxis=dict(title="Chance it goes out before the fuel is gone (%)", range=[0, 112]),
                      legend=dict(orientation="h", y=-0.22, x=0))

    left, right = st.columns([1.3, 1], gap="medium")
    with left, st.container(border=True):
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="fl_curve")
    with right:
        theme = getattr(getattr(st.context, "theme", None), "type", None) or "light"
        st.iframe(sim.animation_html(s, height=372, g=sim.GRAVITY[grav], theme=theme), height=380)  # our own HTML/JS

    # what the model says, how sure it is, and what NASA actually ran in this air
    cert = sim.certainty(s)
    cert["says"] = ("goes out by itself before the fuel is gone" if cert["says"] == "goes out by itself"
                    else "keeps burning until the fuel is gone")
    match = sim.matching(fuel, o2, co2, kpa)
    ends = {"extinction": "went out by itself", "disruption": "the droplet burst", "completion": "burned to the end"}
    if len(match):
        tally = match.test_end.map(ends).value_counts()
        nasa = (f"NASA ran <b>{len(match)}</b> {fuel.lower()} burns in this air: "
                + ", ".join(f"{n} {e}" for e, n in tally.items()) + ".")
    else:
        nasa = f"NASA never burned {fuel.lower()} in this air: this part is an estimate, not a measurement."
    gv = sim.GRAVITY[grav]
    if gv == 0:
        gnote = ""
    elif gv >= 1:
        gnote = (" <br><b>On Earth</b> rising hot gas keeps feeding the flame fresh air. The drawing shows it; the "
                 "numbers still come from microgravity burns.")
    else:
        gnote = (f" <br><b>{grav.split(' (')[0]} gravity has never been tested in these experiments:</b> the drawing "
                 "shows hot gas starting to rise, but every number comes from microgravity burns. Short ground "
                 "experiments in simulated partial gravity found some materials burn at lower oxygen than on Earth.")
    st.markdown(f"""<div class='ov tight'><div class='verdict'>
  The model says your flame <b>{cert['says']}</b>, with <b>{cert['certainty']:.0%} certainty</b>.
  On the {cert['k']} most similar NASA burns it was right <b>{cert['local_accuracy']:.0%}</b> of the time.
  It lasts about <b>{s['t_end']:.0f} s</b>. {nasa}{gnote}</div></div>""", unsafe_allow_html=True)

    # the data, filtered to this flame: every NASA burn, the ones like yours highlighted
    v1, v2 = st.columns([3, 1.2], vertical_alignment="bottom")
    view = v1.segmented_control("NASA burns like yours", ["Flow", "Map"], default="Flow", key="fl_view",
                                format_func={"Flow": "From conditions to outcome", "Map": "Look-alike map"}.get)
    v2.markdown(f"<div style='text-align:right'><a href='?page=visual&data=flex&chart=similarity&o2={o2:g}"
                f"&kpa={kpa:g}&fuel={fuel}' target='_self'>Explore around this flame →</a></div>",
                unsafe_allow_html=True)
    with st.container(border=True):
        if view == "Map":
            st.plotly_chart(_burn_map_fig(sim, s, match), theme="streamlit", width="stretch", key="fl_map")
        else:
            st.plotly_chart(_burn_flow_fig(sim, match), theme="streamlit", width="stretch", key="fl_flow")
        st.caption(f"Highlighted: the {len(match)} NASA burns with {fuel.lower()} within ±3 points of O₂, ±10 kPa "
                   f"and ±10 points of CO₂ of your flame. Grey: all other {len(sim.data()) - len(match)} burns.")
    m, h = r["Methanol"]["o50"], r["Heptane"]["o50"]
    with st.expander("How this works"):
        st.markdown(f"""
* Half of NASA's methanol flames put themselves out below **{m[1]:.0f}% O₂** ({m[0]:.0f}–{m[2]:.0f}%), heptane below
  **{h[1]:.0f}%** ({h[0]:.0f}–{h[2]:.0f}%). In Moon-base air, **{r['moon_ext']} of {r['moon_n']}** NASA burns went out by themselves.
* Curve and star: a logistic model trained on {r['n']} FLEX droplet burns (cross-validated AUC {r['auc']:.2f}).
  Faded dots: NASA's measured share at 101 kPa with no suppressant.
* The burn follows the d² law: the droplet diameter squared shrinks at a steady rate learned from FLEX.
* The drawing is an illustration, not a measurement: its contours follow the inputs, and gravity stretches the
  flame upward (buoyancy) and makes the soot rise. All NASA data here is from microgravity.
* These are fuel droplets, not solid materials: they show how flames behave in microgravity, not whether a
  specific fabric is safe.""")


def gap_counts(df):
    g = _gaps(df)[MOON]
    solids = g[g.material_class.str.startswith("solid")]
    return dict(solids_near_moon=int(solids.tests_near.sum()))


@st.cache_data
def _gaps(df):
    return {name: gap(df, name, o2_tol=TOL_O2, kpa_tol=TOL_KPA) for name, _ in COLUMNS}


def _covered(points, tests):
    """Which grid points have at least one test within the 'near' box."""
    if len(tests) == 0:
        return np.zeros(len(points), bool)
    return ((np.abs(points[:, None, 0] - tests[None, :, 0]) <= TOL_O2)
            & (np.abs(points[:, None, 1] - tests[None, :, 1]) <= TOL_KPA)).any(axis=1)


@st.cache_data
def next_experiments(df):
    """Per pillar: the single new test condition inside the Moon-habitat range that covers the most of it."""
    o2 = np.linspace(*REGION["o2"], 17)
    kpa = np.linspace(*REGION["kpa"], 29)
    grid = np.array([(a, b) for a in o2 for b in kpa])
    out = {}
    for pillar, _, _ in PILLARS:
        t = df[df.pillar == pillar].dropna(subset=["oxygen_pct", "pressure_kpa"])[["oxygen_pct", "pressure_kpa"]].values
        now = _covered(grid, t)
        gain = [(now | _covered(grid, grid[i:i + 1])).mean() for i in range(len(grid))]
        best = int(np.argmax(gain))
        out[pillar] = dict(o2=grid[best, 0], kpa=grid[best, 1], before=now.mean(), after=gain[best])
    return out


# ------------------------------------------------------------------------------------------------ teasers
O2_AX, KPA_AX = (10.0, 36.0), (45.0, 320.0)


def _xy(o2, kpa, w, h, pad=4):
    o2 = np.clip(np.asarray(o2, float), *O2_AX)
    kpa = np.clip(np.asarray(kpa, float), *KPA_AX)
    x = pad + (o2 - O2_AX[0]) / (O2_AX[1] - O2_AX[0]) * (w - 2 * pad)
    y = h - pad - (np.log(kpa) - np.log(KPA_AX[0])) / (np.log(KPA_AX[1]) - np.log(KPA_AX[0])) * (h - 2 * pad)
    return x, y


@st.cache_data
def mini_map(df, pillar=None, scenario=None, w=160, h=84):
    """Tiny oxygen x pressure map: grey = all tests, blue = this pillar, red box = Moon range, red dot = scenario."""
    pts = df.dropna(subset=["oxygen_pct", "pressure_kpa"])
    pts = pts.assign(o=pts.oxygen_pct.round(0), k=pts.pressure_kpa.round(-1))
    parts = [f"<svg viewBox='0 0 {w} {h}' class='teaser' role='img' aria-label='Oxygen versus pressure of the tests'>",
             f"<rect x='0' y='0' width='{w}' height='{h}' rx='4' class='tbg'/>"]
    (bx0, by1), (bx1, by0) = [tuple(map(float, _xy(o, k, w, h))) for o, k in
                              ((REGION["o2"][0], REGION["kpa"][0]), (REGION["o2"][1], REGION["kpa"][1]))]
    parts.append(f"<rect x='{bx0:.1f}' y='{by0:.1f}' width='{bx1 - bx0:.1f}' height='{by1 - by0:.1f}' class='tbox'/>")
    for is_pillar, g in pts.groupby(pts.pillar.eq(pillar) if pillar else pd.Series(False, index=pts.index)):
        u = g.drop_duplicates(["o", "k"])
        xs, ys = _xy(u.o, u.k, w, h)
        cls = "tpt on" if is_pillar else "tpt"
        parts += [f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{2.2 if is_pillar else 1.6}' class='{cls}'/>" for x, y in zip(xs, ys)]
    if scenario:
        p = PRESETS[scenario]
        x, y = map(float, _xy(p["o2"], p["kpa"], w, h))
        off = "→" if p["o2"] > O2_AX[1] else ""
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' class='tsc'/>")
        if off:
            parts.append(f"<text x='{x - 14:.1f}' y='{y + 3:.1f}' class='tlab'>{p['o2']:g}%</text>")
    parts.append("</svg>")
    return "".join(parts)


def mini_icon(kind, w=160, h=70):
    if kind == "evidence":
        bars = [(250, "#2a78d6"), (154, "#eb6834"), (274, "#1baf7a")]
        x, out = 6, []
        for n, c in bars:
            bw = (w - 12) * n / 678
            out.append(f"<rect x='{x:.1f}' y='26' width='{bw - 2:.1f}' height='18' rx='3' fill='{c}'/>")
            x += bw
        body = "".join(out)
    elif kind == "table":
        body = "".join(f"<rect x='8' y='{10 + i * 11}' width='{w - 16}' height='7' rx='2' class='trow'/>" for i in range(5))
    else:  # models: a falling S-curve
        xs = np.linspace(8, w - 8, 40)
        ys = 12 + (h - 24) / (1 + np.exp(-(xs - w / 2) / 12))
        body = "<polyline fill='none' stroke='#eb6834' stroke-width='3' points='" + \
               " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys)) + "'/>"
    return (f"<svg viewBox='0 0 {w} {h}' class='teaser' role='img' aria-hidden='true'>"
            f"<rect x='0' y='0' width='{w}' height='{h}' rx='4' class='tbg'/>{body}</svg>")


def _cell(n, total, scenario, pillar):
    state, label = ("ok", "tested") if n >= 10 else ("warn", "a few") if n > 0 else ("bad", "never tested")
    url = f"?page=scenario&scenario={quote(scenario)}&pillar={pillar}"
    title = escape(f"{n} of {total} {pillar} tests within ±{TOL_O2:g} pp O₂ and ±{TOL_KPA:g} kPa of {scenario}",
                   quote=True)
    return (f"<td class='{state}'><a href='{url}' target='_self' title='{title}'>"
            f"<b>{n}</b><span>{label}</span></a></td>")


def overview(df):
    gaps = _gaps(df)
    st.markdown(CSS, unsafe_allow_html=True)

    # evidence table: atmospheres across, pillars down
    head = "".join(
        f"<th><a href='?page=scenario&scenario={quote(sc)}' target='_self' "
        f"title='{escape(PRESETS[sc]['source'], quote=True)}'>{short}</a>"
        f"<span>{PRESETS[sc]['o2']:g}% O₂ · {PRESETS[sc]['kpa'] / 6.895:.1f} psi</span>"
        f"<small>{PRESETS[sc]['gravity']} gravity</small></th>"
        for sc, short in COLUMNS)
    rows = []
    for pillar, name, question in PILLARS:
        exps = {"prevention": "BASS · BASS-II", "detection": "SAME · SAME-R", "suppression": "FLEX"}[pillar]
        n_p = int((df.pillar == pillar).sum())
        cells = "".join(_cell(int(gaps[sc].loc[gaps[sc].pillar == pillar, "tests_near"].sum()),
                              int(gaps[sc].loc[gaps[sc].pillar == pillar, "tests_total"].sum()), sc, pillar)
                        for sc, _ in COLUMNS)
        rows.append(f"<tr><th class='rowh'><a href='?page=scenario&pillar={pillar}' target='_self'>{name}</a>"
                    f"<span>{question}</span><small>{exps} · {n_p} tests</small></th>{cells}</tr>")
    matrix = (f"<div class='mapwrap'><table class='et'><thead><tr><th class='corner'></th>"
              f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")

    st.markdown(f"""
<div class='ov'>
  {section(1, "What NASA has tested", "Number of tests run near each atmosphere. Click a name or a number to read the evidence.")}
  {matrix}

</div>""", unsafe_allow_html=True)
    fire_lab_block()  # then: change the air, watch the flame


def branches_block(df):
    from guide import BRANCHES, url
    cards = "".join(
        f"<a class='branch' href='{url(b['params'])}' target='_self'>"
        f"{mini_map(df, pillar=b['teaser']) if b['teaser'] in ('prevention', 'detection', 'suppression') else mini_icon(b['teaser'])}"
        f"<b>{b['title']}</b><span>{b['sub']}</span></a>" for b in BRANCHES)
    st.markdown(f"""<div class='ov'>{section(0, "Start with a question", "Each card opens the page that answers it.")}
  <nav class='branches' aria-label='Main paths into the atlas'>{cards}</nav></div>""", unsafe_allow_html=True)
