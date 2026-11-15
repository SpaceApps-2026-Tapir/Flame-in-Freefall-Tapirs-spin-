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
  .ov.tight {margin-top: .9rem;}
  .ov .card .p, .ov .kicker {font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--atlas-muted);}

  /* map: scenarios x pillars */
  .ov .mapwrap {overflow-x: auto;}
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
    return (f"<h3><span class='step'>{n}</span>{title}</h3>" + (f"<div class='lead'>{lead}</div>" if lead else ""))


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


def finding_block():
    if not FLEX_RAW.exists():
        return
    r = self_extinction()
    lo_o2, hi_o2 = REGION["o2"]
    fig = go.Figure()
    fig.add_vrect(x0=lo_o2, x1=hi_o2, fillcolor="rgba(127,127,127,.15)", line_width=0)
    fig.add_annotation(x=(lo_o2 + hi_o2) / 2, y=100, text="Moon base", showarrow=False, yref="y", yanchor="bottom")
    for fuel, color in FUELS:
        c = r[fuel]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=np.r_[r["o2"], r["o2"][::-1]], y=np.r_[c["hi"], c["lo"][::-1]] * 100,
                                 fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},.15)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=r["o2"], y=c["fit"] * 100, mode="lines", name=fuel,
                                 line=dict(color=color, width=2.5),
                                 hovertemplate=f"{fuel}: %{{y:.0f}}% at %{{x:.1f}}% O₂<extra>model</extra>"))
        ob = r["obs"][r["obs"].fuel == fuel]
        fig.add_trace(go.Scatter(x=ob.o2, y=ob.share * 100, mode="markers", showlegend=False,
                                 marker=dict(color=color, size=np.clip(ob.n * 1.2, 8, 22),
                                             line=dict(color="white", width=1.5)),
                                 customdata=ob.n, hovertemplate=f"{fuel}: %{{y:.0f}}% of %{{customdata}} burns "
                                                                "near %{x:.0f}% O₂<extra>measured</extra>"))
    tidy(fig, 340)
    fig.update_layout(
                      xaxis=dict(title="Oxygen (%)", range=[10, 36]),
                      yaxis=dict(title="Flames that went out on their own (%)", range=[0, 108]),
                      legend=dict(orientation="h", y=-0.25, x=0))

    m, h = r["Methanol"]["o50"], r["Heptane"]["o50"]
    st.markdown(f"""
<div class='ov'>
  {section(3, 'Why the Moon-base air matters', 'In microgravity a burning droplet often puts itself out. Above about 20–23% oxygen that safety margin is gone.')}
  <div class='cards'>
    <div class='card'><div class='p'>Half the flames self-extinguish below</div>
      <div class='cond'>{m[1]:.0f}% O₂</div><div class='what'>Methanol (90% range {m[0]:.0f}–{m[2]:.0f}%)</div></div>
    <div class='card'><div class='p'>Half the flames self-extinguish below</div>
      <div class='cond'>{h[1]:.0f}% O₂</div><div class='what'>Heptane (90% range {h[0]:.0f}–{h[2]:.0f}%)</div></div>
    <div class='card'><div class='p'>Burns in Moon-base air</div>
      <div class='cond'>{r['moon_ext']} of {r['moon_n']}</div><div class='what'>went out on their own
      (≥{lo_o2:g}% O₂, ~70 kPa)</div></div>
  </div>
</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="finding_fig")
    st.markdown(f"<div class='ov tight'><div class='foot'>Logistic regression on {r['n']} FLEX droplet burns "
                f"(oxygen, CO₂, helium, pressure, fuel, droplet size); cross-validated AUC {r['auc']:.2f}. "
                "Lines: model at 101 kPa with no suppressant; bands: 90% bootstrap range; dots: measured share "
                "(size = number of burns).</div></div>", unsafe_allow_html=True)


DEMO_BURNS = {  # real FLEX burns worth comparing; first one is the default (Moon-base air)
    "191F001": "Moon-base air · methanol, 30% O₂, 72 kPa",
    "195F001": "Moon-base air · heptane, 30% O₂, 72 kPa",
    "192F001": "Exploration air · methanol, 34% O₂, 72 kPa",
}


@st.cache_data(show_spinner=False)
def _compare(test_id):
    import sim
    s, row = sim.compare(test_id)
    return s, row.to_dict()


@st.cache_data(show_spinner=False)
def _simulate(fuel, o2, co2, he, kpa, d0):
    import sim
    return sim.simulate(fuel, o2, co2, he, kpa, d0)


def simulator_block():
    import sim
    if not sim.RAW.exists():
        return
    st.markdown("<div class='ov'>" + section(1, "Light a fire in Moon-base air",
                "A droplet of fuel burning in microgravity at 30% oxygen: a mini simulation built from 262 NASA "
                "burns, next to what NASA measured in the same air. Press ▶.") + "</div>", unsafe_allow_html=True)
    d = sim.data()
    options = list(DEMO_BURNS) + sorted(set(d.test_id) - set(DEMO_BURNS)) + ["custom"]
    pick = st.selectbox("Burn", options, format_func=lambda t: "Design your own conditions" if t == "custom"
                        else DEMO_BURNS.get(t, f"NASA burn {t}"), label_visibility="collapsed", key="sim_pick")
    measured = None
    if pick == "custom":
        c = st.columns(5)
        fuel = c[0].radio("Fuel", ["Methanol", "Heptane"], key="sim_fuel")
        o2 = c[1].slider("Oxygen (%)", 12.0, 34.0, 30.0, 0.5, key="sim_o2")
        kpa = c[2].slider("Pressure (kPa)", 50.0, 110.0, 70.3, 0.5, key="sim_kpa")
        co2 = c[3].slider("CO₂ added (%)", 0.0, 60.0, 0.0, 1.0, key="sim_co2")
        d0 = c[4].slider("Droplet (mm)", 1.5, 4.5, 3.0, 0.1, key="sim_d0")
        s = _simulate(fuel, o2, co2, 0.0, kpa, d0)
        near = sim.nearby_burns(o2, kpa).iloc[0]
        st.caption(f"Closest real burn: {near.test_id} ({near.fuel}, {near.o2_pct:g}% O₂, {near.pressure_kpa:.0f} kPa). "
                   "Far from every real burn, the simulation is a guess.")
    else:
        s, row = _compare(pick)
        measured = pd.Series(row)

    with st.container(border=True):
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            st.plotly_chart(tidy(sim.animation(s), 330), theme="streamlit", width="stretch", key="sim_anim")
        with right:
            st.plotly_chart(tidy(sim.d2_chart(s, measured), 330), theme="streamlit", width="stretch", key="sim_d2")

    out_sim = "goes out by itself" if s["p_self_ext"] >= 0.5 else "keeps burning"
    cards = [("Burn time", f"{s['t_end']:.1f} s",
              f"NASA: {measured.burn_time_s:.1f} s" if measured is not None else "simulated"),
             ("The flame…", out_sim, "NASA: " + {"extinction": "went out by itself", "disruption": "droplet burst",
                                                  "completion": "burned to the end"}[measured.test_end]
              if measured is not None else f"{s['p_self_ext']:.0%} chance it goes out by itself"),
             ("Fuel burned", f"{s['fuel_burned']:.0%}",
              (f"NASA: {1 - (measured.d_ext_mm / measured.d0_mm) ** 3:.0%}" if measured is not None
               and pd.notna(measured.d_ext_mm) else "of the droplet volume"))]
    st.markdown("<div class='ov tight'><div class='cards'>" + "".join(
        f"<div class='card'><div class='p'>{a}</div><div class='cond'>{b}</div><div class='what'>{c}</div></div>"
        for a, b, c in cards) + "</div><div class='foot'>d² law (diameter² shrinks at a steady rate) with the rate, "
        "the extinction size and the chance of self-extinction learned from FLEX. For a NASA burn, the models are "
        "re-trained without that burn first. The blue shell around the droplet is illustrative.</div></div>",
        unsafe_allow_html=True)


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
    return (f"<a class='cell {state}' href='{url}' target='_self' title='{title}'>"
            f"<span class='n'>{n}</span><span class='s'>{label}</span></a>")


def overview(df):
    gaps = _gaps(df)
    st.markdown(CSS, unsafe_allow_html=True)
    simulator_block()  # the hook: a fire in Moon-base air

    # map: scenarios x pillars (CSS grid, every row the same height)
    head = "".join(
        f"<a class='hcard' href='?page=scenario&scenario={quote(sc)}' target='_self' "
        f"title='{escape(PRESETS[sc]['source'], quote=True)}'>"
        f"<b>{short}</b><span>{PRESETS[sc]['o2']:g}% O₂ · {PRESETS[sc]['kpa'] / 6.895:.1f} psi</span>"
        f"{mini_map(df, scenario=sc)}<em>{PRESETS[sc]['gravity']} gravity</em></a>"
        for sc, short in COLUMNS)
    rows = []
    for pillar, name, question in PILLARS:
        exps = {"prevention": "BASS · BASS-II", "detection": "SAME · SAME-R", "suppression": "FLEX"}[pillar]
        n_p = int((df.pillar == pillar).sum())
        rows.append(f"<a class='hcard' href='?page=scenario&pillar={pillar}' target='_self'>"
                    f"<b>{name}</b><span>{question}</span>{mini_map(df, pillar=pillar, h=60)}"
                    f"<em>{exps} · {n_p} tests</em></a>")
        rows += [_cell(int(gaps[sc].loc[gaps[sc].pillar == pillar, "tests_near"].sum()),
                       int(gaps[sc].loc[gaps[sc].pillar == pillar, "tests_total"].sum()), sc, pillar)
                 for sc, _ in COLUMNS]
    matrix = (f"<div class='mapwrap'><div class='mapgrid'><div class='kicker' style='align-self:end'>"
              f"Pillar ↓ · Atmosphere →</div>{head}{''.join(rows)}</div></div>")

    # 3 ----------------------------------------------------------------- test next
    nxt = next_experiments(df)
    cards = "".join(
        f"<div class='card'><div class='p'>{name}</div>"
        f"<div class='cond'>{nxt[p]['o2']:.1f}% O₂ · {nxt[p]['kpa']:.0f} kPa</div>"
        f"<div class='what'>{WHAT_TO_BURN[p]}</div>"
        f"<div class='cov'>Moon-habitat range covered<br><b>{nxt[p]['before']:.0%} → {nxt[p]['after']:.0%}</b>"
        f" with this one test</div></div>"
        for p, name, _ in PILLARS)

    lo_o2, hi_o2 = REGION["o2"]
    lo_k, hi_k = REGION["kpa"]
    st.markdown(f"""
<div class='ov'>
  {section(2, "What NASA has tested", "Each number is how many tests ran near that atmosphere. Click any box to go "
           "deeper. Mini maps: oxygen → vs pressure ↑, red box = Moon-habitat range.")}
  {matrix}
  <div class='foot'>Every test ran in microgravity on the ISS: Moon and Mars gravity have never been tested.</div>
</div>""", unsafe_allow_html=True)
    finding_block()
    st.markdown(f"""
<div class='ov'>
  {section(4, "What to test next", f"The one new test per pillar that would cover the most of the Moon-habitat range "
           f"({lo_o2:g}–{hi_o2:g}% O₂, {lo_k:g}–{hi_k:g} kPa).")}
  <div class='cards'>{cards}</div>
  <div class='foot'>"Covered" = a test within ±{TOL_O2:g} pp O₂ and ±{TOL_KPA:g} kPa.</div>
</div>""", unsafe_allow_html=True)

    branches_block(df)


def branches_block(df):
    from guide import BRANCHES, url
    cards = "".join(
        f"<a class='branch' href='{url(b['params'])}' target='_self'>"
        f"{mini_map(df, pillar=b['teaser']) if b['teaser'] in ('prevention', 'detection', 'suppression') else mini_icon(b['teaser'])}"
        f"<b>{b['title']}</b><span>{b['sub']}</span></a>" for b in BRANCHES)
    st.markdown(f"""<div class='ov'>{section(5, "Where to next?", "Pick a question. Charts open with the assistant "
                                                "ready to keep guiding you.")}
  <nav class='branches' aria-label='Main paths into the atlas'>{cards}</nav></div>""", unsafe_allow_html=True)
