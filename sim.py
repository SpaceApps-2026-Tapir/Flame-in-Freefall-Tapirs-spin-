"""Mini burn simulator backed by the FLEX data, and its comparison with a real NASA burn.

A burning droplet in microgravity follows the d² law: its diameter squared shrinks at a constant rate K,
    d(t)² = d0² - K·t
until the flame goes out (at the extinction diameter) or the fuel is gone. K and the extinction diameter
are learned from the 262 FLEX burns (random forests on oxygen, CO₂, helium, pressure, droplet size, fuel);
whether the flame goes out by itself comes from a logistic model.

compare(test_id) re-trains every model WITHOUT that burn, simulates it, and puts the simulation next to what
NASA measured, so the comparison is honest.
Written with AI assistance (Claude); see README "Use of AI".
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

RAW = Path(__file__).resolve().parent / "data" / "raw" / "flex_droplets.csv"
FEATS = ["o2_pct", "co2_pct", "he_pct", "pressure_kpa", "d0_mm", "heptane"]
FUEL_COLOR = {"Methanol": "#eb6834", "Heptane": "#2a78d6"}


@lru_cache(maxsize=1)
def data():
    d = pd.read_csv(RAW)
    d = d[d.pressure_valid & d.d0_mm.notna()].copy()
    d["heptane"] = (d.fuel == "Heptane").astype(int)
    d["self_ext"] = d.test_end.eq("extinction").astype(int)
    return d.reset_index(drop=True)


def _fit(d):
    k = d.dropna(subset=["burning_rate_mm2s"])
    r = d.dropna(subset=["extinction_ratio"])
    return dict(
        K=RandomForestRegressor(300, min_samples_leaf=3, random_state=0).fit(k[FEATS], k.burning_rate_mm2s),
        R=RandomForestRegressor(300, min_samples_leaf=3, random_state=0).fit(r[FEATS], r.extinction_ratio),
        P=LogisticRegression(C=10, max_iter=5000).fit(d[FEATS], d.self_ext))


@lru_cache(maxsize=1)
def models():
    return _fit(data())


def _row(fuel, o2, co2, he, kpa, d0):
    return pd.DataFrame([dict(o2_pct=o2, co2_pct=co2, he_pct=he, pressure_kpa=kpa, d0_mm=d0,
                              heptane=int(fuel == "Heptane"))])[FEATS]


def simulate(fuel, o2, co2=0.0, he=0.0, kpa=101.3, d0=3.0, m=None):
    """Simulated burn for these conditions. Returns rates, times and the d(t) curve."""
    m = m or models()
    x = _row(fuel, o2, co2, he, kpa, d0)
    K = float(m["K"].predict(x)[0])
    p_out = float(m["P"].predict_proba(x)[0, 1])
    ratio = float(np.clip(m["R"].predict(x)[0], 0, 0.99)) if p_out >= 0.5 else 0.0
    d_end = ratio * d0
    t_end = (d0 ** 2 - d_end ** 2) / K
    t = np.linspace(0, t_end, 60)
    d = np.sqrt(np.clip(d0 ** 2 - K * t, d_end ** 2, None))
    burned = 1 - (d_end / d0) ** 3
    return dict(fuel=fuel, o2=o2, co2=co2, he=he, kpa=kpa, d0=d0, K=K, p_self_ext=p_out, d_end=d_end,
                t_end=t_end, t=t, d=d, fuel_burned=burned)


def p_curve(fuel, co2, he, kpa, d0, o2_grid, m=None):
    """Chance the flame goes out by itself, across oxygen, with everything else fixed."""
    m = m or models()
    x = pd.concat([_row(fuel, o, co2, he, kpa, d0) for o in o2_grid], ignore_index=True)
    return m["P"].predict_proba(x)[:, 1]


def nearby_burns(o2, kpa, n=1, fuel=None):
    d = data() if fuel is None else data()[data().fuel == fuel]
    dist = np.hypot((d.o2_pct - o2) / 3, (d.pressure_kpa - kpa) / 10)
    return d.assign(_dist=dist).sort_values("_dist").head(n)


def compare(test_id):
    """Simulate a real burn with models that never saw it; return (simulation, measured row)."""
    d = data()
    row = d[d.test_id == test_id].iloc[0]
    m = _fit(d[d.test_id != test_id])
    sim = simulate(row.fuel, row.o2_pct, row.co2_pct, row.he_pct, row.pressure_kpa, row.d0_mm, m=m)
    return sim, row


# ------------------------------------------------------------------------------------------------ figures
def animation(sim, title=None, height=360):
    """Droplet shrinking frame by frame with an (illustrative) flame shell around it."""
    color = FUEL_COLOR[sim["fuel"]]
    lim = sim["d0"] * 3.2
    th = np.linspace(0, 2 * np.pi, 80)

    def circle(r):
        return r * np.cos(th), r * np.sin(th)

    def traces(i):
        r = sim["d"][i] / 2
        lit = i < len(sim["t"]) - 1 or sim["d_end"] == 0
        fx, fy = circle(r * 2.6)
        dx, dy = circle(r)
        return [go.Scatter(x=fx, y=fy, fill="toself", mode="lines", line=dict(color="rgba(80,140,255,.7)", width=3),
                           fillcolor="rgba(80,140,255,.18)" if lit else "rgba(0,0,0,0)", hoverinfo="skip",
                           opacity=1 if lit else 0),
                go.Scatter(x=dx, y=dy, fill="toself", mode="lines", line=dict(color=color, width=1),
                           fillcolor=color, hoverinfo="skip")]

    idx = list(range(0, len(sim["t"]), 3)) + [len(sim["t"]) - 1]
    frames = [go.Frame(data=traces(i), name=f"{sim['t'][i]:.1f}",
                       layout=go.Layout(annotations=[dict(x=0, y=-lim * 0.9, showarrow=False,
                                                          text=f"t = {sim['t'][i]:.1f} s · d = {sim['d'][i]:.2f} mm")]))
              for i in idx]
    fig = go.Figure(data=traces(0), frames=frames)
    fig.update_layout(
        height=height, showlegend=False, margin=dict(l=0, r=0, t=30 if title else 5, b=0),
        title=dict(text=title, font=dict(size=14)) if title else None,
        xaxis=dict(visible=False, range=[-lim, lim]), yaxis=dict(visible=False, range=[-lim, lim], scaleanchor="x"),
        annotations=[dict(x=0, y=-lim * 0.9, showarrow=False, text=f"t = 0.0 s · d = {sim['d0']:.2f} mm")],
        updatemenus=[dict(type="buttons", x=0, y=1.08, xanchor="left", showactive=False, buttons=[
            dict(label="Burn", method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=False, mode="immediate")])])])
    return fig


def d2_chart(sim, measured=None, height=300):
    """d² vs time: the simulated line and, if given, NASA's measured start and end of the burn."""
    color = FUEL_COLOR[sim["fuel"]]
    fig = go.Figure(go.Scatter(x=sim["t"], y=sim["d"] ** 2, mode="lines", name="Simulated",
                               line=dict(color=color, width=3),
                               hovertemplate="%{x:.1f} s · d² = %{y:.2f} mm²<extra>simulated</extra>"))
    if measured is not None:
        d_end = measured.d_ext_mm if pd.notna(measured.d_ext_mm) else 0.0
        fig.add_trace(go.Scatter(x=[0, measured.burn_time_s], y=[measured.d0_mm ** 2, d_end ** 2],
                                 mode="markers+lines", name="NASA measured",
                                 line=dict(color="#555", width=1.5, dash="dot"),
                                 marker=dict(size=11, symbol="diamond", color="#111", line=dict(color="white", width=1.5)),
                                 hovertemplate="%{x:.1f} s · d² = %{y:.2f} mm²<extra>measured</extra>"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Time (s)",
                      yaxis_title="Droplet diameter² (mm²)", legend=dict(orientation="h", y=1.12, x=0))
    return fig


# ------------------------------------------------------------------------------------------------ neighbours and certainty
@lru_cache(maxsize=1)
def _scaler():
    d = data()
    return d[FEATS].mean(), d[FEATS].std().replace(0, 1)


def similar(fuel, o2, co2=0.0, he=0.0, kpa=101.3, d0=3.0, k=15):
    """The k NASA burns closest to these conditions (standardised distance over the model inputs)."""
    mu, sd = _scaler()
    d = data()
    x = (_row(fuel, o2, co2, he, kpa, d0).iloc[0] - mu) / sd
    dist = np.sqrt((((d[FEATS] - mu) / sd - x) ** 2).sum(axis=1))
    return d.assign(_dist=dist).sort_values("_dist").head(k)


def matching(fuel, o2, co2=0.0, kpa=101.3, o2_tol=3.0, kpa_tol=10.0, co2_tol=10.0):
    """NASA burns run in (almost) this air with this fuel: the evidence behind the simulation."""
    d = data()
    return d[(d.fuel == fuel) & ((d.o2_pct - o2).abs() <= o2_tol) & ((d.pressure_kpa - kpa).abs() <= kpa_tol)
             & ((d.co2_pct - co2).abs() <= co2_tol) & (d.he_pct <= 1)]


@lru_cache(maxsize=1)
def held_out_probs():
    """Self-extinction probability for every NASA burn from a model that never saw it (5-fold)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    d = data()
    return cross_val_predict(LogisticRegression(C=10, max_iter=5000), d[FEATS], d.self_ext,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]


def certainty(sim, k=15):
    """How sure the model is for this flame, and how often it was right on the k most similar real burns."""
    p = sim["p_self_ext"]
    near = similar(sim["fuel"], sim["o2"], sim["co2"], sim["he"], sim["kpa"], sim["d0"], k)
    probs = held_out_probs()[near.index]
    right = ((probs >= 0.5).astype(int) == near.self_ext.values).mean()
    return dict(says="goes out by itself" if p >= 0.5 else "keeps burning", certainty=max(p, 1 - p),
                local_accuracy=float(right), k=k, mean_distance=float(near._dist.mean()))


@lru_cache(maxsize=1)
def burn_map():
    """3D look-alike map (t-SNE) of every NASA burn, with axes named after what they follow."""
    from sklearn.manifold import TSNE
    from lab_engine import axis_names
    mu, sd = _scaler()
    d = data()
    emb = TSNE(n_components=3, perplexity=30, random_state=0, init="pca").fit_transform(((d[FEATS] - mu) / sd).values)
    names = axis_names(emb, d, {"o2_pct": "Oxygen (%)", "co2_pct": "CO₂ (%)", "he_pct": "Helium (%)",
                                "pressure_kpa": "Pressure (kPa)", "d0_mm": "Droplet size (mm)"}, ["fuel"])
    return emb, names


def place_on_map(fuel, o2, co2=0.0, he=0.0, kpa=101.3, d0=3.0, k=5):
    """Where a new flame sits on the burn map: the average position of its k closest real burns."""
    emb, _ = burn_map()
    near = similar(fuel, o2, co2, he, kpa, d0, k)
    w = 1 / (near._dist.values + 0.05)
    return (emb[near.index] * w[:, None]).sum(axis=0) / w.sum()


# ------------------------------------------------------------------------------------------------ animated canvas
GRAVITY = {"Microgravity (ISS)": 0.0, "Moon (0.16 g)": 0.16, "Mars (0.38 g)": 0.38, "Earth (1 g)": 1.0}


def animation_html(sim, height=330, g=0.0, theme="light"):
    """The burn drawn like a figure in a paper (in the site's font): thin contours, stippled soot, scale bar, labels.

    Burn time and whether the flame goes out come from the models (microgravity data). The drawing is a free
    illustration tied to the inputs: fuel, oxygen, CO₂, helium, pressure and gravity. With gravity, hot gas
    rises (buoyancy): the flame contours stretch upward and the soot drifts up instead of spreading out.
    """
    import json
    params = json.dumps(dict(fuel=sim["fuel"], o2=sim["o2"], co2=sim["co2"], he=sim["he"], kpa=sim["kpa"],
                             d0=sim["d0"], K=sim["K"], t_end=sim["t_end"], d_end=sim["d_end"],
                             out=bool(sim["p_self_ext"] >= 0.5), g=float(g), dark=theme == "dark"))
    serif = "'Source Sans Pro','Source Sans 3',system-ui,-apple-system,'Segoe UI',sans-serif"  # the site's font
    return f"""<meta charset="utf-8">
<div id="w" style="position:relative;width:100%;height:{height}px;overflow:hidden;font-family:{serif}">
  <canvas id="c" style="width:100%;height:100%;display:block"></canvas>
  <button id="rp" style="position:absolute;right:10px;top:8px;background:transparent;border:none;padding:0;
       font:italic 12px {serif};text-decoration:underline;cursor:pointer">replay</button>
</div>
<script>
(() => {{
  const P = {params}, SERIF = "{serif.replace('"', "")}";
  // follow the page's theme live (also when it is switched from the app menu): read the app background
  const THEMES = {{dark: {{bg: "#202735", ink: "#e8ecf2", soft: "#9aa3b2", faint: "#3a4558"}},
                  light: {{bg: "#ffffff", ink: "#1b1b1b", soft: "#6b6b6b", faint: "#e4e4e4"}}}};
  let C = THEMES[P.dark ? "dark" : "light"];
  function readTheme() {{
    try {{
      const app = window.parent.document.querySelector('[data-testid="stApp"]') || window.parent.document.body;
      const m = getComputedStyle(app).backgroundColor.match(/[\d.]+/g).map(Number);
      const dark = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) < 128;
      C = Object.assign({{}}, THEMES[dark ? "dark" : "light"], {{bg: `rgb(${{m[0]}},${{m[1]}},${{m[2]}})`}});
    }} catch (e) {{
      C = THEMES[(window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light"];
    }}
    const darkNow = C.ink === THEMES.dark.ink;
    ACC = P.fuel === "Heptane" ? (darkNow ? "#ff9a5c" : "#c4551f") : (darkNow ? "#8ab8f5" : "#2a5fb0");
    try {{ rp.style.color = C.soft; }} catch (e) {{}}   // the button may not exist yet on the first call
  }}
  readTheme(); setInterval(readTheme, 700);
  let ACC = "#2a5fb0";                                                  // one accent colour per fuel, per theme
  const cv = document.getElementById("c"), x = cv.getContext("2d"), rp = document.getElementById("rp");
  rp.style.color = C.soft;
  let W, H;
  function size() {{ const d = window.devicePixelRatio || 1; W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * d; cv.height = H * d; x.setTransform(d, 0, 0, d, 0, 0); }}
  size(); addEventListener("resize", size);

  const hep = P.fuel === "Heptane";
  const o2 = Math.max(.4, Math.min(1.6, (P.o2 - 10) / 13));
  const damp = Math.max(.35, 1 - P.co2 / 90 - P.he / 140);
  const lowP = P.kpa < 85 ? 1.12 : 1;
  const stretch = 1.6 * Math.sqrt(P.g);
  const soot = (hep ? 1 : .2) * o2 * damp;
  const speed = Math.max(1, P.t_end / 8);
  let t0 = performance.now(), last = t0, dots = [], outT = null;
  rp.onclick = () => {{ t0 = performance.now(); dots = []; outT = null; }};
  const diam = t => Math.sqrt(Math.max(P.d_end * P.d_end, P.d0 * P.d0 - P.K * t));

  function contour(cx, cy, r, k, T, ph) {{                            // a smooth closed iso-line
    x.beginPath();
    for (let i = 0; i <= 96; i++) {{
      const a = i / 96 * Math.PI * 2, up = Math.max(0, -Math.sin(a));
      let rr = r * (1 + .012 * Math.sin(a * 4 + T * 1.6 + ph));
      rr *= 1 + k * Math.pow(up, 2.6) - .1 * Math.min(1, k) * Math.max(0, Math.sin(a));
      const px = cx + Math.cos(a) * rr * (1 - .22 * Math.min(1, k) * up), py = cy + Math.sin(a) * rr;
      i ? x.lineTo(px, py) : x.moveTo(px, py);
    }}
    x.closePath();
  }}
  function label(text, x1, y1, x2, y2, align) {{                       // annotation with a leader line
    x.strokeStyle = C.soft; x.lineWidth = .8; x.beginPath(); x.moveTo(x1, y1); x.lineTo(x2, y2); x.stroke();
    x.fillStyle = C.ink; x.font = `italic 12px ${{SERIF}}`; x.textAlign = align; x.textBaseline = "middle";
    x.fillText(text, x2 + (align === "left" ? 4 : -4), y2);
  }}

  function frame(now) {{
    const dt = Math.min(.05, (now - last) / 1000); last = now; const T = now / 1000;
    if (outT !== null && now - outT > 6000) {{ t0 = now; dots = []; outT = null; }}
    const tr = (now - t0) / 1000 * speed, t = Math.min(tr, P.t_end), burning = tr < P.t_end;
    if (!burning && outT === null) outT = now;
    const fade = outT === null ? 1 : Math.max(0, 1 - (now - outT) / 700);
    const d = diam(t), s = Math.min(W, H) / (P.d0 * 10), cx = W * .46, cy = H / 2 + 14 + 16 * Math.min(1, stretch);
    const rd = d * s / 2, prog = t / P.t_end;
    let rf = Math.max(rd * 2.4, rd * (4.6 - (P.o2 - 21) * .06) * (1 + .35 * prog) * lowP);
    rf = Math.min(rf, (cy - 40) / (1 + stretch * 1.1), H * .34) * fade;

    x.fillStyle = C.bg; x.fillRect(0, 0, W, H);

    // stippled soot / vapour: tiny dots drifting out (or up, with gravity)
    const rate = burning ? .4 + 3 * soot : (P.d_end > 0 && now - outT < 3000 ? 1.2 : 0);
    for (let i = 0; i < rate * dt * 30; i++) {{
      const a = P.g > 0 ? -Math.PI / 2 + (Math.random() - .5) * 1.6 : Math.random() * 7, r0 = burning ? rf * .9 : rd * 1.4;
      const v = 8 + Math.random() * 12;
      dots.push({{ x: cx + Math.cos(a) * r0, y: cy + Math.sin(a) * r0, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0,
                  max: 3 + Math.random() * 3, vap: !burning }});
    }}
    for (const p of dots) {{ p.life += dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vy -= 20 * P.g * dt; p.vx *= .99; p.vy *= .99;
      x.globalAlpha = Math.max(0, 1 - p.life / p.max) * (p.vap ? .5 : .8); x.fillStyle = p.vap ? C.soft : C.ink;
      x.fillRect(p.x, p.y, 1.3, 1.3); }}
    x.globalAlpha = 1; dots = dots.filter(p => p.life < p.max).slice(-900);

    // flame: outer reaction front in the accent colour, inner iso-lines dashed, a faint wash inside
    if (rf > 1) {{
      const k = stretch * (1 + .06 * Math.sin(T * 5));
      contour(cx, cy, rf * (0.8 + .2 * damp), k, T, 0);
      x.globalAlpha = .07 + .08 * Math.min(1, o2 * damp); x.fillStyle = ACC; x.fill(); x.globalAlpha = 1;
      x.strokeStyle = ACC; x.lineWidth = 1.6; x.stroke();
      x.setLineDash([3, 3]); x.lineWidth = .8; x.strokeStyle = C.soft;
      for (const f of [.78, .58]) {{ contour(cx, cy, rf * f * (0.8 + .2 * damp), k * f, T, f * 3); x.stroke(); }}
      x.setLineDash([]);
      if (hep && soot > .3) {{                                        // soot shell: a ring of fine stipple
        x.fillStyle = C.ink; x.globalAlpha = .55 * Math.min(1, soot);
        for (let i = 0; i < 140; i++) {{ const a = (i * 2.399) % (Math.PI * 2), up = Math.max(0, -Math.sin(a));
          const rr = rf * (0.66 + .05 * Math.sin(i * 7.1 + T)) * (1 + k * .66 * Math.pow(up, 2.6)) * (0.8 + .2 * damp);
          x.fillRect(cx + Math.cos(a) * rr * (1 - .2 * Math.min(1, k) * up), cy + Math.sin(a) * rr, 1.2, 1.2); }}
        x.globalAlpha = 1;
      }}
    }}

    // droplet: outlined circle with hatching
    if (rd > .6) {{
      x.save(); x.beginPath(); x.arc(cx, cy, rd, 0, 7); x.fillStyle = C.bg; x.fill(); x.clip();
      x.strokeStyle = C.soft; x.lineWidth = .7;
      for (let h = -rd * 2; h < rd * 2; h += 3) {{ x.beginPath(); x.moveTo(cx + h - rd, cy + rd); x.lineTo(cx + h + rd, cy - rd); x.stroke(); }}
      x.restore(); x.strokeStyle = C.ink; x.lineWidth = 1.2; x.beginPath(); x.arc(cx, cy, rd, 0, 7); x.stroke();
    }}

    // the ending, spelled out on the figure
    if (!burning) {{
      x.textAlign = "center"; x.textBaseline = "middle"; x.font = `600 14px ${{SERIF}}`;
      if (P.out) {{
        const left = Math.round(Math.pow(P.d_end / P.d0, 3) * 100);
        x.fillStyle = "#1b7a55"; x.fillText("the flame went out by itself", cx, cy - Math.max(rd, 10) - 46);
        x.font = `italic 13px ${{SERIF}}`; x.fillStyle = C.ink;
        x.fillText(`${{left}}% of the fuel is left unburned`, cx, cy - Math.max(rd, 10) - 28);
      }} else {{
        x.fillStyle = "#a8332b"; x.fillText("it kept burning until the fuel ran out", cx, cy - 14);
        x.font = `italic 13px ${{SERIF}}`; x.fillStyle = C.ink; x.fillText("no fuel left: 100% burned", cx, cy + 6);
      }}
    }}

    // labels with leader lines
    const lx = cx + rf * .72 + 30;
    if (burning && rf > 6) label("flame front", cx + rf * .70, cy - rf * .55, Math.min(W - 92, lx + 8), cy - rf * .55 - 18, "left");
    if (hep && soot > .3 && rf > 6) label("soot shell", cx + rf * .5, cy + rf * .35, Math.min(W - 92, lx), cy + rf * .5 + 12, "left");
    if (rd > .6) label(burning ? "droplet" : "unburned fuel", cx - rd * .7, cy + rd * .7, cx - rd - 40, cy + rd + 30, "right");

    // scale bar (1 mm) and gravity vector
    const mm = s >= 30 ? 1 : (s >= 15 ? 2 : 5), bar = s * mm, bx = W - 22 - bar, by = H - 22;
    x.strokeStyle = C.ink; x.lineWidth = 1.2; x.beginPath(); x.moveTo(bx, by); x.lineTo(bx + bar, by);
    x.moveTo(bx, by - 4); x.lineTo(bx, by + 4); x.moveTo(bx + bar, by - 4); x.lineTo(bx + bar, by + 4); x.stroke();
    x.fillStyle = C.ink; x.font = `12px ${{SERIF}}`; x.textAlign = "center"; x.fillText(`${{mm}} mm`, bx + bar / 2, by - 9);
    if (P.g > 0) {{
      const gx = W - 22, gy0 = 40, gy1 = 40 + 24 + 30 * Math.min(1, P.g);
      x.beginPath(); x.moveTo(gx, gy0); x.lineTo(gx, gy1); x.moveTo(gx - 4, gy1 - 6); x.lineTo(gx, gy1); x.lineTo(gx + 4, gy1 - 6); x.stroke();
      x.font = `italic 13px ${{SERIF}}`; x.textAlign = "right"; x.fillText(`g = ${{P.g}} g₀`, gx - 7, gy0 + 8);
    }}

    // caption-style read-out
    x.textAlign = "left"; x.textBaseline = "alphabetic"; x.fillStyle = C.ink; x.font = `13px ${{SERIF}}`;
    const gas = `${{P.fuel.toLowerCase()}}, ${{P.o2.toFixed(1)}}% O₂, ${{P.kpa.toFixed(0)}} kPa` + (P.co2 > 0 ? `, ${{P.co2.toFixed(0)}}% CO₂` : "");
    x.fillText(gas, 12, 20);
    x.font = `italic 13px ${{SERIF}}`; x.fillStyle = C.soft;
    x.fillText(`t = ${{t.toFixed(1)}} s,  d = ${{d.toFixed(2)}} mm` + (speed > 1.05 ? `   (time ×${{speed.toFixed(1)}})` : ""), 12, 38);
    x.font = `13px ${{SERIF}}`; x.fillStyle = burning ? C.ink : (P.out ? "#1b7a55" : "#a8332b");
    x.fillText(burning ? (P.out ? "burning · will go out by itself" : "burning · will keep going")
                       : (P.out ? "ending: self-extinguished" : "ending: fuel used up"), 12, H - 14);
    requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}})();
</script>"""
