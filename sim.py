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


def nearby_burns(o2, kpa, n=1):
    d = data()
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
            dict(label="▶ Burn", method="animate",
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
