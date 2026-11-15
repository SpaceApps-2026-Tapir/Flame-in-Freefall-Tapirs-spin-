"""Tool-calling agent for "Explore the data visually".

The model never sees raw numbers it did not get from a tool: it changes the design matrix and the view,
runs analyses from lab_engine.ANALYSES, and explains what came back. Uses the same Groq
OpenAI-compatible endpoint as rag.py (GROQ_API_KEY; model from GROQ_AGENT_MODEL or GROQ_MODEL).
Written with AI assistance (Claude); see README "Use of AI".
"""
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

import lab_engine as L

URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_ROUNDS = 6

SYSTEM = """You drive an interactive 3D chart of NASA microgravity fire-safety tests. You can:
- set_design_matrix: choose dataset, which variables enter the analysis, their weight and transform, and filters.
- set_view: pick the chart that best answers the question (see charts_available in get_state), the colour,
  the axes, and rotate/tilt/zoom the camera of 3D charts.
- run_analysis: run one of the available analyses on the current design matrix.
- get_state: see the current dataset, variables, filters, view and available analyses.
- suggest_next: offer the reader one next stop (another chart, the evidence page for a scenario/pillar, the
  test explorer, or Ask the evidence for answers from NASA reports) when it would answer their question better. Use it at most once per turn.
Rules:
- Act with tools first, then answer in 2-4 short sentences in plain language: what the chart now shows and
  what stands out. Talk about tests, oxygen, pressure, flames and smoke, not about statistics: do not name
  methods (t-SNE, PCA, random forest...) unless the user asks how it was done.
- Choose the chart yourself: 'space' for where tests sit vs a mission, 'flows' for conditions -> outcome,
  'evidence' for what exists, 'mixture' for FLEX gas mixtures, 'paths' to compare many quantities at once,
  'similarity' for which tests resemble each other.
- Only state numbers that a tool returned. Never invent tests, values or citations.
- If a request needs data that is not available (e.g. lunar-gravity tests), say so plainly.
- Variable names must match get_state exactly. Call get_state first if unsure.
- Answer in the user's language."""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_state", "description": "Current dataset, design matrix, filters, view and available analyses.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_design_matrix",
        "description": "Change what goes into the analysis. Switching dataset resets the matrix to its defaults.",
        "parameters": {"type": "object", "properties": {
            "dataset": {"type": "string", "enum": list(L.DATASETS)},
            "only": {"type": "array", "items": {"type": "string"},
                     "description": "Use exactly these variables and switch all others off."},
            "variables": {"type": "array", "items": {"type": "object", "properties": {
                "variable": {"type": "string"}, "use": {"type": "boolean"},
                "weight": {"type": "number", "description": "0-10, default 1"},
                "transform": {"type": "string", "enum": L.TRANSFORMS}}, "required": ["variable"]}},
            "filters": {"type": "object", "description": "column -> list of allowed values, or [min, max] for numbers. "
                                                         "Empty object clears filters."}}}}},
    {"type": "function", "function": {
        "name": "set_view", "description": "Change how the 3D chart looks or move the camera.",
        "parameters": {"type": "object", "properties": {
            "chart": {"type": "string", "enum": list(L.CHARTS)},
            "perplexity": {"type": "number", "description": "look-alike map only: neighbourhood size 5-50"},
            "x": {"type": "string"}, "y": {"type": "string"}, "z": {"type": "string"},
            "color_by": {"type": "string"},
            "rotate_degrees": {"type": "number", "description": "turn around the vertical axis"},
            "tilt_degrees": {"type": "number", "description": "look more from above (+) or below (-)"},
            "zoom": {"type": "number", "description": ">1 zooms in, <1 zooms out"},
            "preset_view": {"type": "string", "enum": ["top", "front", "side", "default"]},
            "clear_highlight": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "suggest_next", "description": "Offer a link to the next place to look.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "short link text, e.g. 'Read the evidence for the Moon base'"},
            "page": {"type": "string", "enum": ["ask", "scenario", "visual", "explore", "about"],
                     "description": "ask = answer from NASA reports; scenario = evidence report for a scenario/pillar"},
            "scenario": {"type": "string"}, "pillar": {"type": "string", "enum": ["prevention", "detection", "suppression"]},
            "data": {"type": "string", "enum": list(L.DATASETS)}, "chart": {"type": "string", "enum": list(L.CHARTS)},
            "q": {"type": "string", "description": "question to carry over"}}, "required": ["label", "page"]}}},
    {"type": "function", "function": {
        "name": "run_analysis", "description": "Run an analysis on the current design matrix.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "enum": list(L.ANALYSES)},
            "params": {"type": "object"}}, "required": ["name"]}}},
]


def _short_state(spec):
    s = L.describe(spec)
    return {k: s[k] for k in ("dataset", "rows_shown", "chart", "axes", "color_by", "filters")} | {
        "variables_in_use": [r["variable"] for r in s["design_matrix"] if r["use"]]}


def execute(spec, name, args):
    """Run one tool call. Returns (new_spec, json_result_for_model, ui_result_or_None)."""
    try:
        if name == "suggest_next":
            if args.get("scenario") and args["scenario"] not in L.PRESETS:
                raise ValueError(f"scenario must be one of {list(L.PRESETS)}")
            return spec, json.dumps({"ok": True, "shown": "link offered to the reader"}), None
        if name == "get_state":
            return spec, json.dumps(L.describe(spec), default=str), None
        if name in ("set_design_matrix", "set_view"):
            spec = L.apply_tool(spec, name, args)
            return spec, json.dumps({"ok": True, "state": _short_state(spec)}, default=str), None
        if name == "run_analysis":
            spec, res = L.run_analysis(spec, args.get("name"), args.get("params") or {})
            out = {"ok": True, "summary": res["summary"], "state": _short_state(spec)}
            if isinstance(res.get("table"), pd.DataFrame):
                out["table_head"] = res["table"].head(10).to_dict(orient="records")
            return spec, json.dumps(out, default=str), res
        return spec, json.dumps({"ok": False, "error": f"unknown tool {name}"}), None
    except (ValueError, KeyError, TypeError) as exc:
        return spec, json.dumps({"ok": False, "error": str(exc)}), None


def _call(messages, api_key, model):
    body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": 0.2}
    req = Request(URL, data=json.dumps(body).encode(), method="POST",
                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                           "User-Agent": "FlameInFreefall/0.1", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace")
        detail = re.sub(r"gsk_[A-Za-z0-9_-]+", "[redacted]", detail.replace(api_key, "[redacted]"))[:300]
        raise RuntimeError(f"Groq request failed (HTTP {exc.code}). {detail}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Groq could not be reached. Try again later.") from None


def run(spec, history, user_text, api_key=None, model=None, call=None):
    """One user turn. history: list of {"role", "content"} (user/assistant text only).

    Returns (spec, reply_text, actions, results): actions are human-readable tool calls for the UI,
    results are analysis outputs (tables/figures) to display.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    model = model or os.environ.get("GROQ_AGENT_MODEL") or os.environ.get("GROQ_MODEL", "")
    if call is None:
        if not api_key or not model:
            raise ValueError("Set GROQ_API_KEY and GROQ_MODEL (or GROQ_AGENT_MODEL) to use the assistant.")
        call = lambda msgs: _call(msgs, api_key, model)  # noqa: E731

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "system", "content": "Current state: " + json.dumps(_short_state(spec), default=str)},
                *history[-8:], {"role": "user", "content": user_text}]
    actions, results = [], []
    for _ in range(MAX_ROUNDS):
        msg = call(messages)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return spec, (msg.get("content") or "").strip(), actions, results
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            spec, out, res = execute(spec, name, args)
            ok = json.loads(out).get("ok", True)
            actions.append(dict(tool=name, args=args, ok=ok, error=None if ok else json.loads(out).get("error")))
            if res:
                results.append(res)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    return spec, "I stopped after several steps; here is where the chart is now.", actions, results
