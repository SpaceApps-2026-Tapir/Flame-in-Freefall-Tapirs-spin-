"""Where should a question take the reader? A small router used by the home search box.

route("Can smoke tell which material is burning?") ->
    {"page": "visual", "data": "same", "chart": "space", "q": "...", "why": "..."}

Rules first (works offline and is predictable); the visual page's assistant then picks up the question
and keeps guiding. Questions that ask for an explanation go to the article, where the PDF assistant answers.
Written with AI assistance (Claude); see README "Use of AI".
"""
import re

from evidence import PRESETS

SCENARIO_WORDS = {
    "Moon base (challenge brief)": r"\b(moon|lunar|artemis|rover|base)\b",
    "Exploration atmosphere": r"\b(exploration|34 ?%|8\.2 ?psi)\b",
    "Spacesuit": r"\b(suit|spacesuit|eva|pure oxygen|100 ?%)\b",
    "Mars habitat (assumed)": r"\b(mars|martian)\b",
    "ISS cabin": r"\b(iss|station|cabin|earth air|sea level)\b",
}
PILLAR_WORDS = {
    "detection": r"\b(smoke|detect\w*|alarm|sensor|particle\w*|notice)\b",
    "suppression": r"\b(put (it )?out|extinguish\w*|suppress\w*|co2|carbon dioxide|helium|droplet\w*|fuel)\b",
    "prevention": r"\b(ignit\w*|flammab\w*|catch fire|material\w*|fabric\w*|plastic\w*|pmma|nomex|spread\w*|safe)\b",
}
SHOW_WORDS = r"\b(show|see|map|chart|plot|graph|visual\w*|compare|where|which tests|pattern|group|cluster|unusual)\b"
EXPLAIN_WORDS = r"\b(why|explain|what does|what is|how does|meaning|paper|report|source|cite)\b"


def route(question):
    q = (question or "").strip()
    t = q.lower()
    scenario = next((s for s, rx in SCENARIO_WORDS.items() if re.search(rx, t)), None)
    pillar = next((p for p, rx in PILLAR_WORDS.items() if re.search(rx, t)), None)
    wants_view = bool(re.search(SHOW_WORDS, t))
    wants_text = bool(re.search(EXPLAIN_WORDS, t))

    if not q:
        return dict(page="home", why="")
    if wants_view or (not wants_text and pillar in ("detection", "suppression") and not scenario):
        data = {"detection": "same", "suppression": "flex"}.get(pillar, "all")
        chart = {"same": "space", "flex": "mixture"}.get(data, "space" if scenario else "evidence")
        return dict(page="visual", data=data, chart=chart, q=q,
                    why="This is a question about patterns in the data, so you start on a chart.")
    params = dict(page="ask", q=q)
    if scenario:
        params["scenario"] = scenario
    if pillar:
        params["pillar"] = pillar
    params["why"] = "This needs an explanation from NASA's reports, so you start on Ask the evidence."
    return params


def url(params):
    from urllib.parse import quote
    return "?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items() if k != "why" and v is not None)


BRANCHES = [  # the main paths from the landing page into the atlas, from broad to specific
    dict(title="Is a material safe on the Moon?", sub="Prevention · solids at Moon-base air",
         params=dict(page="scenario", scenario="Moon base (challenge brief)", pillar="prevention"), teaser="prevention"),
    dict(title="Would we notice the fire?", sub="Detection · smoke particles and detectors",
         params=dict(page="visual", data="same", chart="space",
                     q="Can smoke particles tell which material is burning?"), teaser="detection"),
    dict(title="How do we put it out?", sub="Suppression · gas mixtures that stop a flame",
         params=dict(page="visual", data="flex", chart="mixture",
                     q="Which gas mixtures put droplet flames out?"), teaser="suppression"),
    dict(title="See all the evidence", sub="What NASA tested, by pillar and outcome",
         params=dict(page="visual", data="all", chart="evidence"), teaser="evidence"),
    dict(title="Filter every test", sub="All 678 tests, any column, with sources",
         params=dict(page="explore"), teaser="table"),
    dict(title="How the numbers were made", sub="Models, scores and their limits",
         params=dict(page="analysis"), teaser="models"),
]

assert all(b["params"].get("scenario", "ISS cabin") in PRESETS for b in BRANCHES)
