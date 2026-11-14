"""Plain-language definitions for every technical term shown in the dashboard (used as hover tooltips).

Keep each entry to one or two sentences. If a definition is not confirmed in a NASA source, say so.
"""
from html import escape

GLOSSARY = {
    # --- gravity
    "microgravity": "Near-weightlessness, as on the ISS. Hot gas no longer rises (no buoyancy), so flames are fed "
                    "by slow diffusion or by an imposed airflow instead. Every test in this atlas ran in microgravity.",
    "micro": "Microgravity: near-weightlessness, as on the ISS. Hot gas does not rise, so flames behave very differently.",
    "lunar": "Moon surface gravity, about 0.16 g (one sixth of Earth's). PSI has no combustion data at this gravity yet.",
    "mars": "Mars surface gravity, about 0.38 g. PSI has no combustion data at this gravity yet.",
    "gravity": "The gravity level the test ran at. All 678 tests in this atlas ran in microgravity on the ISS.",
    "partial gravity": "Gravity between zero and Earth's, like the Moon (0.16 g) or Mars (0.38 g). No PSI combustion data yet.",

    # --- atmosphere and units
    "o2": "Oxygen concentration: the share of oxygen in the gas, by volume. Earth's air is about 21%. "
          "More oxygen makes materials ignite more easily and burn faster.",
    "pp": "Percentage points: the plain difference between two percentages. 30% vs 21% O₂ is 9 pp (not 9%).",
    "kpa": "Kilopascal, a unit of pressure. Sea-level air is 101.3 kPa (14.7 psi). 1 psi = 6.895 kPa.",
    "psi": "Pounds per square inch. Sea level is 14.7 psi; the challenge's Moon base is 10.2 psi or less; "
           "a spacesuit is 4.3 psi.",
    "airflow": "Speed of the gas flowing past the sample, in cm/s. In microgravity there is no rising hot air, "
               "so this imposed flow controls how much fresh oxygen reaches the flame.",
    "airflow_min": "Lowest airflow setting recorded for the test, in cm/s.",
    "flow direction": "Which way the gas flowed relative to the flame.",
    "opposed": "Opposed flow: the flame spreads against the airflow (the flow comes toward the flame front).",
    "concurrent": "Concurrent flow: the flame spreads in the same direction as the airflow, which pushes it ahead. "
                  "This is often the faster-spreading case.",
    "wake": "The flame sits in the wake (downstream side) of the sample, for example behind a candle.",
    "stagnation": "The flow hits the sample face-on and spreads outward over it (stagnation-point flow).",
    "quiescent (no forced flow)": "Still gas, no fan-driven flow, as in the FLEX droplet chamber.",
    "smoke generation tube flow": "SAME/SAME-R: material heated in a tube with gas flowing through it, and the "
                                  "resulting smoke sent to the detectors.",
    "not stated": "The NASA record does not say which flow direction was used.",
    "suppressant": "Gas added to make burning harder, such as CO₂ or helium in FLEX. 'none (N₂ balance)' means no "
                   "suppressant: the rest of the gas was nitrogen.",

    # --- fire safety pillars
    "pillar": "NASA's three lines of spacecraft fire safety: prevention, detection and suppression.",
    "prevention": "Stopping fires before they start: controlling ignition sources and choosing materials that are "
                  "hard to ignite (flammability limits). Data: BASS, BASS-II.",
    "detection": "Noticing a fire early, mainly by sensing smoke particles (aerosols). Data: SAME, SAME-R.",
    "suppression": "Putting a fire out, for example with extinguishing agents or by diluting the oxygen with an inert "
                   "gas. Data: FLEX.",

    # --- fuel types and materials
    "fuel type": "What was burned or heated: plastic, fabric, wax, a liquid fuel droplet, or a reference aerosol.",
    "solid (polymer)": "Plastic samples, mostly thin PMMA (acrylic) films.",
    "solid (fabric)": "Fabric samples, such as Nomex and SIBAL (a cotton/fiberglass blend).",
    "solid (wax)": "Candle and wax-slab samples (paraffin, Japan wax).",
    "liquid fuel droplet": "A free-floating droplet of liquid fuel (heptane or methanol) burned as a sphere in FLEX.",
    "reference aerosol": "Dibutyl phthalate (DBP): a test aerosol with known particle sizes, used as a reference "
                         "for the smoke detectors rather than smoke from a burning material.",
    "pmma": "Poly(methyl methacrylate), the clear acrylic plastic also sold as Plexiglas. A standard reference solid fuel.",
    "nomex": "A flame-resistant aramid fabric used in flight suits and firefighter gear.",

    # --- outcomes (our categories, read from NASA notes)
    "outcome": "Our category for how the test ended, read from NASA's original note. Pick the test in "
               "'Show details' to read the note itself.",
    "extinguished": "BASS/BASS-II: the flame went out (often after the airflow was turned down).",
    "extinction": "FLEX test-end label 'Extinction': the flame went out; the droplet size at that moment is logged "
                  "as the extinction diameter. Likely the same idea as BASS's 'extinguished'.",
    "disruption": "FLEX test-end label 'Disruption'. In droplet combustion this usually means the droplet broke "
                  "apart (bubbling or micro-explosion). The exact FLEX definition is still to be confirmed.",
    "completion": "FLEX test-end label 'Completion', as recorded by NASA. The exact definition is still to be "
                  "confirmed in the FLEX papers.",
    "blowoff": "The airflow pushed the flame off the sample and put it out. Check the original note: some notes in "
               "this category say the flame did NOT blow off.",
    "spread": "The flame spread along the sample and kept burning.",
    "no_ignition": "The sample could not be lit, or the flame did not sustain itself.",
    "burned_entire_sample": "The flame burned all the way to the end of the sample.",
    "smoke_measured": "Detection test: smoke was produced, and particle sizes and detector signals were recorded.",
    "see_notes": "The outcome was not clear enough to categorise; read the original note.",
    "not_reported": "The NASA record gives no outcome for this test.",

    # --- ranking and gaps
    "scenario": "A place where crews would live, described by oxygen %, pressure and gravity (e.g. Moon base).",
    "tolerance": "How far a test may be from the scenario and still count as close. A test must be inside both the "
                 "oxygen and the pressure tolerance.",
    "relevance": "How close a test's oxygen and pressure are to the scenario: 1.00 means identical, and it falls as "
                 "the test gets further away. It ignores material, airflow and gravity.",
    "weight": "How much oxygen or pressure differences count in the ranking: 0 ignores it, 1 is normal, "
              "3 makes it three times as important.",
    "why": "How far the test is from the scenario in O₂ and pressure, plus any gravity mismatch or nominal values.",
    "covered": "At least one test is inside the tolerance box, at the scenario's gravity.",
    "covered except gravity": "Tests match the oxygen and pressure, but they ran in microgravity, not at the "
                              "scenario's gravity.",
    "gap": "No test is inside the tolerance box: NASA has no data near these conditions.",
    "nominal": "A value NASA did not log per test, filled in with the standard value for the setup "
               "(for example ISS cabin pressure). Listed in 'Nominal values used for'.",

    # --- data and sources
    "psi_db": "Physical Sciences Informatics: NASA's public database of physical-science experiment data from the ISS.",
    "psi_id": "The NASA Physical Sciences Informatics (PSI) dataset the test comes from.",
    "experiment": "The ISS investigation that ran the test (BASS, BASS-II, SAME, SAME-R or FLEX).",
    "test_id": "NASA's identifier for the test within its experiment.",
    "material": "The sample as NASA described it (material, size, shape).",
    "condition_detail": "Test settings as logged by NASA, such as the investigator, display readings and flow settings.",
    "original note": "NASA's own text for how the test went; our outcome category is read from this.",
    "source": "The NASA file and row or page each value comes from.",

    # --- experiments and hardware
    "bass": "Burning and Suppression of Solids (ISS): how solid materials ignite, spread and go out in low-speed airflow.",
    "bass-ii": "Burning and Suppression of Solids II: the second BASS campaign, with more materials and flows.",
    "same": "Smoke and Aerosol Measurement Experiment: smoke particle sizes from heated spacecraft materials and how "
            "smoke detectors respond.",
    "same-r": "SAME Reflight: a second SAME campaign with more materials.",
    "flex": "Flame Extinguishment Experiment: burning fuel droplets in different oxygen, pressure and suppressant "
            "(CO₂, helium) mixtures.",
    "msg": "Microgravity Science Glovebox: a sealed ISS work area. BASS ran in it, in a fan-driven flow duct.",
    "cir": "Combustion Integrated Rack: the ISS rack with a sealed combustion chamber, used by FLEX.",
    "mdca": "Multi-user Droplet Combustion Apparatus: the CIR insert that deploys and ignites fuel droplets (FLEX).",
}

# Data column -> glossary key (for table headers and filter widgets)
COLUMN_HELP = {
    "experiment": "experiment", "psi_id": "psi_id", "pillar": "pillar", "test_id": "test_id", "material": "material",
    "material_class": "fuel type", "oxygen_pct": "o2", "pressure_kpa": "kpa", "airflow_cms": "airflow",
    "airflow_min_cms": "airflow_min", "flow_direction": "flow direction", "gravity": "gravity",
    "suppressant": "suppressant", "condition_detail": "condition_detail", "outcome_category": "outcome",
    "outcome_raw": "original note", "assumed_fields": "nominal", "source_file": "source", "source_location": "source",
    "relevance": "relevance", "why": "why",
}


def tip(key):
    """Plain-text definition for widget `help=` and table headers (None if unknown)."""
    return GLOSSARY.get(str(key).lower())


def term(label, key=None):
    """HTML: `label` with a dotted underline that shows the definition on hover."""
    text = GLOSSARY.get(str(key if key is not None else label).lower())
    if text is None:
        return escape(str(label))
    return f"<abbr class='gl' title='{escape(text, quote=True)}'>{escape(str(label))}</abbr>"
