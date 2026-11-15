"""Experiment references and dataset provenance.

The master test table keeps the row-level provenance (`source_file` and
`source_location`). This module adds the canonical, shareable references:
PSI datasets, technical reports, and papers. Citations use IEEE style.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / "references.csv"
EXPERIMENT_REFERENCES = ROOT / "experiment_references.csv"

ROLE_ORDER = {"primary": 0, "supporting": 1, "context": 2}
VALID_TYPES = {"dataset", "report", "paper", "webpage"}
VALID_ROLES = set(ROLE_ORDER)


@lru_cache(maxsize=1)
def load_references(path: Path = REFERENCES) -> tuple[dict, ...]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return tuple(csv.DictReader(f))


@lru_cache(maxsize=1)
def load_experiment_references(path: Path = EXPERIMENT_REFERENCES) -> tuple[dict, ...]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return tuple(csv.DictReader(f))


@lru_cache(maxsize=1)
def _reference_index() -> dict[str, dict]:
    return {r["ref_id"]: r for r in load_references()}


def references_for(experiment: str, role: str | None = None) -> list[dict]:
    """Return the references attached to one experiment, primary first."""
    refs = _reference_index()
    links = [r for r in load_experiment_references() if r["experiment"] == experiment]
    if role is not None:
        links = [r for r in links if r["role"] == role]
    links.sort(key=lambda r: (ROLE_ORDER.get(r["role"], 99), r["ref_id"]))
    return [{**refs[r["ref_id"]], "role": r["role"]} for r in links if r["ref_id"] in refs]


def dataset_reference(experiment: str) -> dict | None:
    """Return the primary PSI dataset for an experiment, if one is recorded."""
    for ref in references_for(experiment):
        if ref["ref_type"] == "dataset" and ref["role"] == "primary":
            return ref
    return None


def validate() -> list[str]:
    """Return human-readable problems; an empty list means the files are consistent."""
    errors: list[str] = []
    refs = load_references()
    ref_ids = {r["ref_id"] for r in refs}

    for ref in refs:
        rid = ref.get("ref_id", "?")
        if not rid:
            errors.append("reference without ref_id")
        if ref.get("ref_type") not in VALID_TYPES:
            errors.append(f"{rid}: invalid ref_type {ref.get('ref_type')!r}")
        if not ref.get("title"):
            errors.append(f"{rid}: missing title")
        if not ref.get("authors"):
            errors.append(f"{rid}: missing authors")
        if not ref.get("url") and not ref.get("doi"):
            errors.append(f"{rid}: needs a URL or DOI")
        if ref.get("doi") and not ref["doi"].startswith("10."):
            errors.append(f"{rid}: DOI does not look like a DOI: {ref['doi']!r}")
        if not ref.get("license"):
            errors.append(f"{rid}: missing license")
        citation = ref.get("citation", "")
        if not citation:
            errors.append(f"{rid}: missing citation")
        elif '"' not in citation or "Available:" not in citation:
            errors.append(f"{rid}: citation does not look like IEEE style")
        elif re.search(r"\(\d{4}\)\.", citation):
            errors.append(f"{rid}: citation looks like APA, not IEEE")

    for link in load_experiment_references():
        rid = link.get("ref_id", "?")
        if rid not in ref_ids:
            errors.append(f"{link.get('experiment', '?')}: unknown ref_id {rid!r}")
        if link.get("role") not in VALID_ROLES:
            errors.append(f"{rid}: invalid role {link.get('role')!r}")
        if link.get("role") == "primary" and link.get("ref_id") in ref_ids:
            ref = next(r for r in refs if r["ref_id"] == rid)
            if ref.get("ref_type") != "dataset":
                errors.append(f"{link.get('experiment')}: primary reference {rid} is not a dataset")

    return errors


if __name__ == "__main__":
    problems = validate()
    if problems:
        print("Reference validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"References OK: {len(load_references())} sources, "
          f"{len(load_experiment_references())} experiment links.")
