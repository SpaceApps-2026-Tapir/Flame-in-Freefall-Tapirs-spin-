# Third-party notices and license audit

Last reviewed: 2026-11-14.

This repository combines original code with third-party data, documents and
Python packages. The MIT license in [LICENSE](LICENSE) applies only to the
original code, documentation and `assets/logo.svg`. It does not relicense the
items below.

## NASA data and documents

| Item | Rights | Notes |
| --- | --- | --- |
| NASA PSI **BASS-II** dataset (PSI-25), DOI `10.60555/4qc4-de67` | **CC0-1.0** | DataCite rights metadata verified 2026-11-14. |
| NASA PSI **BASS** dataset (PSI-26), DOI `10.60555/58h4-dy80` | **CC0-1.0** | DataCite rights metadata verified 2026-11-14. |
| NASA PSI **FLEX** dataset (PSI-69), DOI `10.60555/mbq8-0451` | **CC0-1.0** | DataCite rights metadata verified 2026-11-14. |
| NASA PSI **SAME-R** dataset (PSI-101), DOI `10.60555/2np7-jv14` | **CC0-1.0** | DataCite rights metadata verified 2026-11-14. |
| NASA PSI **SAME** dataset (PSI-102), DOI `10.60555/gd4k-8h62` | **CC0-1.0** | DataCite rights metadata verified 2026-11-14. |
| **BASS-II Summary Report**, NASA/TM-20210011385 | **NASA public use permitted** | NTRS record `20210011385`: distribution PUBLIC, determination `GOV_PERMITTED`, no third-party material. Not CC0. |
| NASA-authored PDFs in the shared Drive folder | NASA public-use terms | Downloaded at ingestion time and ignored by Git. Check each original document before redistribution. |
| **Detection of Smoke from Microgravity Fires**, SAE 2005-01-2930 | **Copyright © 2005 SAE International** | Explicit copyright notice in the PDF. Not covered by this repository's MIT license. Do not redistribute without permission. |

The PDFs are not committed to this repository. `documents/pdfs/` and
`documents/chroma/` are ignored; only `documents/manifest.csv` is tracked. If
the SAE paper remains in the shared Drive folder, the team must confirm that it
has permission to redistribute it or remove it from the folder and re-index.

NASA content is generally not subject to copyright in the United States for
non-commercial use, but NASA names, insignia, logotypes and identifiers are
protected. See the
[NASA media usage guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).
This project is not affiliated with or endorsed by NASA.

## Python dependencies

Direct runtime dependencies, as declared in `pyproject.toml` and locked in
`uv.lock`:

| Package | Version | License |
| --- | --- | --- |
| chromadb | 1.5.9 | Apache-2.0 |
| gdown | 6.5.1 | MIT |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pandas | 3.1.1 | BSD-3-Clause |
| plotly | 7.1.0 | MIT |
| pypdf | 7.1.0 | BSD-3-Clause |
| streamlit | 1.67.0 | Apache-2.0 |
| Python | 3.14 | PSF-2.0 |

The locked environment resolves 106 packages. An audit of the installed
metadata found no GPL, AGPL, SSPL or BUSL dependencies. Most transitive
packages are MIT, Apache-2.0, BSD or PSF-licensed. The following packages use
MPL-2.0, a file-level copyleft license:

- `certifi` (MPL-2.0)
- `orjson` (MPL-2.0 AND (Apache-2.0 OR MIT))
- `tqdm` (MPL-2.0 AND MIT)

MPL-2.0 permits distribution as part of a larger work provided that the MPL
covered files remain under MPL-2.0 and their source is available. The package
metadata and license files are included with the installed distributions.

The exact dependency graph and versions are in `uv.lock`. If dependencies
change, re-run the audit before release.

## External services

The Ask page can send questions, retrieved passages and selected scenario
context to the Groq API for answer generation. Groq is an external service
governed by its own terms and privacy policy. No API key is stored in this
repository; credentials are supplied through environment variables or the
app's Streamlit secrets.
