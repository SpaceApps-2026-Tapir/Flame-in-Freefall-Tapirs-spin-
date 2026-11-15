# Flame-in-Freefall-Tapirs-spin-

## PDF questions and answers (RAG)

PDF passages are stored in persistent local ChromaDB with local ONNX MiniLM
embeddings and optional Groq answer generation. Retrieval needs no API key.
Search combines semantic and keyword rankings, skips title slides and tables of
contents, and includes neighbouring chunks on the same page for context.
The first embedding run downloads model weights (internet access required).
MiniLM is primarily English-oriented; evaluate Spanish questions against your
English NASA PDFs before relying on cross-language retrieval.

1. Run `uv sync --locked`.
2. The manifest uses the PDFs from the team's shared
   [Google Drive folder](https://drive.google.com/drive/folders/1YXhHQEK82FCtzmncAq9-JR7_xbUgDaXv).
   Refresh its file list after adding or removing PDFs:

   ```bash
   uv run python scripts/sync_drive_manifest.py
   ```

   This replaces `documents/manifest.csv` with the current Drive PDF listing.
   Public link access and downloads must be enabled; private folders are not
   supported by this ingestion path. Experiment tags are inferred from filenames;
   review them before indexing. You can edit titles/tags manually, but the next
   sync regenerates them. Keep original NASA provenance in your source records.
3. Download and index the documents:

   ```bash
   uv run python scripts/ingest_documents.py
   ```

   Downloads are cached in `documents/pdfs/`. The script rebuilds
   `documents/chroma/`, preserving PDF page numbers with overlapping 100-word passages.
   Re-run it after changing the manifest. Each run publishes a full snapshot,
   so removed documents disappear from searches. Older snapshots remain on disk
   for active readers; stop the app and remove `documents/chroma/` before
   rebuilding if you need to reclaim space. The old JSON index is no longer used. To refresh a changed PDF at the same
   URL, remove its cached file first. Scanned PDFs need OCR before ingestion.
   PDF page numbers may differ from printed report page numbers.

4. To enable answers, set these environment variables in your terminal:

   ```bash
   export GROQ_API_KEY='your-api-key'
   export GROQ_MODEL='your-supported-groq-model-id'
   uv run streamlit run streamlit_template.py
   ```

   Choose a model available to your account; use its exact model ID from the Groq console.
   Instead of `export`, you can put the same two lines (`GROQ_API_KEY = "..."`, `GROQ_MODEL = "..."`)
   in `.streamlit/secrets.toml` (gitignored), or in the app's Secrets settings on Streamlit Cloud.
   The app does not load `.env` files.
   Never commit API keys. Generated answers send the question, retrieved
   passages, and selected scenario/test context to Groq and may incur API charges.

5. Search from the home page: the answer appears at the top of the next page, under
   **Ask the evidence**, with its sources. Ask follow-ups in the same box. Review the
   excerpts and **Open source PDF** links.
   Drive opens its document viewer; use the displayed PDF page number if the
   viewer does not honor the page anchor.
   Without credentials, the app shows passages only. With a mission scenario,
   the model also receives ranked tests and computed evidence gaps.

Commit the manifest and code; downloaded PDFs and generated indexes are ignored.
Deployment must run ingestion before serving PDF searches. Vector distance is only a retrieval signal: nearest neighbours may be
irrelevant even when there is no supporting evidence. No relevance threshold
is enabled until calibrated on real questions. Citation IDs are checked, but factual support still needs
human review. Try supported, unrelated, and unanswerable questions before use.

Offline checks: `uv run --python 3.14 python -m unittest discover -s tests`.
Groq integration follows the [Chat Completions API](https://console.groq.com/docs/api-reference).
With `openai/gpt-oss-120b` or `openai/gpt-oss-20b`, answers use Groq's strict
structured output: each claim includes validated source IDs, and the app renders
the citation markers. Other models retain text citation validation. Definitions
are allowed when supported by retrieved passages. A refusal means these passages
were insufficient, not that every document in the collection lacks the answer.
Source IDs validate references, not the factual correctness of a claim.
The UI displays IEEE-style numeric citations in first-appearance order, such as
`[1, p. 43]`. Multiple passages from one document share a reference number. Document
titles, PDF page numbers and source links appear in the expandable retrieved
passages; there is no separate reference list below the answer.

## Security

Basic hardening and deployment guidance are in [SECURITY.md](SECURITY.md).
The app has no built-in authentication: put it behind an authenticated reverse
proxy if it is public, and use a dedicated Groq key with a spending limit. Keep
keys out of Git and review the deployment checklist before exposing the app.

The checked-in `.streamlit/config.toml` enables XSRF/CORS protection, disables
static file serving, limits uploads and hides error details. Questions are
capped at 2,000 characters; model output cannot add Markdown links or images;
PDF links must be `http(s)`; and CSV downloads neutralise spreadsheet formula
prefixes. `SECURITY.md` also records the current `chromadb` advisories and why
they do not apply to this app's local, in-process index.

## Team members

- Andres Blanco
- Fabian Gomez
- Lesmes Torres
- Patrick Nepveu
- Xavier Barrios

## Quick guide: Python with uv on Linux

`uv` manages Python dependencies and virtual environments.

**1. Install uv on Linux**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal and verify the installation with `uv --version`. See the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) for more options.

**2. Install the project dependencies** from the repository root:

```bash
uv sync --locked
```

The project is named `flame-in-freefall` in `pyproject.toml`. The repository folder
ends with a hyphen, so plain `uv init` cannot infer a valid project name. This
repository is already initialized; no further `uv init` is needed. For a fresh
setup without a project file, use `uv init --bare --name flame-in-freefall`.

This is a Streamlit application with `package = false`; it runs directly from
`streamlit_template.py` and does not need a build backend or an `__init__.py`.
The current configuration requires Python 3.14 or newer, matching the existing
development environment. `uv` manages `.venv` automatically.

**3. Make sure the evidence files are available.** The app needs `evidence.py`,
`master_tests.csv`, `references.py`, `references.csv`, and
`experiment_references.csv` alongside `streamlit_template.py`. If the evidence
files are missing on your branch, you can temporarily copy them from
`rank-and-gap` without switching branches:

```bash
git restore --source=rank-and-gap -- evidence.py master_tests.csv
```

If you only have the remote branch, use `--source=origin/rank-and-gap` instead.
Keep these files uncommitted if you only need them for local testing. The
reference files are tracked on `feat/experiment-references`.

**4. Start the Streamlit app:**

```bash
uv run streamlit run streamlit_template.py
```

Open the **Local URL** shown in the terminal (usually http://localhost:8501).
Keep the terminal running while using the app. Press **Ctrl+C** to stop it.

Use the full command above: `uv run streamlit_template.py` executes a regular
Python script and produces `missing ScriptRunContext` warnings because the
Streamlit server has not started.

The evidence module imports NumPy and pandas. Both are declared in
`pyproject.toml` and pinned in `uv.lock`, so `uv sync --locked` installs them.
`requirements.txt` is generated from the lock file and is not the source of
truth:

```bash
uv export --format requirements-txt --no-hashes --output-file requirements.txt
```

If you see `No module named 'evidence'`, check step 3. If the error mentions
`master_tests.csv`, make sure you copied the dataset as well as the module.

**5. Common commands** (package names are examples):

| Command | Purpose |
| --- | --- |
| `uv add requests` | Add a dependency. |
| `uv add --dev pytest` | Add a development dependency. |
| `uv remove requests` | Remove a dependency. |
| `uv sync` | Sync the environment with the project's dependencies. |
| `uv export --format requirements-txt --no-hashes --output-file requirements.txt` | Regenerate `requirements.txt` from `uv.lock`. |
| `uv run python -c "print('Hello from uv')"` | Run Python in the project's environment. |

With `uv run`, you do not need to activate the environment manually. Track `pyproject.toml` and `uv.lock` in Git, and add `.venv/` to `.gitignore`. See the [official project guide](https://docs.astral.sh/uv/guides/projects/) for more details.

## Run with Docker

The image uses Astral's official `uv` image with Python 3.14 and installs exactly
what `uv.lock` pins. `evidence.py` and `master_tests.csv` must be in the repository
root before building (see step 3), because they are copied into the image.

```bash
docker compose up --build      # build and start → http://localhost:8501
docker compose up -d           # start in the background
docker compose logs -f         # follow logs
docker compose down            # stop
```

Without Compose:

```bash
docker build -t flame-in-freefall .
docker run --rm -p 8501:8501 flame-in-freefall
```

API keys (e.g. for the AI assistant) go in a `.env` file next to `compose.yaml`;
it is git-ignored and loaded automatically. After changing dependencies with
`uv add`, rebuild with `docker compose up --build`.

## Explore the data visually (`?page=visual`, "Explore visually" in the top bar)

3D view of the tests (real axes, PCA, UMAP or t-SNE), an editable **design matrix** (which variables enter
the analysis, their weight and transform), one-click analyses (PCA drivers, clusters, outliers, tests nearest
a scenario, correlations, what predicts a category) and an assistant that drives all of it through tool calls.

- Engine: `lab_engine.py` (no Streamlit, no LLM). Agent: `lab_agent.py`. Page: `visual_lab.py`.
- The assistant uses the same Groq key: `GROQ_API_KEY`, and `GROQ_AGENT_MODEL` (falls back to `GROQ_MODEL`).
  The model must support tool use. Without a key, the rest of the page still works.
- Raw FLEX/SAME tables: `python scripts/extract_raw_data.py "<NASA RESOURCES>" data/raw`.
- Tests: `uv run python -m unittest tests.test_lab`.

## Landing and simulator

The home page tells one story under the search box: (1) what NASA has tested, (2) what the data reveals
(flames stop putting themselves out as oxygen rises), (3) a droplet-burn simulator compared with a real NASA
burn, (4) what to test next. Code: `story.py`, `sim.py`.

## License

The original code, documentation and the logo in this repository are licensed
under the MIT License. See [LICENSE](LICENSE).

Third-party data and documents keep their own terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full audit. In short:

- The five NASA PSI datasets used by the atlas are released under **CC0-1.0**.
- The BASS-II Summary Report is a NASA/NTRS public-use document, not CC0.
- NASA PDFs are downloaded from the team's Google Drive folder and are not
  tracked in Git. One of them, SAE paper 2005-01-2930, is **copyright SAE
  International** and is not covered by this repository's MIT license. Confirm
  permission before redistributing it.
- Python dependencies are permissively licensed (MIT, Apache-2.0, BSD and
  MPL-2.0); there are no GPL, AGPL, SSPL or BUSL dependencies.

This license does not grant rights to NASA names, logos or trademarks. This
project is not affiliated with or endorsed by NASA.
