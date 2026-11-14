# Flame-in-Freefall-Tapirs-spin-

## PDF questions and answers (RAG)

PDF passages are stored in persistent local ChromaDB with local ONNX MiniLM
embeddings and optional Groq answer generation. Retrieval needs no API key.
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
   The app reads environment variables directly; it does not load `.env` files.
   Never commit API keys. Generated answers send the question, retrieved
   passages, and selected scenario/test context to Groq and may incur API charges.

5. Search from the home page, then select **Find evidence and answer** in
   **Ask the evidence**. Review the excerpts and **Open source PDF** links.
   Drive opens its document viewer; use the displayed PDF page number if the
   viewer does not honor the page anchor.
   Without credentials, the app shows passages only. With a mission scenario,
   the model also receives ranked tests and computed evidence gaps.

Commit the manifest and code; downloaded PDFs and generated indexes are ignored.
Deployment must run ingestion before serving PDF searches. Vector distance is only a retrieval signal: nearest neighbours may be
irrelevant even when there is no supporting evidence. No relevance threshold
is enabled until calibrated on real questions. Citation IDs are checked, but factual support still needs
human review. Try supported, unrelated, and unanswerable questions before use.

Offline checks: `uv run python -m unittest discover -s tests`.
Groq integration follows the [Chat Completions API](https://console.groq.com/docs/api-reference).

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

**3. Make sure the evidence files are available.** The app needs `evidence.py`
and `master_tests.csv` alongside `streamlit_template.py`. If they are missing
on your branch, you can temporarily copy them from `rank-and-gap` without
switching branches:

```bash
git restore --source=rank-and-gap -- evidence.py master_tests.csv
```

If you only have the remote branch, use `--source=origin/rank-and-gap` instead.
Keep these files uncommitted if you only need them for local testing.

**4. Start the Streamlit app:**

```bash
uv run streamlit run streamlit_template.py
```

Open the **Local URL** shown in the terminal (usually http://localhost:8501).
Keep the terminal running while using the app. Press **Ctrl+C** to stop it.

Use the full command above: `uv run streamlit_template.py` executes a regular
Python script and produces `missing ScriptRunContext` warnings because the
Streamlit server has not started.

The evidence module imports NumPy and pandas. If either is missing, install
them for the run without changing the project's dependency files:

```bash
uv run --with numpy --with pandas streamlit run streamlit_template.py
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
| `uv run python -c "print('Hello from uv')"` | Run Python in the project's environment. |

With `uv run`, you do not need to activate the environment manually. Track `pyproject.toml` and `uv.lock` in Git, and add `.venv/` to `.gitignore`. See the [official project guide](https://docs.astral.sh/uv/guides/projects/) for more details.
