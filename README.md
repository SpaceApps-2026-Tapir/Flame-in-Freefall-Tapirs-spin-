# Flame-in-Freefall-Tapirs-spin-

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
