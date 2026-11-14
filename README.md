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

**2. Set up the project** from the repository root:

```bash
uv sync --locked
uv run streamlit run streamlit_template.py
```

The project is named `flame-in-freefall` in `pyproject.toml`. The repository folder
ends with a hyphen, so plain `uv init` cannot infer a valid project name. This
repository is already initialized; no further `uv init` is needed. For a fresh
setup without a project file, use `uv init --bare --name flame-in-freefall`.

This is a Streamlit application with `package = false`; it runs directly from
`streamlit_template.py` and does not need a build backend or an `__init__.py`.
The current configuration requires Python 3.14 or newer, matching the existing
development environment. `uv` manages `.venv` automatically.

**Startup prerequisite:** `streamlit_template.py` imports `PRESETS` and `load`
from a local `evidence.py`, which is currently missing from the repository.
Restore that module and any data files it requires before running the app.
`PRESETS` must map scenario names to `o2`, `kpa`, `gravity`, and `source` values;
`load()` must return a DataFrame with at least `experiment` and `gravity`
columns. The template also mentions `build_master.py`, which is absent.
The cached `.pyc` file in `__pycache__` is not a replacement for the source.

**3. Common commands** (package names are examples):

| Command | Purpose |
| --- | --- |
| `uv add requests` | Add a dependency. |
| `uv add --dev pytest` | Add a development dependency. |
| `uv remove requests` | Remove a dependency. |
| `uv sync` | Sync the environment with the project's dependencies. |
| `uv run python -c "print('Hello from uv')"` | Run Python in the project's environment. |

With `uv run`, you do not need to activate the environment manually. Track `pyproject.toml` and `uv.lock` in Git, and add `.venv/` to `.gitignore`. See the [official project guide](https://docs.astral.sh/uv/guides/projects/) for more details.
