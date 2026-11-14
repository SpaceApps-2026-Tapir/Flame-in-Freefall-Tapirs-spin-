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
uv init  # Run once, only if pyproject.toml does not exist yet.
uv sync  # Create .venv and install the project's dependencies.
```

This repository does not have a `pyproject.toml` file yet. Once it is added, run `uv sync` after cloning.

**3. Common commands** (package names are examples):

| Command | Purpose |
| --- | --- |
| `uv add requests` | Add a dependency. |
| `uv add --dev pytest` | Add a development dependency. |
| `uv remove requests` | Remove a dependency. |
| `uv sync` | Sync the environment with the project's dependencies. |
| `uv run python -c "print('Hello from uv')"` | Run Python in the project's environment. |

With `uv run`, you do not need to activate the environment manually. Track `pyproject.toml` and `uv.lock` in Git, and add `.venv/` to `.gitignore`. See the [official project guide](https://docs.astral.sh/uv/guides/projects/) for more details.
