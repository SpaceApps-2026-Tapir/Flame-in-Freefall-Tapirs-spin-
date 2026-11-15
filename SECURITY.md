# Security policy

Last reviewed: 2026-11-14.

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting** for this repository. Do
not open a public issue with exploit details, credentials or personal data.
Include the affected version, reproduction steps and the impact you expect.

## Supported versions

Security fixes target the current `main` and `develop` branches. Feature
branches are development snapshots and may be rewritten.

## Security model

- This is a public Streamlit demo. It has **no built-in authentication**.
  Deploy it behind an authenticated reverse proxy if it is exposed to the
  internet, especially when a paid Groq key is configured.
- Questions, retrieved NASA passages and selected scenario context are sent to
  the Groq API when answer generation is enabled. Do not submit confidential or
  personal information.
- NASA PDFs and the local Chroma index are public research data. They are not
  trusted code and their text may contain misleading content.
- The model has no tools, filesystem access or network access. It can only
  return text. Citation IDs are validated, but factual support still needs
  human review.
- Chroma runs in-process through `PersistentClient`. Do not expose a Chroma HTTP
  server or enable remote model repositories/`trust_remote_code`.

## Secrets

Keep `GROQ_API_KEY` outside Git. Use an environment variable,
`.streamlit/secrets.toml` (ignored), or the deployment's secret store.

```bash
export GROQ_API_KEY='your-api-key'
export GROQ_MODEL='your-supported-model-id'
```

Use a dedicated key with a spending limit and rotate it if it was ever pasted
into a chat, issue, log or commit. Provider error messages are redacted before
they reach the UI, but logs and browser devtools can still reveal what you put
there yourself.

Never commit:

- `.env`, `.env.*`, `.streamlit/secrets.toml`
- API keys, tokens, cookies or private keys
- downloaded PDFs, generated Chroma indexes or local caches

## Deployment checklist

- Serve only over HTTPS.
- Put authentication and rate limiting in front of the app. Streamlit's session
  state is not an authentication system.
- Keep Streamlit's XSRF protection and CORS checks enabled. The checked-in
  `.streamlit/config.toml` enables both, limits uploads, disables static file
  serving and hides error details. For local debugging, override the last one
  with `STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=full`.
- Do not expose `documents/pdfs/`, `documents/chroma/`, the repository or the
  `.streamlit` directory as static files.
- Run ingestion as a trusted operator task. Treat `documents/manifest.csv` as
  code: URLs must be HTTPS and PDFs are size/page limited.
- Pin dependencies with `uv.lock` and deploy with `uv sync --locked --python 3.14`.
- Monitor Groq usage and set a project budget/alerts.
- Review logs without copying secrets into them.

## Input and output hardening

- Questions are capped at 2,000 characters before retrieval or generation.
- Model output is rendered as Markdown without raw HTML; Markdown links and
  images are stripped so the model cannot add navigation or tracking payloads.
- External links and PDF links are accepted only if they use `http(s)` and do
  not contain embedded credentials.
- CSV downloads neutralise spreadsheet formula prefixes in text cells.
- The model prompt treats documents and questions as untrusted data and tells
  the model to ignore instructions found inside them.

These measures reduce risk; they do not make prompt injection or model mistakes
impossible. Always show sources and keep human review in the loop.

## Dependency and static checks

Run these before a release:

```bash
uv lock --check
uv sync --locked --python 3.14
uv run --python 3.14 python -m unittest discover -s tests
uvx bandit -q -r . -x ./.venv,./documents,./tests,./.git
uvx --python 3.14 pip-audit -r requirements.txt --format=columns
```

### Known advisories

As of 2026-11-14, `pip-audit` reports five advisories for `chromadb==1.5.9`:

- `PYSEC-2026-311`: pre-authentication code injection through the Chroma HTTP
  collection API and `trust_remote_code`.
- `PYSEC-2026-3813` and `PYSEC-2026-3815`: authorization/tenant isolation
  failures in Chroma server deployments.
- `PYSEC-2026-3814`: code injection through the Chroma HTTP collection API with
  `UPDATE_COLLECTION` permission.

No fixed release is available in the package index at the time of this review.
This project does not run Chroma's HTTP server, does not enable
`trust_remote_code`, and does not use Chroma authentication or multi-tenancy;
the affected server endpoints are not exposed. Re-check the advisories and
upgrade as soon as a patched `chromadb` release is available. If you deploy
Chroma as a service, treat these as blocking until patched.

`bandit` previously flagged `urlopen`. `rag.py` uses a fixed HTTPS Groq
endpoint; ingestion validates the manifest URL scheme and host before fetching.
Those two call sites carry narrow `# nosec B310` annotations with the reason.
