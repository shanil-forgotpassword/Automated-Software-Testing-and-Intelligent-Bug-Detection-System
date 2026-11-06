# CodeGuard AI

**Academic project:** AI-Powered Automated Software Testing and Intelligent Bug Detection System  
**Tagline:** Test Smart. Detect Early. Code Better.

CodeGuard AI is a local Streamlit application for organizing Python project requirements, reviewing test cases, running approved tests in a restricted Docker container, reviewing local Ollama suggestions, and preserving execution and code-version history. Test results are stored only after an actual Pytest run; unavailable Docker never falls back to running project code on the host.

## Features

- Four sidebar destinations: New Project, Recents, Demo, and About.
- SQLite persistence for projects, requirements, original source, code versions, tests, executions, per-test results, AI analyses, correction decisions, and report metadata.
- Manual test creation and local Ollama (`qwen2.5:3b`) test generation, with requirement mapping, explainable complexity classification, duplicate removal, search/filter, and approval review.
- Automatically generated cases are approved only after source/signature validation; manually created tests remain drafts for review. Generation starts after submitting requirements with Python source, or after adding source in the workspace; implementation code remains user-authored.
- Basic and intermediate requirements are sent for test generation. Advanced requirements are identified with reasons and excluded from automated generation; they remain visible as untested scope in the project overview and reports.
- Intermediate generation uses AST-visible parameters, annotations, branches, comparison thresholds, raised exceptions, and local calls to guide empty/null/type, boundary, edge, exception, conditional, combination, validation, and failure cases. Ollama must mark each category applicable or not applicable with a reason; applicable categories require a matching generated case.
- Per-category case targets adapt to inspected parameters, branches, boundaries, and exceptions. Generation combines categories into bounded batches, streams progress from local Ollama, uses a 60-second idle timeout per request, retries incomplete batches once, preserves valid cases, and rejects malformed or truncated JSON.
- Generated test cases retain separate descriptions and testing objectives, and are tagged Basic or Intermediate from the actual test category. Positional arguments are raw ordered values; named arguments are passed only through keyword input objects.
- Editable Python code editor; generation and execution save the current editor contents as a version before checking tests against that source.
- Approved module-function and class-method tests (including constructor arguments, setup calls, and keyword arguments) run with Pytest in a Docker container configured with no network, a read-only root filesystem, a read-only source mount, a non-root user, resource limits, dropped Linux capabilities, and a timeout.
- Code-version history, diffs, restore-as-a-new-version, actual failure analysis, reviewed correction suggestions, and retesting.
- Actual Coverage.py line measurements, requirement-to-test linkage, execution comparisons, and PDF/JSON/CSV/code downloads.

## Project layout

```text
app.py                    Streamlit entry point
codeguard/
  ai.py                   Local Ollama requests and response validation
  database.py             SQLite schema and connection management
  execution.py            Docker-only sandbox orchestration
  reports.py              PDF, JSON, and CSV exports
  services.py             Persistence, validation, coverage, comparisons
  ui.py                   Navigation and project workflows
sandbox/
  Dockerfile              Isolated Python/Pytest/Coverage image
  runner.py               In-container test generation and result protocol
assets/styles.css         Permanent dark design system
tests/                    Automated application tests
data/                     Local SQLite database (created at runtime)
```

## Requirements

- Windows 10/11 with Python 3.11.
- Docker Desktop using Linux containers, started before test execution.
- Optional: Ollama running locally for AI test generation, analysis, and suggestions.
- Internet access is needed for initial package/model/image downloads. The app's Ollama client always connects to `127.0.0.1:11434`; it does not accept a user-supplied endpoint.

## Setup (Windows PowerShell)

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use this for the current terminal only, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install Docker Desktop, select Linux containers, start Docker Desktop, and verify:

```powershell
docker info
```

The application builds `codeguard-sandbox:latest` automatically the first time a test run needs it. You can build it ahead of time:

```powershell
docker build --tag codeguard-sandbox:latest --file sandbox\Dockerfile sandbox
```

To enable local AI features, install and start Ollama, then download the model:

```powershell
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

Keep Ollama bound to the local machine. AI is optional; test execution does not require Ollama.

## Start the application

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local URL printed by Streamlit. The SQLite database is created in `data\codeguard.sqlite3`; set `CODEGUARD_DATA_DIR` before starting the app to choose another data directory.

## Test and function contract

Run the automated application tests:

```powershell
python -m pytest -q
```

Test cases target either an exact module-level function or a named class method. Positional arguments are JSON arrays; keyword arguments are JSON objects and are checked against the inspected signature. Class tests keep constructor arguments separate from target-method arguments. Setup steps run on the same instance before the target call and can store return values for later steps or the target call. Expected values and optional allowlisted exceptions are checked by real Pytest inside Docker. Source targets and signatures are revalidated when a test is saved, approved, and executed; stale or incompatible tests cannot run. AI-generated tests are automatically approved only after validation against the current source, while manually created tests remain pending review. The schema migration resets existing approvals so legacy cases must be revalidated.

On workspace entry, CodeGuard generates drafts once when requirements and Python source are available. If requirements are submitted without source, add or paste the implementation into the editor; CodeGuard saves it as a Python version and starts generation automatically. If generation fails, the workspace offers a retry. Basic and intermediate scope is supported by the configured `qwen2.5:3b` model. Advanced scope is disclosed with per-requirement reasons but is not automatically generated or executable; provide specialist test design and fixtures separately. Generated cases are deduplicated and can be filtered by search, category, complexity, and approval status before review.

The test manager shows per-category applicability, the model's reason, generated/approved counts, and actually passed cases from the latest run. This coverage assessment is tied to the exact source SHA-256 and is refreshed when generation runs against updated code. Generated cases are validated against inspected callable signatures and source-visible behavior; unambiguous positional `{name, value}` wrappers are normalized, while incompatible signatures and malformed cases may trigger a bounded corrective generation attempt. For source-derived thresholds, boundary generation tracks and validates each exact requested value. The generation summary reports duplicates removed, invalid cases rejected, and cases that could not be generated after retries.

The sandbox image is built from `python:3.11-slim` and installs pinned Pytest and Coverage.py versions. Container execution may be unavailable if Docker Desktop is stopped, not using Linux containers, or cannot build the image. In that case the failed infrastructure attempt is recorded with an explicit error; source is never executed on the host.

## Common issues and limitations

- **Docker engine unavailable:** Start Docker Desktop and confirm `docker info` succeeds. CodeGuard does not bypass this by running code locally.
- **Sandbox image build fails:** Check the Docker Desktop connection and network access for the first image build.
- **Ollama unavailable or slow:** Start Ollama and verify `ollama list` includes `qwen2.5:3b`. Connection failures and inference timeouts are reported separately; retry after confirming the model is responsive. The AI feature will not save a fabricated response.
- **Uploaded code rejected:** Upload UTF-8 `.py` text no larger than 200 KB with valid Python syntax.
- **Unsupported test model:** Tests target inspected synchronous Python functions and methods with JSON-compatible inputs/outputs, setup steps, and allowlisted expected exceptions. Async callables and arbitrary user-authored executable pytest code are not supported.
- **Security boundary:** Docker restrictions reduce exposure but do not guarantee perfect isolation. Keep Docker and its host current, do not mount sensitive host folders, and review AI code suggestions before approval.
- **Coverage:** Line coverage is reported only when Coverage.py produced data inside a real sandbox execution. It is not requirement coverage.
- **Local-only data:** SQLite and generated downloads are not synchronized or backed up automatically. Protect and back up the data directory as appropriate.
