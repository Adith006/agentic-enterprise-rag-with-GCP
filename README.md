# Enterprise Agentic RAG

An agentic retrieval-augmented generation (RAG) application for answering
enterprise knowledge questions. It ingests documents into a searchable
knowledge base, retrieves and ranks relevant passages, and returns an answer
with source material. A separate evaluation interface runs live questions,
guardrail checks, and answer-quality metrics.

## Capabilities

- Routes conversational requests directly to an answer and research requests
  through retrieval before answering.
- Parses supported documents, splits their text into chunks, creates
  embeddings, and indexes the chunks for semantic retrieval.
- Supports PDF, HTML, plain-text, Word, and PowerPoint documents in ingestion.
- Returns source passages alongside answers.
- Applies input guardrails before requests reach the answering workflow.
- Supports in-memory conversation state locally and optional persistent
  checkpointing when configured.
- Includes an evaluation UI for golden questions, guardrail cases, metric
  scores, and evaluation history. The live run defaults to five RAG questions
  and three guardrail cases; both counts are adjustable in the UI.

## Project layout

| Path | Purpose |
| --- | --- |
| `app/main.py` | HTTP API, health check, query endpoint, and graph endpoint |
| `app/agents/` | Agent state, planning, retrieval, response, and workflow graph |
| `app/guardrails/` | Input guardrail setup and rules |
| `app/ingestion/` | Document parsing, chunking, and indexing workflow |
| `app/services/retrieval/` | Embedding, vector search, and reranking helpers |
| `app/gateway/` | Model gateway client setup |
| `ui/app.py` | Interactive chat interface |
| `evals/` | Golden data, live pipeline, guardrail checks, metrics, and history |
| `DATA/` | Example local ingestion input, organized into source subdirectories |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition for the API |

## Requirements

- Python 3.13 or newer.
- Access to the configured model, embedding, vector-index, and optional
  persistence resources.
- Private credentials and environment-specific settings for those resources.

Keep credentials outside source control. Create a local `.env` file or use your
deployment platform's secret-management facility. Do not paste credentials
into this README, source files, notebooks, logs, or issue reports. The required
configuration names are defined in `app/config.py` and the evaluation modules;
the values are specific to each deployment.

## Local setup

Run these commands from the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, allow the activation
script for the current shell according to your machine's policy, or invoke
`.venv\Scripts\python.exe` directly for the commands below.

Set the required private configuration before starting the application. The
main API uses the project configuration in `app/config.py`; evaluation also
requires its judge and embedding configuration. Some optional integrations,
such as persistent conversation state and evaluation history, require their
corresponding resources to be configured.

## Run the application

Start the API from the repository root:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the chat UI:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui/app.py
```

The UI uses the local API address by default. If the API runs elsewhere, set
the backend URL in the UI environment before launching it.

The API provides:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `POST` | `/query` | Run a user query |
| `GET` | `/graph` | Return a visualization of the agent workflow |

Example request body for `POST /query`:

```json
{
  "q": "How does the system handle document retrieval?",
  "thread_id": "example-conversation"
}
```

`thread_id` is optional. Reuse a thread ID to associate requests with the same
conversation/checkpoint; use a new ID for an independent conversation.

## Ingest documents

Place supported files under a directory and run the ingestion module:

```powershell
python -m app.ingestion.processor DATA
```

The input directory may contain source subdirectories. Their names are used to
classify content; directories containing `true` or `noisy` are recognized as
such. If the input directory has no subdirectories, an explicit source type
may be supplied:

```powershell
python -m app.ingestion.processor .\path\to\documents true
```

The ingestion flow parses supported files, chunks their text, creates
embeddings, and indexes the results. It also uses the configured storage and
index resources, so configure those before ingestion.

> **Warning:** `--wipe` deletes the existing vector collection before
> ingestion. Only use it when you intend to rebuild the index.

```powershell
python -m app.ingestion.processor DATA --wipe
```

The ingestion module also exposes an HTTP webhook mode for deployment
environments that deliver document-created events. The container definition
starts the API process on port `8080`.

## Run evaluations

Start the API first, then launch the evaluation UI from another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run evals/app.py
```

The evaluation workflow is:

1. Review the golden RAG questions and guardrail test cases.
2. Choose how many RAG questions and guardrail cases to send to the live API.
   The defaults are five and three, respectively.
3. Run the live pipeline and review captured responses and guardrail results.
4. Run the answer-quality and tool-selection metrics on responses that were
   collected successfully.
5. Review prior runs when evaluation-history storage is configured.

The metrics include faithfulness, answer relevancy, context precision, context
recall, answer correctness, and tool correctness. Guardrail results include
precision, recall, and accuracy.

The evaluation UI loads `evals/golden_dataset.json` when present and otherwise
falls back to `evals/og_golden_dataset.json`. Keep the dataset in the
evaluation directory and do not include private source documents or
credentials in it.

Metric evaluation can take significant time because model-backed checks are
rate-limited and run in small batches. Configure the evaluation model and
embedding access before starting that step.

## Configuration and operations

- Configuration is loaded from environment variables and a local `.env` file.
  Keep `.env` out of version control.
- Local conversation memory is used by default. Persistent checkpointing is
  optional and requires its database configuration.
- Request and evaluation traces can be enabled with the appropriate
  observability configuration.
- The API, UI, ingestion process, and evaluation UI may be run independently;
  the API must be available for the chat UI and live evaluations.

## Container

The root `Dockerfile` builds the API image and starts the application on port
`8080`. Supply deployment configuration and credentials through the runtime
environment; do not bake secrets into the image.

## Troubleshooting

- **Connection refused on `/query`:** Start the API and confirm that the
  configured backend URL and port match the address used by the UI or
  evaluation interface.
- **Golden dataset not found:** Ensure one of the supported dataset filenames
  is present in `evals/`.
- **Evaluation or ingestion cannot reach external dependencies:** Check that
  the deployment's private configuration and required resources are available.
- **Guardrails unavailable at startup:** Review the API logs and its guardrail
  configuration. The API can start even when guardrail initialization fails.
