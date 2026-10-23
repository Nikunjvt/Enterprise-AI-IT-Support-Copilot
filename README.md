# Enterprise IT Support Agentic RAG Copilot

An end-to-end **Forward Deployed Engineer (FDE) project** that turns a notebook-style Agentic RAG workflow into a deployable internal product using **LangGraph, FastAPI, Pinecone, OpenAI, Tavily, HTML, CSS, and JavaScript**.

---

## 1. Business Problem

### Customer
**NovaRetail**, a fictional 3,000-employee retail company.

### Problem
The internal IT team maintains many documents: VPN instructions, password rules, MFA policy, software installation rules, laptop troubleshooting guides, and service-desk runbooks.

Employees still create repetitive support tickets because:

- They do not know where the correct document is.
- Traditional keyword search returns too many results.
- A normal chatbot may hallucinate an answer.
- Internal documents can be incomplete or outdated.
- Some questions require current vendor information from the public web.

### Example
An employee asks:

> **“How do I connect to the company VPN from home?”**

The answer exists in the private company KB, so the system should **not search the public internet**.

Another employee asks:

> **“What is the latest Microsoft Teams outage guidance?”**

The internal KB may not contain current outage information. The system should recognize weak private evidence, use an external search, grade that evidence, and answer with an external-source warning.

### Business Goal
Build a secure IT Support Copilot that:

1. Searches trusted private knowledge first.
2. Checks whether retrieved evidence is good enough.
3. Uses web search only when private knowledge is insufficient.
4. Rewrites weak queries and retries.
5. Generates grounded answers.
6. Shows the decision path for transparency and debugging.
7. Lets authorized staff add new company documents.

---

## 2. Why This Is an FDE Project

A Forward Deployed Engineer does more than create an LLM notebook. The FDE must turn the customer's problem into a usable product.

```text
Customer Problem
      ↓
Discovery & Requirements
      ↓
Solution Architecture
      ↓
Data / Knowledge Integration
      ↓
Agentic RAG Development
      ↓
API Development
      ↓
User Interface
      ↓
Security + Audit + Testing
      ↓
Deployment
      ↓
Observe + Improve
```

This repository demonstrates each layer.

---

## 3. Simple Architecture

```text
                   ┌─────────────────────┐
                   │   Employee / User   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ HTML/CSS/JavaScript │
                   │      Web UI         │
                   └──────────┬──────────┘
                              │ POST /api/chat
                              ▼
                   ┌─────────────────────┐
                   │       FastAPI       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │     LangGraph       │
                   │ Agentic RAG Control │
                   └──────────┬──────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
       ┌───────────────┐              ┌──────────────┐
       │ Private KB    │              │ Tavily Web   │
       │ Pinecone      │              │ Search       │
       └───────┬───────┘              └──────┬───────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                     ┌────────────────┐
                     │ OpenAI LLM     │
                     │ Grounded Answer│
                     └────────────────┘
```

---

## 4. Agentic RAG Workflow

This follows the same core implementation pattern as the reference notebook.

```text
Question
   ↓
[1] Route Question
   ├── Greeting / simple chat ─────────────→ Direct Answer
   │
   └── IT support question
                ↓
[2] Retrieve from Private Pinecone KB
                ↓
[3] Grade Private Evidence
       ┌────────┴────────┐
       │                 │
     GOOD               WEAK
       │                 │
       ▼                 ▼
Generate from KB    [4] Tavily Web Search
                         ↓
                  [5] Grade Web Evidence
                    ┌────┴─────┐
                    │          │
                  GOOD        WEAK
                    │          │
                    ▼          ▼
              Generate Web  [6] Rewrite Query
                               ↓
                         Retry Private KB
                               ↓
                        Max retry reached?
                               ↓
                    Insufficient Evidence
```

### Why Agentic?

Normal RAG does approximately this:

```text
Question → Retrieve → Generate
```

This project makes decisions:

```text
Question
 → Route
 → Retrieve
 → Evaluate evidence
 → Choose KB or Web
 → Rewrite if needed
 → Retry
 → Generate grounded answer
```

The system therefore controls **what to do next** based on its current state.

---

## 5. Main Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent workflow | LangGraph | Stateful routing and conditional decisions |
| LLM | OpenAI `gpt-4o-mini` | Routing, grading, rewriting, answer generation |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dimensional embeddings |
| Vector DB | Pinecone | Private enterprise knowledge base |
| External search | Tavily | Fallback when company KB is insufficient |
| API | FastAPI | Backend and REST endpoints |
| Frontend | HTML/CSS/JavaScript | Employee-facing interface |
| Audit | SQLite | Query and document-ingestion audit logging |
| Packaging | Docker | Reproducible deployment |

---

## 6. Project Structure

```text
FDE_Agentic_RAG_IT_Copilot/
│
├── app/
│   ├── api/
│   │   └── routes.py              # Chat, health and ingestion APIs
│   │
│   ├── core/
│   │   ├── config.py              # Environment configuration
│   │   └── logging.py             # Logging configuration
│   │
│   ├── rag/
│   │   ├── state.py               # LangGraph state + structured decisions
│   │   ├── vectorstore.py         # Pinecone + embeddings
│   │   └── workflow.py            # Complete Agentic RAG graph
│   │
│   ├── services/
│   │   ├── audit.py               # SQLite query audit
│   │   └── ingestion.py           # PDF/TXT/MD/DOCX loading + chunking
│   │
│   └── main.py                    # FastAPI application
│
├── data/
│   └── sample_kb/
│       ├── company_it_handbook.md
│       └── service_desk_runbook.md
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── templates/
│   └── index.html
│
├── tests/
│   └── test_ingestion.py
│
├── uploads/
├── .env.example
├── Dockerfile
├── ingest_sample_kb.py
├── requirements.txt
├── run.py
└── README.md
```

---

## 7. Setup

### Step 1 — Use the supported Windows Conda environment

```powershell
conda create -n it-copilot python=3.11
conda activate it-copilot
```

The validated interpreter is:

```text
C:\Users\Nikunj\miniconda3\envs\it-copilot\python.exe
```

Use it explicitly for project commands:

```powershell
$py = "C:\Users\Nikunj\miniconda3\envs\it-copilot\python.exe"
$py -m pip install -r requirements.txt
$py -m pip check
```

### Step 2 — Install dependencies

```powershell
C:\Users\Nikunj\miniconda3\envs\it-copilot\python.exe -m pip install -r requirements.txt
```

### Step 3 — Configure environment

Create a `.env` file in the project root directory with the following variables:

```env
# LLM
OPENAI_API_KEY=replace_with_openai_key
OPENAI_MODEL=gpt-4o-mini

# External web fallback
TAVILY_API_KEY=replace_with_tavily_key

# Vector database
PINECONE_API_KEY=replace_with_pinecone_key
PINECONE_INDEX_NAME=fde-it-support-rag
PINECONE_NAMESPACE=company-it-kb
EMBEDDING_MODEL=text-embedding-3-small

APP_ENV=development
MAX_UPLOAD_BYTES=10485760

# Security
ADMIN_API_KEY=replace_with_long_random_admin_key
```

#### Environment Variable Reference

| Variable | Description | Example | Required |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI key for the LLM and embeddings | `replace_with_openai_key` | Yes |
| `TAVILY_API_KEY` | Tavily key for web fallback | `replace_with_tavily_key` | Yes for web fallback |
| `PINECONE_API_KEY` | Pinecone vector database key | `replace_with_pinecone_key` | Yes |
| `PINECONE_INDEX_NAME` | Pinecone index name | `fde-it-support-rag` | ⚠️ Optional (default: `fde-it-support-rag`) |
| `PINECONE_NAMESPACE` | Pinecone namespace for document isolation | `company-it-kb` | ⚠️ Optional (default: `company-it-kb`) |
| `OPENAI_MODEL` | OpenAI chat model | `gpt-4o-mini` | Optional |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` | Optional |
| `ADMIN_API_KEY` | Secret for `POST /api/ingest` | `replace_with_long_random_admin_key` | Required for uploads |
| `MAX_UPLOAD_BYTES` | Maximum upload size | `10485760` | Optional |
| `APP_ENV` | Application environment | `development` or `production` | Optional |

**⚠️ Important Security Notes:**

- Never commit `.env` to version control. Add it to `.gitignore`.
- Change `ADMIN_API_KEY` to a secure random value in production.
- Use environment variables instead of hardcoding secrets.
- Restrict API key access to only the services that need them.

### Step 4 — Load sample company knowledge

```powershell
C:\Users\Nikunj\miniconda3\envs\it-copilot\python.exe ingest_sample_kb.py
```

This demonstrates the RAG ingestion pipeline:

```text
Company Documents
   ↓
Load Documents
   ↓
Chunk Text
   ↓
OpenAI Embeddings
   ↓
Pinecone Vector Database
```

### Step 5 — Run the application

```powershell
C:\Users\Nikunj\miniconda3\envs\it-copilot\python.exe run.py
```

Open:

```text
http://127.0.0.1:8080
```

FastAPI API docs:

```text
http://127.0.0.1:8080/docs
```

---

## 8. Classroom Demo Scenarios

### Demo A — Private KB Success

Ask:

> **How do I connect to the company VPN from home?**

Expected path:

```text
Router → KB
Private KB Retrieval
KB Grade → GOOD
Generate from Private KB
```

Teaching point:

> Trusted internal company knowledge is preferred. No public web call is necessary.

---

### Demo B — Company Policy Question

Ask:

> **Can IT support ask me to share my MFA code?**

The internal handbook says employees must never share passwords or MFA codes with support personnel.

Expected path:

```text
Router → KB
Private KB Retrieval
KB Grade → GOOD
Private KB Answer
```

Teaching point:

> RAG allows the model to answer using company-specific information it was never trained on.

---

### Demo C — External / Current Information

Ask:

> **What is the latest Microsoft Teams outage guidance?**

Expected path when internal documents do not answer it:

```text
Router → KB
Private KB Retrieval
KB Grade → WEAK
Tavily Search
Web Grade → GOOD
Web Answer
```

Teaching point:

> Agentic RAG can choose another information source rather than blindly answering from irrelevant vectors.

---

### Demo D — Query Rewrite

Ask a deliberately vague question such as:

> **My work communication app is acting strange after the new update. What should I do?**

If both KB and first external retrieval are weak, the workflow can rewrite the query and retry.

Teaching point:

> Retrieval failure does not immediately mean failure. An agent can improve the search query and try again while using a retry guard to avoid loops.

---

### Demo E — Direct Conversation

Ask:

> **Hello!**

Expected path:

```text
Router → DIRECT
Direct Answer
```

Teaching point:

> Not every message should trigger expensive vector search or web search.

---

## 9. Document Upload Demo

The sidebar contains **Add Company Document**.

Use the `ADMIN_API_KEY` from `.env`.

Upload one of:

- `.pdf`
- `.txt`
- `.md`
- `.docx`

The backend performs:

```text
Upload
  ↓
Validate admin key, extension, size, and readable content
  ↓
Hash and duplicate check
  ↓
Recursive Chunking
  ↓
Embeddings
  ↓
Pinecone Indexing and SQLite ingestion audit
  ↓
Immediately Available for Retrieval
```

This is useful in an FDE demonstration because the customer does not want to edit Python every time a new policy is published.

---

## 10. API Endpoints

### `GET /api/health`

Health check.

### `POST /api/chat`

Request:

```json
{
  "question": "How do I reset my company password?"
}
```

Response contains:

```json
{
  "answer": "...",
  "source_used": "private_kb",
  "trace": [
    "Router → KB",
    "Private KB retrieval → 4 chunks",
    "KB evidence grade → GOOD",
    "Answer generation → PRIVATE KB"
  ],
  "citations": []
}
```

### `POST /api/ingest`

Protected by the `X-Admin-Key` request header.

Uploads one PDF, TXT, Markdown, or DOCX document and adds its chunks to Pinecone.

Required header:

```text
X-Admin-Key: <ADMIN_API_KEY>
```

The endpoint rejects invalid keys (`401`), unsupported extensions (`400`), missing files (`422`), empty or unreadable files (`400`), oversized files (`413`), and duplicate content (`409`). Successful uploads use generated storage names and create an ingestion audit record.

## 11. Production-readiness limitations

- The upload limit is enforced in the application; production deployments should also enforce body limits at the reverse proxy.
- The admin-key header is suitable for this internal demo but should be replaced with managed identity, rotation, rate limiting, HTTPS, and authorization scopes.
- Duplicate tracking covers documents uploaded through this endpoint. Existing vectors ingested by older scripts are not automatically backfilled into the local audit ledger.
- Pinecone writes and SQLite audit writes are not one atomic transaction; a failed audit write after a successful vector write needs operational reconciliation.
- Uploaded content should be scanned for malware and policy violations before production indexing.
- Web citations are external information and should be validated before changing company-managed systems.

---

