# RL-Style RAG Chatbot API

A GitHub-ready FastAPI project that demonstrates:
- OpenAI integration
- embeddings-based retrieval
- Retrieval-Augmented Generation (RAG)
- SQLite-backed persistence with SQLModel
- a reviewer sub-agent for grounding and citations

## Why this project is strong for an internship

This project is designed to show practical backend and AI engineering skills:
- Python API development with FastAPI
- database-backed system design
- document chunking and ingestion
- embeddings-based retrieval
- multi-step answer generation with a sub-agent
- conversation persistence

## Architecture

1. A document is ingested through `/documents/ingest`
2. The backend chunks the text
3. Each chunk is embedded with OpenAI embeddings
4. Chunks and metadata are stored in the database
5. A user asks a question through `/chat`
6. The backend embeds the question and retrieves the most relevant chunks
7. The primary agent drafts an answer
8. The reviewer sub-agent checks grounding and adds inline citations
9. The final answer and conversation history are stored

## Tech stack

- FastAPI
- OpenAI Python SDK
- SQLModel
- SQLite

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Copy `.env.example` to `.env` and add your key.

Windows PowerShell:

```powershell
setx OPENAI_API_KEY "your_key_here"
```

Mac/Linux:

```bash
export OPENAI_API_KEY="your_key_here"
```

### 4. Run the app

```bash
uvicorn main:app --reload
```

### 5. Open the interactive API docs

Open:

```text
http://127.0.0.1:8000/docs
```

## Example API flow

### Ingest a document

POST `/documents/ingest`

```json
{
  "title": "Sample Contract",
  "text": "This agreement begins on January 1 and terminates after 12 months unless renewed..."
}
```

### Ask a question

POST `/chat`

```json
{
  "document_id": "paste-document-id-here",
  "question": "What is the termination clause?",
  "top_k": 4
}
```

## Endpoints

- `GET /` — basic root route
- `GET /health` — health check
- `GET /documents` — list stored documents
- `POST /documents/ingest` — ingest a document and create embeddings
- `POST /chat` — ask questions over a stored document using RAG
- `GET /conversations/{conversation_id}` — inspect conversation history

## Suggested interview explanation

> I built a FastAPI-based RAG chatbot with persistent storage, OpenAI embeddings for retrieval, and a reviewer sub-agent that checks the draft answer and adds grounded citations. The demo uses SQLite for simplicity, but the design can be upgraded to PostgreSQL with pgvector for production-scale retrieval.

## Suggested next upgrades

- add PDF upload support
- add Postgres + pgvector
- add auth
- add a frontend in Streamlit or React
- add streaming responses
