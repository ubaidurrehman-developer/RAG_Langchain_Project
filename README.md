# Ubaid's Retrieval System

A document vector intelligence and semantic question-answering system built with **FastAPI**, **LangChain**, **OpenAI Embeddings**, and **ChromaDB**, protected by dual-layer **Input and Output Security Guardrails**.

---

## 🚀 Key Features

- **Document Ingestion:** Multi-format parser supporting `PDF`, `DOCX`, `TXT`, `Markdown`, `CSV`, and `JSON`.
- **Vector Pipeline:** Automated text chunking with `RecursiveCharacterTextSplitter` and embeddings generated via `OpenAIEmbeddings`.
- **In-Memory Chroma Vector Store:** Fast similarity search with dynamic collection indexing.
- **🛡️ Input Guardrail (Pre-Retrieval Filter):** Evaluates user prompts *before* executing vector search or synthesis. Blocks jailbreaks, prompt injections, superuser/admin role-claims, and code generation requests.
- **🔒 Output Guardrail (Post-Generation Sanitizer):** Validates generated answers prior to delivery to ensure no code snippets are provided and internal system instructions remain secure.
- **Engineering Console UI:** Clean, solid dark-mode frontend (Black, White, Blue, Red) with live document telemetry, latency tracking, and 1-click response copying.

---

## 🏗️ Architecture & Pipeline Flow

```
User Query
    │
    ▼
[ 1. Input Guardrail ] ──(Unsafe)──► Immediate Blocked Response (0 Chroma queries)
    │
    ▼ (Safe)
[ 2. Chroma Vector Retrieval ] ────► Top-k relevant document chunks retrieved
    │
    ▼
[ 3. RAG Synthesis ] ──────────────► GPT-4o-mini grounded answer generation
    │
    ▼
[ 4. Output Guardrail ] ─(Unsafe)──► Sanitized Fallback Response
    │
    ▼ (Safe)
Client UI Delivery
```

---

## 📦 Tech Stack

- **Backend:** FastAPI, Python 3.10+
- **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small` / OpenAI Embeddings)
- **Framework & Vector DB:** LangChain Core, ChromaDB
- **Frontend:** Vanilla HTML5, CSS3 (No gradients / Clean Dark Mode), JavaScript, Marked.js, Lucide Icons

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment
```bash
cd langchain-master
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run Application Server
```bash
uvicorn main:api --host 0.0.0.0 --port 8000 --reload
```

Open your browser and visit: **`http://localhost:8000`**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/upload` | Ingests and vectorizes an uploaded document file. |
| `POST` | `/api/ask` | Executes guarded RAG question answering pipeline. |
| `GET` | `/api/status` | Returns indexing status and active document telemetry. |
| `POST` | `/api/reset` | Purges Chroma collection and clears active document. |

---

## 📁 Project Structure

```
├── main.py              # FastAPI server routes & static file serving
├── rag.py               # RAG pipeline, Chroma retriever, and Guardrails
├── requirements.txt     # Python project dependencies
├── .env                 # Environment variables (OpenAI API key)
├── static/
│   ├── index.html       # Console user interface
│   ├── style.css        # Solid dark-mode styling (Black, White, Blue, Red)
│   └── app.js           # Client ingestion & query workflow
└── README.md            # Project documentation
```
