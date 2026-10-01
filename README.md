# 📚 BiblioCloud RAG

A production-grade **Retrieval-Augmented Generation (RAG)** system for BiblioCloud, a digital library platform. The system answers questions about library rules, loans, access policies, and digital resources using an AI assistant that cites its sources.

**🔗 Live Demo**: [https://bibliocloud-rag.onrender.com/docs](https://bibliocloud-rag.onrender.com/docs)

---

## 📖 What it does

BiblioCloud RAG is an AI assistant that helps students, professors, and library staff get instant answers to questions about:

- 📕 **Loan rules** (durations, limits, renewals, fines)
- 🚪 **Access policies** (opening hours, study rooms, quiet areas)
- 💻 **Digital resources** (ebooks, databases, digital loans)

The assistant **never invents answers**. It replies only based on the official documents, and it always **cites the source** of its information.

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Hybrid Search               │
│  ┌─────────────┐   ┌─────────────┐  │
│  │ Vector      │ + │ BM25        │  │
│  │ (Cohere)    │   │ (keywords)  │  │
│  └─────────────┘   └─────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Top 10 chunks  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Groq LLM       │
│  (gpt-oss-20b)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Answer +       │
│  Sources        │
└─────────────────┘
```

### Tech Stack

| Component | Technology |
|---|---|
| **Framework** | LangChain |
| **Vector Store** | FAISS |
| **Embeddings** | Cohere (`embed-english-light-v3.0`) |
| **LLM** | Groq (`openai/gpt-oss-20b`) |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Evaluation** | RAGAS |
| **Deployment** | Render |

---

## ✨ Features

- ✅ **Hybrid Search** — combines vector search (semantic) with BM25 (keyword) for better accuracy on codes and exact terms
- ✅ **Source Citations** — every answer includes the source documents
- ✅ **Anti-Hallucination** — the system says "I don't know" when the answer isn't in the knowledge base
- ✅ **REST API** — fully documented with Swagger UI at `/docs`
- ✅ **Chat UI** — Streamlit interface for interactive conversations
- ✅ **Evaluation** — RAGAS metrics (Faithfulness, Answer Relevancy)
- ✅ **Logging** — every request is logged with method, path, time, and status

---

## 🚀 Installation

### Prerequisites

- Python 3.12.10 or higher
- Git
- A GitHub account
- API keys:
  - [Groq](https://console.groq.com/keys) (free)
  - [Cohere](https://dashboard.cohere.com/api-keys) (free trial)

### Step 1: Clone the repository

```bash
git clone https://github.com/petersmuditha/bibliocloud-rag.git
cd bibliocloud-rag
```

### Step 2: Create a virtual environment

```bash
py -3.12 -m venv venv
venv\Scripts\Activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up environment variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=gsk_your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
```

### Step 5: Run the RAG in local

```bash
python rag.py
```

---

## 💻 Usage

### Option 1: Chat UI (Streamlit)

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Option 2: REST API (FastAPI)

```bash
uvicorn api:app --reload --port 8000
```

Open your browser at `http://localhost:8000/docs`.

### Option 3: Ask a question via API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many books can a student borrow?"}'
```

**Response:**

```json
{
  "answer": "A student can borrow a maximum of 5 books at a time.",
  "sources": ["data/loans.txt"]
}
```

### Option 4: Evaluation with RAGAS

```bash
python valutazione.py
```

This runs the RAG on a test dataset and computes:
- **Faithfulness** — is the answer grounded in the sources?
- **Answer Relevancy** — is the answer relevant to the question?

---

## 📊 Example Questions

Try asking:

- *"How many books can a student borrow?"*
- *"What is the fine for late returns?"*
- *"What are the opening hours on Saturday?"*
- *"Can I renew a reserved book?"*
- *"What is the phone number of the library?"* (should reply "I don't know")

---

## 📁 Project Structure

```
bibliocloud-rag/
│
├── data/                          # Knowledge base
│   ├── loans.txt                  # Loan rules
│   ├── access.txt                 # Access policies
│   └── digital.txt                # Digital resources rules
│
├── rag.py                         # Core RAG logic (hybrid search)
├── api.py                         # FastAPI REST API
├── app.py                         # Streamlit chat UI
├── valutazione.py                 # RAGAS evaluation script
│
├── requirements.txt               # Python dependencies
├── render.yaml                    # Render deployment config
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## ☁️ Deployment on Render

This project is deployed on **Render** (free tier).

### How it works

1. Code is pushed to GitHub
2. Render auto-detects the new commit
3. Render builds the app with `pip install -r requirements.txt`
4. Render starts the app with `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Environment Variables

Set these on Render:

| Key | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |
| `COHERE_API_KEY` | Your Cohere API key |
| `PYTHON_VERSION` | `3.12.10` |

### Live URL

👉 [https://bibliocloud-rag.onrender.com/docs](https://bibliocloud-rag.onrender.com/docs)

**Note**: The free tier of Render spins down after 15 minutes of inactivity. The first request after a pause may take 30-60 seconds.

---

## 📈 Evaluation Results

The system was evaluated using RAGAS on 5 test questions:

| Metric | Score |
|---|---|
| **Faithfulness** | 0.80 |
| **Answer Relevancy** | 0.74 |

These scores indicate a production-ready system with strong grounding in source documents.

---

## 🛠️ Technical Challenges Solved

- ✅ **Memory constraints on Render** — replaced `sentence-transformers` (requires `torch`, 500+ MB) with Cohere API for embeddings
- ✅ **API key management** — used environment variables instead of hardcoding
- ✅ **GitHub size limits** — excluded `venv/` from git with `.gitignore`
- ✅ **RAGAS compatibility** — pinned `ragas==0.2.15` to avoid breaking changes
- ✅ **Hybrid search** — combined BM25 + vector search for better accuracy

---

## 🎓 What I Learned

This project was built as part of a hands-on AI Engineering learning journey. Key takeaways:

1. **RAG architecture** — chunking, embeddings, vector stores, retrieval
2. **Production concerns** — memory, latency, API keys, logging
3. **Evaluation** — measuring RAG quality objectively with RAGAS
4. **Cloud deployment** — deploying Python apps on Render
5. **Version management** — pinning dependencies to avoid breakage

---

## 📄 License

This project is for educational purposes. Feel free to use it as a reference.

---

## 👤 Author

**Muditha Anuruddha Peters**
- GitHub: [@petersmuditha](https://github.com/petersmuditha)

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) — RAG framework
- [Groq](https://groq.com/) — fast LLM inference
- [Cohere](https://cohere.com/) — embeddings API
- [Render](https://render.com/) — free hosting
- [RAGAS](https://docs.ragas.io/) — RAG evaluation
