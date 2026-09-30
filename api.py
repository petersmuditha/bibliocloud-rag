import time
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag import create_qa_chain

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# API INITIALIZATION
# ============================================================
app = FastAPI(title="BiblioCloud RAG API", version="1.0.0")

qa_chain = create_qa_chain()

# ============================================================
# LOGGING MIDDLEWARE
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Method: {request.method} | "
        f"Path: {request.url.path} | "
        f"Time: {process_time:.4f}s | "
        f"Status: {response.status_code}"
    )
    return response

# ============================================================
# DATA MODELS
# ============================================================
class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[str]

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "BiblioCloud RAG API is running!"
    }


@app.post("/ask", response_model=Answer)
def ask_question(q: Question):
    logger.info(f"Received question: {q.question}")
    start = time.time()

    result = qa_chain.invoke({"query": q.question})

    elapsed = time.time() - start
    sources = [doc.metadata.get('source', 'unknown') for doc in result['source_documents']]
    logger.info(f"Answered in {elapsed:.2f}s | Sources: {sources}")

    return Answer(
        answer=result['result'],
        sources=sources
    )