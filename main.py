import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from rag import rag_service, answer_question

api = FastAPI(
    title="Ubaid's Retrieval System",
    description="High-precision document vector retrieval system powered by LangChain, OpenAI Embeddings, and Chroma DB",
    version="2.4.0"
)

# CORS configuration
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class Query(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the RAG system")


@api.get("/api/status")
def get_status():
    """Returns the current document indexing status."""
    return rag_service.get_status()


@api.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives an uploaded document, parses text, builds embeddings, and stores in Chroma."""
    try:
        filename = file.filename or "uploaded_document"
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        docs = rag_service.parse_document(contents, filename)
        doc_info = rag_service.process_and_index(docs, filename, file_size=len(contents))
        
        return {
            "success": True,
            "message": f"Successfully indexed '{filename}' into vector store.",
            "document": doc_info
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/load-sample")
def load_sample():
    """Loads the pre-existing sample.PDF file for quick testing."""
    sample_path = os.path.join(BASE_DIR, "sample.PDF")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="sample.PDF file not found in workspace.")
    try:
        doc_info = rag_service.load_pdf_file(sample_path)
        return {
            "success": True,
            "message": "Successfully indexed sample.PDF into vector store.",
            "document": doc_info
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/ask")
def ask(query: Query):
    """Processes question via LangChain RAG pipeline with Chroma vector retriever and Tavily fallback."""
    try:
        response = rag_service.answer_question(query.question)
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/reset")
def reset_document():
    """Clears current document and resets the vector store."""
    rag_service.reset()
    return {"success": True, "message": "Document and vector store reset successfully."}


# Mount static assets directory
api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@api.get("/")
@api.head("/")
def serve_root():
    """Serves the frontend single-page application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({
        "message": "Intelligent Document RAG API is running.",
        "docs_url": "/docs",
        "frontend": "Place index.html in the static directory to view the web UI."
    })