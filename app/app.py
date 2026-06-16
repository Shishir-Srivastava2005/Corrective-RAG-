import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag_engine import process_document, chat_with_agent

app = FastAPI(title="Agentic RAG API")

class ChatRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # 1. Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Hand off to the RAG engine for loading, chunking, and embedding
        chunk_count = process_document(temp_file_path)
        
        return {"message": f"Successfully embedded {chunk_count} chunks from {file.filename}."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        shutil.rmtree(temp_dir)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Hand off the query to the compiled LangGraph application
        response_data = chat_with_agent(request.question)
        return response_data
        
    except ValueError as ve:
        # Catches cases where the user queries before uploading a document
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))