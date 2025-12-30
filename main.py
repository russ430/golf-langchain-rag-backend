import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware

# LangChain & AI Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = FastAPI()

# 1. CORS Configuration for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Infrastructure Setup
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Use an open-source embedding model that runs locally on your M2
# This turns your text into vectors
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # We'll swap to local OS embeddings in next step

# Initialize ChromaDB
vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# 3. PDF Ingestion Endpoint (for React)
@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: File = UploadFile(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process in background so the UI doesn't hang
    background_tasks.add_task(process_pdf, file_path)
    
    return {"success": True, "fileId": file_id}

def process_pdf(file_path: str):
    # Load and Split the PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # FDE Pro-tip: 1000 char chunks with 200 overlap is standard for research
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Add to Vector DB
    vector_db.add_documents(documents=splits)
    print(f"Successfully vectorized {file_path}")

# 4. ServiceNow Analysis Endpoint
@app.post("/analyze")
async def analyze_swing(payload: dict = Body(...)):
    """
    This is the endpoint ServiceNow will call.
    Payload: {"incident_id": "INC...", "notes": "Big slice..."}
    """
    notes = payload.get("notes", "")
    
    # Setup our "Thinker" Model (Pointed at RunPod)
    llm = ChatOpenAI(
        model="apriel-1.5-15b-thinker",
        openai_api_key="your-runpod-key", 
        openai_api_base="https://YOUR_RUNPOD_ENDPOINT.runpod.net/v1" # This will be our RunPod URL
    )

    # RAG: Search the Vector DB for golf research related to the notes
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # Execute the reasoning
    result = qa_chain.invoke(notes)
    
    return {
        "incident_id": payload.get("incident_id"),
        "analysis": result["result"],
        "status": "completed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)