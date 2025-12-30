import os
import uuid
import sqlite3
from typing import List, Union, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware

# REAL AI & INTEGRATION IMPORTS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
### ROLE
Certified Golf Biomechanics Consultant.

### CONSTRAINTS
1. ANSWER ONLY using [CONTEXT FROM RESEARCH].
2. BE CONCISE. Use bullet points. Limit response to under 200 words.
3. If the context doesn't have the answer, state: "Insufficient research data for this fault."

### STRUCTURE
- **Finding**: One sentence summarizing research.
- **Mechanism**: One sentence on biomechanics.
- **Drill**: One actionable 'feel' or drill.
"""

app = FastAPI()

DB_PATH = "metadata.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files 
                 (id TEXT PRIMARY KEY, name TEXT, upload_date TIMESTAMP, status TEXT)''')
    conn.commit()
    conn.close()

init_db()

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. INFRASTRUCTURE SETUP
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# INITIALIZE THE BRAIN (The Vector DB)
# This model runs locally on your M2 and turns text into mathematical vectors
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# This creates/loads the database in the 'chroma_db' folder
vector_db = Chroma(
    collection_name="golf_research",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# 3. THE INGESTION LOGIC (The "Background Task")
def process_pdf(file_path: str, file_id: str):
    try:
        print(f"📖 Starting processing for: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(pages)
        
        vector_db.add_documents(chunks)
        
        # --- ADD THIS SQL UPDATE ---
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE files SET status = 'completed' WHERE id = ?", (file_id,))
        conn.commit()
        conn.close()
        # ---------------------------
        
        print(f"✅ SUCCESSFULLY VECTORIZED {len(chunks)} chunks from {file_path}")
    except Exception as e:
        print(f"❌ ERROR PROCESSING PDF: {str(e)}")
        # Optional: Update status to 'error' if it fails
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE files SET status = 'error' WHERE id = ?", (file_id,))
        conn.commit()
        conn.close()

# 4. PDF UPLOAD ENDPOINT
@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    print(f"🚀 RECEIVED UPLOAD REQUEST: {file.filename}")
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    # Save the file physically to the 'uploads' folder
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ADD TO BACKGROUND TASKS
    # This returns 'Success' to React immediately while the Mac works on the PDF
    background_tasks.add_task(process_pdf, file_path, file_id)

    # Save to SQLite Registry
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO files VALUES (?, ?, datetime('now'), ?)", 
                 (file_id, file.filename, "processing"))
    conn.commit()
    conn.close()
    
    return {
        "success": True, 
        "fileId": file_id,
        "message": f"Successfully received {file.filename}. Processing started."
    }

# 5. SERVICENOW ENDPOINT (for local AI analysis)
@app.post("/analyze")
async def analyze_swing(payload: dict = Body(...)):
    notes = payload.get("notes", "")
    
    # 1. Retrieve the top 3 most relevant chunks from your PDF library
    docs = vector_db.similarity_search(notes, k=5)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 2. Build the final prompt using the System Prompt + Context + Query
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "CONTEXT FROM RESEARCH:\n{context}\n\nSTUDENT PROBLEM: {query}")
    ])

    # 3. Format and Run
    formatted_prompt = prompt_template.format_messages(
        context=context_text,
        query=notes
    )
    
    # Using your local Ollama 3B model
    llm = OllamaLLM(model="llama3.2:3b")
    response = llm.invoke(formatted_prompt)

    return {
        "incident_id": payload.get("incident_id"),
        "analysis": response,
        "status": "completed"
    }

@app.get("/test-search")
async def test_search(query: str):
    print(f"🔍 TESTING SEARCH FOR: {query}")
    
    # This asks ChromaDB to find the 3 most relevant chunks
    results = vector_db.similarity_search(query, k=3)
    
    # We'll return the chunks so you can read them in your browser
    return {
        "query": query,
        "matches": [
            {
                "content": doc.page_content[:300] + "...", # First 300 chars
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            } 
            for doc in results
        ]
    }

@app.get("/files")
async def list_files():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM files ORDER BY upload_date DESC")
    files = [{"id": r[0], "name": r[1], "date": r[2], "status": r[3]} for r in c.fetchall()]
    conn.close()
    return files

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Get the filename from SQLite before we delete the record
    c.execute("SELECT name FROM files WHERE id=?", (file_id,))
    result = c.fetchone()
    
    if not result:
        return {"success": False, "error": "File not found"}
    
    filename = result[0]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    # 2. DELETE FROM CHROMADB
    # We tell Chroma to find every chunk where the 'source' matches this file path
    vector_db.delete(where={"source": file_path})
    print(f"🗑️ Removed vectors for {filename} from ChromaDB")

    # 3. DELETE FROM SQLITE
    c.execute("DELETE FROM files WHERE id=?", (file_id,))
    conn.commit()
    conn.close()

    # 4. (Optional) DELETE PHYSICAL FILE
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"success": True, "message": f"Deleted {filename} and all associated vectors."}

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://0.0.0.0:8000")
    print("Ready to accept PDF uploads and ServiceNow incidents!")
    uvicorn.run(app, host="0.0.0.0", port=8000)