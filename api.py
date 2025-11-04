import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.helper import load_pdf_file, text_split
from src.store_index import get_vector_store, create_rag_pipeline

app = FastAPI(title="Medical Chatbot API")

# ---- Enable CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change ["*"] to ["https://your-frontend.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load Data & Initialize RAG Pipeline Once ----
DATA_PATH = "Data/"

print("🚀 Starting Medical Chatbot API...")

try:
    docsearch = get_vector_store()
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("📌 Loaded existing Pinecone index.")
except Exception as e:
    print(f"⚠️ Could not load index ({e}), building from PDFs...")
    docs = load_pdf_file(DATA_PATH)
    chunks = text_split(docs)
    docsearch = get_vector_store(chunks)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

rag_chain = create_rag_pipeline(retriever)
print("✅ RAG pipeline initialized.")

# ---- Request Schema ----
class Query(BaseModel):
    question: str

# ---- API Endpoint ----
@app.post("/chat")
async def chat(query: Query):
    """Accepts a medical query and returns an AI-generated answer."""
    try:
        response = rag_chain.invoke({"input": query.question})
        return {"answer": response.get("answer", "⚠️ Sorry, I couldn’t generate an answer.")}
    except Exception as e:
        return {"error": str(e)}
