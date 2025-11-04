import os
from src.helper import load_pdf_file, text_split
from src.store_index import get_vector_store, create_rag_pipeline

# Path to your data
DATA_PATH = "Data/"

def main():
    print("🚀 Starting Medical Chatbot...")

    # Step 1: Try loading existing Pinecone index
    try:
        docsearch = get_vector_store()  # no chunks -> load existing index
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("📌 Loaded existing Pinecone index.")
    except Exception as e:
        print(f"⚠️ Could not load index directly ({e}), reloading PDFs...")
        # If fails, create new embeddings from PDFs
        docs = load_pdf_file(DATA_PATH)
        chunks = text_split(docs)
        docsearch = get_vector_store(chunks)
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print("✅ Retriever ready.")

    # Step 2: Create RAG pipeline
    rag_chain = create_rag_pipeline(retriever)
    print("✅ RAG pipeline initialized.")

    # Step 3: Ask a question
    user_question = "What is Acne?"
    response = rag_chain.invoke({"input": user_question})
    print("\nAnswer:", response.get("answer", "⚠️ No answer generated"))
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
