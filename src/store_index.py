# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from gpt4all import GPT4All
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.runnables import Runnable
# from src.prompt import prompt
# from src.helper import download_hugging_face_embeddings

# # ------------------- Pinecone Setup -------------------
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index_name = "medical-chatbot"

# if index_name not in [index.name for index in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#     print(f"✅ Index '{index_name}' created successfully.")
# else:
#     print(f"⚠️ Index '{index_name}' already exists.")

# embeddings = download_hugging_face_embeddings()

# def get_vector_store(text_chunks=None):
#     """Add new documents if provided, otherwise load existing index."""
#     if text_chunks:
#         return PineconeVectorStore.from_documents(
#             documents=text_chunks,
#             index_name=index_name,
#             embedding=embeddings
#         )
#     else:
#         return PineconeVectorStore.from_existing_index(
#             index_name=index_name,
#             embedding=embeddings
#         )

# # ------------------- GPT4All Runnable -------------------
# class GPT4AllRunnable(Runnable):
#     def __init__(self, model):
#         self.model = model

#     def invoke(self, input, config=None, **kwargs):
#         if isinstance(input, dict) and "input" in input:
#             prompt_value = input["input"]
#         else:
#             prompt_value = input

#         if hasattr(prompt_value, "to_string"):
#             prompt_text = prompt_value.to_string()
#         elif hasattr(prompt_value, "messages"):
#             prompt_text = "\n".join(
#                 [f"{m.type.upper()}: {m.content}" for m in prompt_value.messages]
#             )
#         else:
#             prompt_text = str(prompt_value)

#         with self.model.chat_session() as session:
#             return session.generate(prompt_text)

# # ------------------- Build RAG Pipeline -------------------
# def create_rag_pipeline(retriever):
#     model_path = r"C:\Users\dell\Documents\Experimental Projects\New folder\chatbot\medical-chatbot\models\gpt4all-falcon-q4_0.gguf"
#     llm = GPT4All(model_name=model_path,device="cpu")
#     gpt4all_runnable = GPT4AllRunnable(llm)

#     question_answer_chain = create_stuff_documents_chain(
#         llm=gpt4all_runnable,
#         prompt=prompt
#     )

#     return create_retrieval_chain(
#         retriever=retriever,
#         combine_docs_chain=question_answer_chain
#     )



import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from gpt4all import GPT4All
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable
from src.prompt import prompt
from src.helper import download_hugging_face_embeddings

# ------------------- Pinecone Setup -------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Create index only if it does not exist
existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # Matches sentence-transformers/all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Index '{index_name}' created successfully.")
else:
    print(f"⚠️ Using existing index: '{index_name}'")

embeddings = download_hugging_face_embeddings()

def get_vector_store(text_chunks=None):
    """Add new documents if provided, otherwise load existing index."""
    if text_chunks:
        print("📌 Adding new documents to Pinecone...")
        return PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings
        )
    else:
        print("📌 Loading existing Pinecone index...")
        return PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

# ------------------- GPT4All Runnable -------------------
class GPT4AllRunnable(Runnable):
    def __init__(self, model):
        self.model = model

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, dict) and "input" in input:
            prompt_value = input["input"]
        else:
            prompt_value = input

        if hasattr(prompt_value, "to_string"):
            prompt_text = prompt_value.to_string()
        elif hasattr(prompt_value, "messages"):
            prompt_text = "\n".join(
                [f"{m.type.upper()}: {m.content}" for m in prompt_value.messages]
            )
        else:
            prompt_text = str(prompt_value)

        with self.model.chat_session() as session:
            return session.generate(prompt_text)

# ------------------- Build RAG Pipeline -------------------
# def create_rag_pipeline(retriever):
#     model_path = r"C:\Users\dell\Documents\Experimental Projects\New folder\chatbot\medical-chatbot\models\gpt4all-falcon-q4_0.gguf"

#     # Important: Use `model` not `model_name`, and force CPU
#     llm = GPT4All(model=model_path, device="cpu")
#     gpt4all_runnable = GPT4AllRunnable(llm)

#     question_answer_chain = create_stuff_documents_chain(
#         llm=gpt4all_runnable,
#         prompt=prompt
#     )

#     return create_retrieval_chain(
#         retriever=retriever,
#         combine_docs_chain=question_answer_chain
#     )

def create_rag_pipeline(retriever):
    # model_path = r"C:\Users\dell\Documents\Experimental Projects\New folder\chatbot\medical-chatbot\models\gpt4all-falcon-q4_0.gguf"
    model_path= r"C:\Users\dell\Documents\Experimental Projects\New folder\chatbot\medical-chatbot\models\Llama-3.2-3B-Instruct-IQ3_M.gguf"
    # FIXED: use model_name instead of model
    llm = GPT4All(model_name=model_path, device="cpu", allow_download=False)
    
    gpt4all_runnable = GPT4AllRunnable(llm)

    question_answer_chain = create_stuff_documents_chain(
        llm=gpt4all_runnable,
        prompt=prompt
    )

    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )
