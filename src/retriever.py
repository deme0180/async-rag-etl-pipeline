# src/retriever.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever():
    print("Loading existing FAISS index...")
    
    # We must use the exact same embedding model we used to save it!
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the database from your hard drive
    # Note: allow_dangerous_deserialization=True is required by newer FAISS versions for local files
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    
    # Convert it into a retriever that fetches the top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    return retriever