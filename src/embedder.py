# src/embedder.py
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings # Or HuggingFaceEmbeddings if you want it 100% free
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store(chunks):
    # print("Embedding chunks and creating FAISS index...")
    print("Downloading/Loading HuggingFace model and embedding chunks...")
    # Initialize your embedding model
    # embeddings = OpenAIEmbeddings() # This will use OpenAI's API, which is paid but high-quality
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # This is a free alternative, but you need to have the model downloaded locally')
    
    # Create the vector database
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save it locally so we don't have to re-embed every time
    vector_store.save_local("faiss_index")
    print("Vector store saved locally to 'faiss_index/'")
    
    return vector_store
