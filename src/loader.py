# src/loader.py
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def load_and_chunk(file_path: str):
#     print(f"Loading document: {file_path}")
#     loader = PyPDFLoader(file_path)
#     docs = loader.load()
    
#     # We use RecursiveCharacterTextSplitter because it tries to keep paragraphs/sentences together
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, 
#         chunk_overlap=200
#     )
#     chunks = splitter.split_documents(docs)
    
#     print(f"Created {len(chunks)} chunks.")
#     return chunks

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_directory(directory_path):
    print(f"Scanning directory: {directory_path} for ME textbooks...")
    
    # 1. The DirectoryLoader finds every .pdf file in the folder
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    # 2. Load all documents into memory
    raw_documents = loader.load()
    print(f"Successfully loaded {len(raw_documents)} pages of technical documentation.")
    
    # 3. Chunking (Crucial for technical documents so we don't split formulas)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split textbooks into {len(chunks)} processable chunks.")
    
    return chunks