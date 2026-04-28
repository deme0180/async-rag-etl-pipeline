# ingest.py
# import os
# from src.loader import load_and_chunk
# from src.embedder import create_vector_store

# def main():
#     # 1. Define the path to your data
#     pdf_path = "data/Hydraulics.pdf"
    
#     # Safety check: does the file actually exist?
#     if not os.path.exists(pdf_path):
#         print(f"Error: Could not find {pdf_path}. Did you put a PDF in the data folder?")
#         return

#     # 2. Run the Loader
#     print("\n--- Phase 1: Loading & Chunking ---")
#     chunks = load_and_chunk(pdf_path)
    
#     # 3. Run the Embedder
#     print("\n--- Phase 2: Embedding & Storing ---")
#     vector_store = create_vector_store(chunks)
    
#     print("\n✅ Pipeline complete! Your document is ready for Q&A.")

# if __name__ == "__main__":
#     main()
import os
from src.loader import load_and_chunk_directory
from src.embedder import create_vector_store

def main():
    # 1. Define the path to your new ME curriculum folder
    dir_path = "data/ME_textbooks"
    
    # Safety check: does the directory actually exist?
    if not os.path.exists(dir_path):
        print(f"Error: Could not find {dir_path}. Please create it and add your PDFs.")
        return

    # 2. Run the Directory Loader
    print("\n--- Phase 1: Loading & Chunking Curriculum ---")
    chunks = load_and_chunk_directory(dir_path)
    
    if not chunks:
        print("No chunks were created. Make sure you have PDFs in the directory!")
        return

    # 3. Run the Embedder
    print("\n--- Phase 2: Embedding & Storing ---")
    # Note: If you are embedding hundreds of pages via an API (like OpenAI/Cohere), 
    # keep an eye on your rate limits! Local embeddings (HuggingFace) are safer for bulk.
    vector_store = create_vector_store(chunks)
    
    print("\n✅ Knowledge Bank Pipeline complete! Your ME database is ready.")

if __name__ == "__main__":
    main()