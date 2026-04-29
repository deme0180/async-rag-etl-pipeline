# ⚙️ Mechanical Engineering Async RAG & ETL Pipeline

An enterprise-grade, asynchronous Retrieval-Augmented Generation (RAG) and ETL pipeline designed to ingest, structure, and extract highly technical mechanical engineering data from unstructured textbook PDFs.

## 🚀 Project Overview

Traditional conversational RAG systems are designed for sequential, human-in-the-loop Q&A. However, when processing hundreds of pages of technical documentation (like thermodynamics and fluid dynamics textbooks) to build a structured database, sequential LLM calls become a massive latency bottleneck.

This project solves that by implementing an **Asynchronous Map-Reduce Extraction Engine**. It transforms a standard Conversational RAG tool into an automated ETL (Extract, Transform, Load) factory, dropping multi-attribute extraction latency from several seconds down to under `800ms`.

## 🏗️ Architecture

The system is divided into two distinct modular pipelines:

### 1. The Knowledge Bank (Ingestion Pipeline)
* **Directory Ingestion:** Utilizes LangChain's `DirectoryLoader` to recursively batch-process entire directories of mechanical engineering PDFs.
* **Semantic Chunking:** Implements `RecursiveCharacterTextSplitter` with heavily tuned overlap parameters to ensure complex engineering formulas and tables are not split across chunks.
* **Vector Store:** Embeds the chunks using HuggingFace's open-source embedding models and stores them locally via **FAISS** for rapid similarity search.

### 2. The Extraction Factory (Async Map-Reduce Engine)
* **Map Phase:** Accepts a target domain (e.g., "Optimal Operating Temperature", "Charge Capacity") and maps dynamic extraction prompts concurrently against the retrieved context blocks.
* **Reduce Phase:** Utilizes Python's `asyncio.gather()` to fire all LLM generation tasks at the exact same millisecond. It collects the independent asynchronous responses and structures them into a clean, machine-readable JSON/Dictionary format.
* **LLM Backend:** Powered by the Groq API (Llama-3.1-8B-Instant) for ultra-low latency inference.

## ⚡ Quick Start

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Setup Environment Variables**
*Create a .env file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

**3. Build the Knowledge Bank**
*Place your engineering PDFs in data/me_textbooks/ and run the ingestion script:
```bash
python ingest.py
```
**4. Run the Async Extraction Engine
```bash
python async_extractor.py
```

# ***🔮 Future Scope of Work (Roadmap)***

While the V1 architecture successfully handles text-based semantic extraction, technical documentation relies heavily on visual data. The planned V2 upgrades include:

* Multimodal OCR Ingestion: Replacing the standard PDF loader with Unstructured.io to detect image bounding boxes, extracting thermodynamic diagrams and machine schematics.

* Vision LLM Integration: Passing extracted diagrams through a Vision Model (e.g., Llama-3.2-Vision) to generate text summaries for the FAISS vector index.

* API Wrapper: Exposing the asyncio Map-Reduce engine via a FastAPI endpoint to allow external microservices to trigger bulk extractions.

* Streamlit UI: Building a front-end dashboard to visualize the extracted engineering dictionaries in real-time tabular formats.
