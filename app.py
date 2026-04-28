# app.py
import os
from src.retriever import get_retriever
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# # 1. Set your Groq API Key (Paste your real key here!)
# os.environ["GROQ_API_KEY"] = "gsk_your_actual_api_key_here"
from dotenv import load_dotenv

# Load the hidden keys from the .env file
load_dotenv()

def main():
    print("Loading Retriever...")
    retriever = get_retriever()
    
    print("Loading LLM (Llama 3 via Groq)...")
    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    
    # 2. Create the System Prompt (This tells the AI how to behave)
    system_prompt = (
        "You are a helpful engineering assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer concise, professional, and base it ONLY on the context.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 3. Wire the Retriever and the LLM together into a RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 4. Ask the question
    # question = "What is the definition of fluid pressure?"
    question = "What is the specific weight of water in both US customary and SI units?"
    print(f"\nAsking AI: '{question}'...\n")
    
    response = rag_chain.invoke({"input": question})
    
    # 5. Print the final AI generated answer!
    print("🤖 Final Answer:")
    print(response["answer"])

if __name__ == "__main__":
    main()