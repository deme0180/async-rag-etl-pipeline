import asyncio
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant")

# The Fake Document
MOCK_CONTRACT = """
This Non-Disclosure Agreement is governed by the laws of the State of California. 
Either party may terminate this agreement with 30 days written notice. 
"""

# 1. The Simulated AI Call
async def extract_attribute(attribute_name, contract_text):
    print(f"Starting extraction for: {attribute_name}...")
    # INSTRUCTION 1: Write a prompt that gives the AI the contract_text 
    # and asks it to extract the attribute_name.
    prompt = f"Read this contract and extract the {attribute_name}. If it's not in the text, say 'Not Found'.\n\nContract:\n{contract_text}"
    #2. Call the llm asynchronously
    # Inst: Use llm.ainvoke(prompt) instead of sleep
    response = await llm.ainvoke(prompt)
    #3. Return the AI's response
    return f"{attribute_name}: {response.content}"

    # # 2. The Engine (Map-Reduce Phase)
async def main():
    attributes = ["Governing Law", "Termination Clause", "Liability Limit"]
    
    start_time = time.time()
    
    # MAP PHASE: # INSTRUCTION 2: Update the list comprehension to pass the MOCK_CONTRACT to the function
    tasks = [extract_attribute(attr, MOCK_CONTRACT ) for attr in attributes]
    
    # REDUCE PHASE: Fire them all at the exact same time and collect the results
    results = await asyncio.gather(*tasks) # <-- Notice the 'await' here!
    
    end_time = time.time()
    
    print("\nExtraction Results:")
    results_dict = {}
    for result in results:
        print(result)
        # Extract the attribute name and value from the result string
        attr_name, attr_value = result.split(": ", 1)
        results_dict[attr_name] = attr_value

    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
    print(f"Results Dictionary: {results_dict}")

# 3. The Front Door (Kick off the script)
if __name__ == "__main__":
    asyncio.run(main())