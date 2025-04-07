import sys
from ollama import chat

from app import embed_text, collection

def chat_with_agent(query):
    #embed the query
    query_embedding = embed_text(query)

    #query chromadb with query embedding

    try:
        results = collection.query(query_embeddings=query_embedding, n_results=3)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")


    if not results or not results["documents"]:
        return "No relevant information found."

    documents = results["documents"][0]
    context =  " ".join(documents)
    if not context.strip():
        return "No relevant information found."

    #generate response using ollama
    return ollama_chat(query, context)

def ollama_chat(query, context):
    system_prompt = "You are a retrieval-augmented generation (RAG) model designed to provide accurate and relevant information based on both your internal knowledge and retrieved external sources. When processing a user query, first attempt to find and present the most relevant, factual, and up-to-date information available. If you cannot locate any information that is relevant to the query, simply respond with 'I do not know' rather than attempting to fabricate an answer. Always prioritize accuracy, clarity, and honesty in your responses."
    try:
        response = chat(
            model = "deepseek-r1:1.5b",
            messages = [
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\nContext: {context}"
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return (f"Error generating response: {e}")

def main():
    print("Starting the RAG chatbot!")
    print("Type 'quit' to exit.\n")
    while True:
        user_query = input("Your question: ")
        if user_query.lower() == "quit":
            print("Exiting...")
            break
        response = chat_with_agent(user_query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()





