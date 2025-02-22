from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
import openai
from deep_translator import GoogleTranslator
 
app = Flask(__name__)
 
# Azure AI Search Config
SEARCH_ENDPOINT = "https://synergeticstest2025.search.windows.net"
SEARCH_KEY = "oYxpfUslqNhHBKF7JBAtf9XDCx0LNnjnjKDPSPafujAzSeCG9zLl"
INDEX_NAME = "test"
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))
 
# Azure OpenAI Config
openai.api_key = "BXb4w98alOLmc6KMEkTNKzjNyp1bWt42VZMCvGmoPOB1SSkBQ6QMJQQJ99BBACYeBjFXJ3w3AAABACOG1PjP"
 
# Azure Blob Config
BLOB_CONNECTION_STRING = "BXb4w98alOLmc6KMEkTNKzjNyp1bWt42VZMCvGmoPOB1SSkBQ6QMJQQJ99BBACYeBjFXJ3w3AAABACOG1PjP"
 
# Translate Query to English
def translate_to_english(query, source_lang):
    return GoogleTranslator(source=source_lang, target="en").translate(query)
 
# Translate Response to User's Language
def translate_to_user_lang(text, target_lang):
    return GoogleTranslator(source="en", target=target_lang).translate(text)
 
# Generate Embeddings
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]
 
# Search Vector Index
def search_documents(query_embedding):
    results = search_client.search(search_text="", vector=query_embedding, top=3)
    return [doc["content"] for doc in results]
 
# Generate Response from GPT
def generate_response(context, query):
    prompt = f"Context: {context}\n\nUser Query: {query}\n\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
 
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data["query"]
    user_language = data["language"]
 
    # Translate query to English
    translated_query = translate_to_english(user_query, user_language)
 
    # Get query embedding
    query_embedding = get_embedding(translated_query)
 
    # Search documents
    docs = search_documents(query_embedding)
    context = "\n".join(docs) if docs else "No relevant documents found."
 
    # Generate response
    response = generate_response(context, translated_query)
 
    # Translate response back
    final_response = translate_to_user_lang(response, user_language)
 
    return jsonify({"response": final_response})
 
if __name__ == "__main__":
    app.run(debug=True)