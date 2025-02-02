#!/usr/bin/env python3
import argparse
import yaml
import requests
import streamlit as st
import torch
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from pymilvus import (
    connections,
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType
)

# Set Streamlit page configuration
st.set_page_config(page_title='Your AI Assistant', page_icon='🤖', layout='wide')

# Ensure stopwords are downloaded
nltk.download('stopwords')

def load_credentials(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def remove_stopwords(query: str) -> str:
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in query.split() if word.lower() not in stop_words])

def generate_query_embedding(text: str, tokenizer: BertTokenizer, model: BertModel):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy().tolist()

def connect_milvus(host: str, port: str):
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")

def safe_load_collection(collection: Collection):
    try:
        collection.load()
    except Exception as e:
        if "unknown method GetLoadingProgress" in str(e):
            print("Warning: GetLoadingProgress is not implemented.")
        else:
            raise e

def create_index_if_needed(collection_name: str, field_name: str = "content_embedding"):
    collection = Collection(collection_name)
    if not collection.has_index(field_name):
        collection.create_index(field_name=field_name, index_params={
            "index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}
        })

def milvus_search(collection_name: str, query_embedding, top_k: int = 3):
    collection = Collection(collection_name)
    safe_load_collection(collection)
    results = collection.search(
        data=[query_embedding], anns_field="content_embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k, output_fields=["content"]
    )
    return results

def create_prompt_with_citations(results):
    if not results:
        return "No relevant documents found."
    #context = "\n".join([f"[{i+1}] {hit.entity.get('content', '')}" for i, hit in enumerate(results[0])])
    context = "\n".join([f"[{i+1}] {getattr(hit.entity, 'content', '')}" for i, hit in enumerate(results[0])])

    return f"""
You are an AI assistant that retrieves information from personal documents.
Use only the provided context to answer questions.
Context:\n{context}\n"""

def ask_ollama(base_url: str, system_prompt: str, user_question: str, model: str = "deepseek-r1:1.5b"):
    payload = {"model": model, "messages": [{"role": "user", "content": f"{system_prompt}\n{user_question}"}], "stream": False}
    response = requests.post(f"{base_url}/api/chat", json=payload)
    return response.json().get("message", {}).get("content", "I don't know.")

def streamlit_app(collection_name: str, creds: dict):
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/66/AI_Assistant_Icon.png", width=150)
    st.sidebar.title("🤖 Your AI Assistant")
    st.sidebar.write("This assistant helps you find information in your personal documents.")
    st.sidebar.subheader("Example Queries")
    st.sidebar.markdown("- 🏦 *What's my latest utility bill?*\n- 📄 *Summarize my tax documents.*\n- 🚗 *When was my last car service?*")
    st.sidebar.write("---")
    
    st.title("Your AI Assistant")
    st.subheader("Ask me anything about your documents!")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.write(message["content"])
    
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_question = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching your documents..."):
                query_embedding = generate_query_embedding(user_question, tokenizer, model)
                search_results = milvus_search(collection_name, query_embedding, top_k=3)
                system_prompt = create_prompt_with_citations(search_results)
                answer = ask_ollama(creds['ollama']['base_url'], system_prompt, user_question)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.write(answer)

def main():
    parser = argparse.ArgumentParser(description="AI Assistant for Personal Documents.")
    parser.add_argument("--creds", required=True, help="Path to YAML credentials file.")
    parser.add_argument("--collection", default="bills", help="Milvus collection name (default: bills).")
    args = parser.parse_args()
    
    creds = load_credentials(args.creds)
    connect_milvus(host=creds.get('milvus', {}).get("host", "127.0.0.1"),
                   port=creds.get('milvus', {}).get("port", "19530"))
    
    if not utility.has_collection(args.collection):
        st.error("Collection not found. Please load your documents first.")
        return
    
    streamlit_app(args.collection, creds)

if __name__ == "__main__":
    main()
