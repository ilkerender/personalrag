#!/usr/bin/env python3

import argparse
import json
import os
import random
import time

import yaml
import torch
import nltk
from transformers import BertTokenizer, BertModel
from pymilvus import (
    connections,
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType
)

nltk.download('stopwords')


def load_credentials(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_embedding(text: str, tokenizer: BertTokenizer, model: BertModel):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # Adjust max_length as needed
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Average token embeddings to form a single vector
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    return embedding.tolist()


def connect_milvus(host: str, port: str):
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")


def ensure_collection(collection_name: str):
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' exists. Dropping and recreating it.")
        existing_collection = Collection(collection_name)
        existing_collection.drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=5196),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5196),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, description="Document collection with metadata and embeddings")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection '{collection_name}'.")

    # **CREATE AN INDEX FOR FAST SEARCH**
    index_params = {
        "index_type": "IVF_FLAT",  # or "IVF_PQ" for better performance
        "metric_type": "L2",  # Change to "COSINE" if using cosine similarity
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="content_embedding", index_params=index_params)
    print("Index created on 'content_embedding' field.")

    # **LOAD THE COLLECTION INTO MEMORY**
    collection.load()
    print(f"Collection '{collection_name}' is now loaded and ready for queries.")

    return collection


def load_documents(input_path: str):
    documents = []
    with open(input_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
    print(f"Loaded {len(documents)} documents from {input_path}")
    return documents


def generate_random_documents(num_docs: int = 3):
    sample_names = ["Utility Bill", "Invoice", "Receipt", "Statement"]
    documents = []
    for i in range(num_docs):
        doc = {
            "name": random.choice(sample_names) + f" #{i+1}",
            "content": f"This is the sample content of document #{i+1}. The details include amounts, dates, and various charges.",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "url": f"http://example.com/documents/{i+1}"
        }
        documents.append(doc)
    print(f"Generated {num_docs} random sample documents.")
    return documents


def process_documents(documents, tokenizer, model):
    names = []
    contents = []
    timestamps = []
    urls = []
    embeddings = []
    for doc in documents:
        names.append(doc.get("name", "N/A"))
        contents.append(doc.get("content", ""))
        timestamps.append(doc.get("timestamp", ""))
        urls.append(doc.get("url", ""))
        emb = generate_embedding(doc.get("content", ""), tokenizer, model)
        embeddings.append(emb)
    return names, contents, timestamps, urls, embeddings


def insert_documents(collection: Collection, names, contents, timestamps, urls, embeddings):
    data = [names, contents, timestamps, urls, embeddings]
    insert_result = collection.insert(data)
    print(f"Inserted {len(embeddings)} documents.")
    return insert_result.primary_keys


def save_documents_with_ids(documents, ids, output_path: str):
    if len(documents) != len(ids):
        raise ValueError("The number of documents does not match the number of inserted IDs.")

    with open(output_path, 'w') as f:
        for doc, doc_id in zip(documents, ids):
            doc_with_id = doc.copy()
            doc_with_id["id"] = doc_id
            f.write(json.dumps(doc_with_id) + "\n")
    print(f"Saved {len(documents)} documents with IDs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload documents (with fields name, content, timestamp, url) to Milvus and generate embeddings from 'content'."
    )
    parser.add_argument("--creds", required=True, help="Path to the YAML file containing credentials.")
    parser.add_argument("--collection", default="docs", help="Milvus collection name to use.")
    parser.add_argument("--input", help="Path to the input JSON Lines file. If not provided, random sample documents will be generated.")
    parser.add_argument("--output", default="docs.jsonl", help="Path to output JSON Lines file with inserted document IDs.")
    args = parser.parse_args()

    creds = load_credentials(args.creds)
    connect_milvus(host=creds['milvus']['host'], port=creds['milvus']['port'])

    # Prepare BERT tokenizer and model for generating embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Load or generate documents
    if args.input and os.path.exists(args.input):
        documents = load_documents(args.input)
    else:
        print("No input file provided or file not found; generating random sample documents.")
        documents = generate_random_documents(num_docs=3)

    # Ensure the collection exists (drop/recreate with new schema)
    collection = ensure_collection(args.collection)

    # Process documents: generate embeddings from each document's content
    names, contents, timestamps, urls, embeddings = process_documents(documents, tokenizer, model)

    # Insert documents into Milvus and get the generated IDs
    ids = insert_documents(collection, names, contents, timestamps, urls, embeddings)

    # Save the full document info along with Milvus IDs to the output JSON Lines file
    save_documents_with_ids(documents, ids, args.output)


if __name__ == "__main__":
    main()

