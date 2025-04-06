import os
import json
import time
import sys
import datetime
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings

# ChromaDB configuration
PERSISTENT_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "your_collection_name"

# Initialize ChromaDB client
#Client is main interface for interacting with ChromaDB, it is the manager
client = Client(Settings(
    persist_directory=PERSISTENT_DIRECTORY,
    anonymized_telemetry=False,
    is_persistent=True
))

# Create or get collection that uses cosine similiarity for embeddings
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Set up sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    #Convert text to embeddings and return as numpy array
    embeddings = model.encode(text)
    #ChromaDB expects a list instead of numpy array
    return embeddings.tolist()

#Track Processed Documents

PROCESSED_FILES_PATH = "./processed_files.json"

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "r") as f:
            return json.load(f)
    return {}

def save_processed_files(files):
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(files, f, indent= 2)

# Extract local files from documents directory

def read_local_files(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        #return the content of the file as a string
        return f.read()

# Chunk the text

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    #Split text into chunks with overlap
    chunks = []
    start = 0
    while start < len(text):
        #get the last position of the chunk
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        #Ensure overlap between chunks
        start = end - chunk_overlap
    return chunks


#Process a file

def process_file(file_path):
    file_name = os.path.basename(file_path)

    #1 extract text from file
    content = read_local_files(file_path)

    #2 chunk the text
    chunks = chunk_text(content)

    #3 create embeddings for the chunks and add to ChromaDB
    vector_id = []
    for i, chunk in enumerate(chunks):
        #get vector embeddings for the chunk
        embedding = embed_text(chunk)
        if embedding is None:
            continue

        #add the id of the files' chunk to the list
        vector_id.append(file_name + "_" + str(i))

        metadata = {
            "file_name": file_name,
            "chunk_index": i,
            "text" : chunk [:200] #first 200 characters of the chunk
        }

        #4 add to ChromaDB
        collection.add(
            ids=[vector_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents = [chunk]
        )


