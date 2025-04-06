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

# Create or get collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity for embeddings
)