import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.vector_store import VectorStore, save_chunk_store


def main():
    print("=== Lab 4: Create Vector Database ===")

    embeddings = np.load(config.EMBEDDINGS_FILE)
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Vector Number: {len(embeddings)}, Number of chunks: {len(chunks)}")
    assert len(embeddings) == len(chunks), "Vector Number and chunk Number must be equal"

    store = VectorStore()
    store.build(embeddings)               # หรือ store.build_index(embeddings)
    store.save(config.FAISS_INDEX_FILE)

    save_chunk_store(chunks, config.CHUNK_STORE_FILE)

    print("Vector database created successfully")

if __name__ == "__main__":
    main()