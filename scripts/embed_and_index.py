#!/usr/bin/env python3
"""
Embed chunks and index with FAISS for vector search.

Embedding strategy:
- Prepend [FY{year}] [Section] to content for semantic matching
- Section names abbreviated for efficiency
- Metadata stored separately for filtering
"""

import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
CHUNKS_FILE = "sec_corpus/META/chunked/all_chunks.json"
INDEX_DIR = "vector_store"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
# Solid retrieval model - MIT license, stable, well-tested
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Section name abbreviations for embedding prefix
SECTION_ABBREV = {
    "Item 1A. Risk Factors": "Risk Factors",
    "Summary Risk Factors": "Summary Risks",
    "Risks Related to Our Product Offerings": "Product Risks",
    "Risks Related to Our Business Operations and Financial Results": "Business Risks",
    "Risks Related to Business Operations and Financial Results": "Business Risks",
    "Risks Related to Government Regulation and Enforcement": "Regulatory Risks",
    "Risks Related to Data, Security, Platform Integrity, and Intellectual Property": "Data Security Risks",
    "Risks Related to Data, Security, and Intellectual Property": "Data Security Risks",
    "Risks Related to Ownership of Our Class A Common Stock": "Stock Risks",
    "Introduction": "Introduction",
}


def load_chunks() -> list[dict]:
    """Load chunked data."""
    with open(CHUNKS_FILE) as f:
        return json.load(f)


def get_section_abbrev(section: str) -> str:
    """Get abbreviated section name for embedding prefix."""
    return SECTION_ABBREV.get(section, section[:20])


def prepare_documents(chunks: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Prepare documents for embedding.
    Prepend [FY{year}] [Section] to content for better semantic matching.
    """
    documents = []
    metadatas = []

    for chunk in chunks:
        # Create embedding text with year and section prefix
        year = chunk["fiscal_year"]
        section_abbrev = get_section_abbrev(chunk["section"])
        doc_text = f"[FY{year}] [{section_abbrev}] {chunk['content']}"

        documents.append(doc_text)
        metadatas.append({
            "chunk_id": chunk["chunk_id"],
            "company": chunk["company"],
            "fiscal_year": chunk["fiscal_year"],
            "filed_at": chunk["filed_at"],
            "section": chunk["section"],
            "chunk_type": chunk["chunk_type"],
            "content": chunk["content"],
            "token_count": chunk["token_count"],
        })

    return documents, metadatas


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    print("(First run will download the model)")
    start = time.time()

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - start:.1f}s (device: {model.device})")

    print("\nPreparing documents...")
    documents, metadatas = prepare_documents(chunks)

    print(f"\nEmbedding {len(documents)} documents...")
    start = time.time()

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity with inner product
    )

    embed_time = time.time() - start
    print(f"Embedded in {embed_time:.1f}s ({len(documents)/embed_time:.0f} docs/sec)")

    # Create FAISS index
    print(f"\nBuilding FAISS index...")
    start = time.time()

    dimension = embeddings.shape[1]
    # Using IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    print(f"Index built in {time.time() - start:.2f}s")

    # Save index and metadata
    Path(INDEX_DIR).mkdir(exist_ok=True)

    faiss.write_index(index, f"{INDEX_DIR}/{INDEX_FILE}")
    with open(f"{INDEX_DIR}/{METADATA_FILE}", "wb") as f:
        pickle.dump(metadatas, f)

    # Also save the model name for later
    with open(f"{INDEX_DIR}/config.json", "w") as f:
        json.dump({"model_name": MODEL_NAME, "dimension": dimension}, f)

    print(f"\n" + "="*60)
    print("INDEX SUMMARY")
    print("="*60)
    print(f"Documents indexed: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"Storage: {INDEX_DIR}/")

    # Test query
    print(f"\n" + "="*60)
    print("TEST QUERY: 'AI regulation risks'")
    print("="*60)

    query = "AI regulation risks"
    query_embedding = model.encode([query], normalize_embeddings=True)

    distances, indices = index.search(query_embedding.astype(np.float32), k=3)

    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        meta = metadatas[idx]
        print(f"\n[{i+1}] Score: {score:.3f} | {meta['company']} FY{meta['fiscal_year']} | {meta['section']}")
        print(f"    {meta['content'][:150]}...")


if __name__ == "__main__":
    main()
