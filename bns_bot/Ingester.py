"""
ingestion.py
------------
BNS (Bharatiya Nyaya Sanhita 2023) ingestion pipeline.

Assumes all sections are stored in a single JSON file as a list of dicts:
    [
        {
            "section_num":   <int>,
            "section_topic": <str>,
            "chapter":       <str>,
            "content":       <str>
        },
        ...
    ]

Workflow:
  1. Embed every section and ingest into the 'bns' ChromaDB collection.
  2. Build the 'bns_chapters' collection using mean-pooled section embeddings
     for the dual-stage chapter routing used at query time.

Run once (or re-run to rebuild the database from scratch).

Dependencies:
    pip install sentence-transformers chromadb numpy
"""

import json
import os

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration — edit these paths before running
# ---------------------------------------------------------------------------
JSON_FILE   = "/Volumes/workspace/default/raw_files/BNS_Full_Sanhita.json"
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"


def load_sections(json_file: str) -> list[dict]:
    """Load the list of section dicts from the single JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        sections = json.load(f)
    print(f"Loaded {len(sections)} sections from '{json_file}'.")
    return sections


# ---------------------------------------------------------------------------
# STEP 1: Ingest sections into ChromaDB (section-level collection)
# ---------------------------------------------------------------------------
def ingest_sections(
    sections: list[dict],
    client: chromadb.ClientAPI,
    embedder: SentenceTransformer,
) -> chromadb.Collection:
    """
    Embed each section's content and store it in the 'bns' ChromaDB
    collection (rebuilt from scratch each run).
    """
    # Drop and recreate for a clean rebuild
    try:
        client.delete_collection("bns")
    except Exception:
        pass
    section_col = client.create_collection("bns", metadata={"hnsw:space": "cosine"})

    print(f"Embedding and indexing {len(sections)} sections…")
    for data in sections:
        vector = embedder.encode(data["content"]).tolist()

        section_col.add(
            ids        = [f"BNS_SEC_{data['section_num']}"],
            embeddings = [vector],
            documents  = [data["content"]],
            metadatas  = [{
                "section_num":   data["section_num"],
                "section_topic": data["section_topic"],
                "chapter":       data["chapter"],
            }],
        )

    print(f"Section collection contains {section_col.count()} vectors.")
    return section_col


# ---------------------------------------------------------------------------
# STEP 2: Build chapter-level collection (mean-pooled section embeddings)
# ---------------------------------------------------------------------------
def build_chapter_collection(
    sections: list[dict],
    client: chromadb.ClientAPI,
    embedder: SentenceTransformer,
) -> chromadb.Collection:
    """
    Group section texts by chapter, compute the mean embedding for each
    chapter, and store those representative vectors in the 'bns_chapters'
    collection. These are used in Stage 1 of dual-stage retrieval to route
    queries to the most relevant chapter before searching sections.
    """
    # Drop and recreate for a clean rebuild
    try:
        client.delete_collection("bns_chapters")
    except Exception:
        pass
    chapter_col = client.create_collection("bns_chapters", metadata={"hnsw:space": "cosine"})

    # Aggregate section texts per chapter
    chapter_groups: dict[str, list[str]] = {}
    for data in sections:
        chapter_groups.setdefault(data["chapter"], []).append(data["content"])

    # Embed all sections per chapter and store the mean vector
    for ch_name, texts in chapter_groups.items():
        embeddings  = embedder.encode(texts)           # shape: (n_sections, dim)
        mean_vector = np.mean(embeddings, axis=0).tolist()

        chapter_col.add(
            ids        = [ch_name],
            embeddings = [mean_vector],
            metadatas  = [{"chapter_name": ch_name}],
        )
        print(f"Stored chapter vector: {ch_name}")

    print(f"Chapter collection contains {chapter_col.count()} vectors.")
    return chapter_col


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading embedding model…")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Ensure the ChromaDB directory exists before initializing the client
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Load all sections once and reuse for both ingestion steps
    sections = load_sections(JSON_FILE)
    ingest_sections(sections, client, embedder)
    build_chapter_collection(sections, client, embedder)

    print("\nIngestion complete. ChromaDB is ready for retrieval.")
