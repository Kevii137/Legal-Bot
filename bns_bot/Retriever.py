"""
retrieval.py
------------
BNS (Bharatiya Nyaya Sanhita 2023) retrieval and chat pipeline.

Workflow:
  Stage 1 — Chapter routing:   embed the query and find the most relevant
             chapter(s) using the chapter-level ChromaDB collection.
  Stage 2 — Section retrieval: query only the sections that belong to the
             routed chapter(s), reducing noise and improving precision.
  Stage 3 — Cross-encoder reranking: re-score the candidate sections with
             a cross-encoder to surface the most relevant results.
  Chat    — Pass the top-ranked sections as context to the Groq LLM and
             maintain a multi-turn conversation history.

Dependencies:
    pip install sentence-transformers chromadb groq
    # For reranking:
    pip install sentence-transformers[cross-encoder]
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration — must match the paths used during ingestion
# ---------------------------------------------------------------------------
CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # Set GROQ_API_KEY in your environment
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are an expert AI legal assistant for Indian law. "
    "Answer ONLY from the BNS sections provided. Always cite section numbers. "
    "If the context is insufficient, say so. Do not make up law."
)

# ---------------------------------------------------------------------------
# Load models and connect to ChromaDB
# ---------------------------------------------------------------------------
print("Loading embedding model…")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading reranker model…")
reranker = CrossEncoder(RERANK_MODEL)

print("Connecting to ChromaDB…")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
section_col   = chroma_client.get_collection("bns")
chapter_col   = chroma_client.get_collection("bns_chapters")
print(f"Ready — {section_col.count()} sections loaded.")

# Groq LLM client
llm = Groq(api_key=GROQ_API_KEY)

# Conversation history (grows with every chat turn)
history: list[dict] = []


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k_chapters: int = 2,
    top_k_candidates: int = 10,
    final_n: int = 3,
) -> list[dict]:
    """
    Three-stage retrieval:
      1. Embed the query and find the top-k most relevant chapters.
      2. Query the section collection filtered to those chapters only.
      3. Rerank the candidate sections with a cross-encoder and return
         the top `final_n` results.

    Returns a list of dicts with keys: section_num, section_topic,
    chapter, text, score.
    """
    query_emb = embedder.encode(query).tolist()

    # Stage 1 — Chapter routing
    ch_results = chapter_col.query(
        query_embeddings=[query_emb],
        n_results=top_k_chapters,
    )
    target_chapters = ch_results["ids"][0]
    print(f"Routing to chapters: {target_chapters}")

    # Stage 2 — Candidate section retrieval (filtered by chapter)
    candidates = section_col.query(
        query_embeddings=[query_emb],
        n_results=top_k_candidates,
        where={"chapter": {"$in": target_chapters}},
    )
    docs      = candidates["documents"][0]
    metadatas = candidates["metadatas"][0]

    # Stage 3 — Cross-encoder reranking
    pairs  = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(scores, docs, metadatas),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        {
            "score":         score,
            "text":          doc,
            "section_num":   meta["section_num"],
            "section_topic": meta["section_topic"],
            "chapter":       meta["chapter"],
        }
        for score, doc, meta in reranked[:final_n]
    ]


def build_context(results: list[dict]) -> str:
    """Format retrieved sections into a numbered context block for the LLM."""
    return "\n\n---\n\n".join(
        f"[{i+1}] Section {r['section_num']}: {r['section_topic']}\n"
        f"Chapter: {r['chapter']}\n"
        f"{r['text']}"
        for i, r in enumerate(results)
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def chat(question: str) -> str:
    """
    Retrieve relevant sections for `question`, inject them as context,
    call the Groq LLM, and return the assistant's answer.
    Conversation history is maintained across calls.
    """
    global history

    results = retrieve(question)
    context = build_context(results)

    # System message carries the BNS context for this turn
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nBNS CONTEXT:\n{context}"}]
    messages += history
    messages.append({"role": "user", "content": question})

    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()

    # Append this turn to history for multi-turn continuity
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    return answer


# ---------------------------------------------------------------------------
# Main — simple interactive loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nNyaya Dhwani — BNS Legal Assistant")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        answer = chat(question)
        print(f"\nNyaya Dhwani: {answer}\n")