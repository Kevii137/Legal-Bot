"""
retrieval_schemes.py
--------------------
Government Scheme Eligibility retrieval and chat pipeline.

Workflow:
  Stage 1 — Category routing:  embed the query and find the most relevant
             scheme category using the category-level ChromaDB collection.
  Stage 2 — Scheme retrieval:  query only the schemes that belong to the
             routed category, reducing noise and improving precision.
  Stage 3 — Cross-encoder reranking: re-score candidate schemes with a
             cross-encoder to surface the most relevant results.
  Chat    — Pass the top-ranked schemes as context to the Groq LLM and
             maintain a multi-turn conversation history.

NOTE: ChromaDB uses EphemeralClient (in-memory) to match the ingestion
pipeline. The ingester MUST be run in the same session before this script.

Dependencies:
    %pip install sentence-transformers chromadb groq
"""

import os
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "chromadb", "groq"])

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import chromadb

# ---------------------------------------------------------------------------
# Configuration — must match ingestion_schemes.py exactly
# ---------------------------------------------------------------------------
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # Lightweight, fast cross-encoder
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")    # Set GROQ_API_KEY in your environment
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a friendly and helpful government scheme eligibility assistant for rural users in India.

Your job is to:
1. Tell the user clearly whether they are likely eligible for a scheme based on the context provided.
2. Explain the key benefits of the scheme in simple, plain language (avoid jargon).
3. Give a brief step-by-step guide on how to apply.
4. Always mention the official link if available.

Rules:
- Answer ONLY from the scheme context provided. Do not invent schemes or benefits.
- If eligibility cannot be determined from the context, say so honestly and ask the user for more details (e.g. income, caste, occupation, state).
- Keep your language simple. Assume the user may have low literacy — use short sentences.
- If multiple schemes are relevant, list each one separately with a clear heading.
- Always cite the Scheme Name when referring to a scheme.
- Respond in the same language the user uses (Hindi or English).
"""

# ---------------------------------------------------------------------------
# Load models and connect to ChromaDB
# ---------------------------------------------------------------------------
print("⏳ Loading embedding model…")
embedder = SentenceTransformer(EMBED_MODEL)

print("⏳ Loading reranker model…")
reranker = CrossEncoder(RERANK_MODEL)

# ⚠️ Must use EphemeralClient — same instance as ingestion session
# If running in a new session, re-run ingestion_schemes.py first
print("🔗 Connecting to in-memory ChromaDB…")
chroma_client = chromadb.EphemeralClient()
scheme_col    = chroma_client.get_collection("gov_schemes")
category_col  = chroma_client.get_collection("gov_schemes_categories")
print(f"✅ Ready — {scheme_col.count()} schemes loaded.")

# Groq LLM client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
llm = Groq(api_key=GROQ_API_KEY)

# Conversation history (grows with every turn for multi-turn continuity)
history: list[dict] = []


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k_categories: int = 2,    # Number of categories to route to (Stage 1)
    top_k_candidates: int = 10,   # Candidate schemes per category (Stage 2)
    final_n: int = 3,             # Final schemes after reranking (Stage 3)
) -> list[dict]:
    """
    Three-stage retrieval:
      1. Embed the query and find the most relevant scheme categories.
      2. Query the scheme collection filtered to those categories only.
      3. Rerank candidates with a cross-encoder and return the top `final_n`.

    Returns a list of dicts with keys:
        scheme_name, eligibility, benefits, application_process,
        official_link, category, text, score
    """
    query_emb = embedder.encode(query).tolist()

    # ------------------------------------------------------------------
    # Stage 1 — Category routing
    # ------------------------------------------------------------------
    cat_results = category_col.query(
        query_embeddings=[query_emb],
        n_results=top_k_categories,
    )
    # IDs in gov_schemes_categories are sanitised category names (e.g. "Agriculture___Farming")
    # Retrieve original category_name from metadata for the WHERE filter
    target_categories = [
        meta["category_name"]
        for meta in cat_results["metadatas"][0]
    ]
    print(f"📂 Routing to categories: {target_categories}")

    # ------------------------------------------------------------------
    # Stage 2 — Candidate scheme retrieval (filtered by category)
    # ------------------------------------------------------------------
    candidates = scheme_col.query(
        query_embeddings=[query_emb],
        n_results=top_k_candidates,
        where={"category": {"$in": target_categories}},
    )
    docs      = candidates["documents"][0]
    metadatas = candidates["metadatas"][0]

    if not docs:
        print("⚠️  No candidates found in routed categories. Falling back to global search.")
        candidates = scheme_col.query(
            query_embeddings=[query_emb],
            n_results=top_k_candidates,
        )
        docs      = candidates["documents"][0]
        metadatas = candidates["metadatas"][0]

    # ------------------------------------------------------------------
    # Stage 3 — Cross-encoder reranking
    # ------------------------------------------------------------------
    pairs  = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(scores, docs, metadatas),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        {
            "score":               float(score),
            "text":                doc,
            "scheme_name":         meta.get("scheme_name", "Unknown Scheme"),
            "eligibility":         meta.get("eligibility", ""),
            "benefits":            meta.get("benefits", ""),
            "application_process": meta.get("application_process", ""),
            "official_link":       meta.get("official_link", ""),
            "category":            meta.get("category", ""),
        }
        for score, doc, meta in reranked[:final_n]
    ]


def build_context(results: list[dict]) -> str:
    """
    Format retrieved schemes into a structured context block for the LLM.
    Eligibility and application steps are foregrounded since those are the
    primary fields needed by a rural eligibility-checker agent.
    """
    blocks = []
    for i, r in enumerate(results):
        block = (
            f"[Scheme {i+1}] {r['scheme_name']}\n"
            f"Category       : {r['category']}\n"
            f"Eligibility    : {r['eligibility']}\n"
            f"Benefits       : {r['benefits']}\n"
            f"How to Apply   : {r['application_process']}\n"
            f"Official Link  : {r['official_link']}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def chat(question: str) -> str:
    """
    Retrieve relevant schemes for `question`, inject them as context,
    call the Groq LLM, and return the assistant's answer.
    Conversation history is maintained across calls for multi-turn support.

    Tip: Encourage users to share their profile details (state, income,
    occupation, caste category) for better eligibility matching.
    """
    global history

    results = retrieve(question)
    context = build_context(results)

    # Rebuild messages each turn: system prompt + context + full history + new question
    messages = [{
        "role":    "system",
        "content": f"{SYSTEM_PROMPT}\n\nSCHEME CONTEXT:\n{context}",
    }]
    messages += history
    messages.append({"role": "user", "content": question})

    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,   # Low temperature — factual eligibility answers
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()

    # Append this turn to history
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    return answer


# ---------------------------------------------------------------------------
# Main — simple interactive loop (replace with your Databricks UI / API call)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🌾 Scheme Sahayak — Government Scheme Eligibility Assistant")
    print("Describe your situation and I'll find the right schemes for you.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        answer = chat(question)
        print(f"\nScheme Sahayak: {answer}\n")