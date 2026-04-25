# Databricks notebook source
# MAGIC %md
# MAGIC # Scheme Sahayak — Exploration & Ingestion Notebook
# MAGIC Mirrors `ipc_bns_comparator_explore.ipynb` in structure.
# MAGIC
# MAGIC **Run order:**
# MAGIC 1. Cell 1 — install deps (once per cluster)
# MAGIC 2. Cell 2 — set API key
# MAGIC 3. Cell 3 — explore raw CSV schema
# MAGIC 4. Cell 4 — **ingestion** (ONCE ONLY — builds Delta tables)
# MAGIC 5. Cell 5 — verify Delta tables
# MAGIC 6. Cell 6 — build FAISS + smoke-test all three modes
# MAGIC 7. Cell 7 — interactive chat loop
# MAGIC
# MAGIC > **From the second session onwards**: skip cells 3 and 4, go straight to cells 1 → 2 → 6 → 7.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 1 — Install dependencies

# COMMAND ----------

# MAGIC %pip install sentence-transformers faiss-cpu groq --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer
import numpy as np

# Reuse the exact model you'll always use
MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME)

def compute_embedding(text: str) -> list:
    """Encode one piece of text → list of floats."""
    if not text:
        return []
    # Normalise like your _ensure_index does
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype(np.float32).tolist()

embed_udf = udf(compute_embedding, ArrayType(FloatType()))

SCHEMES_TABLE = "workspace.default.gov_schemes"

print("Reading schemes table...")
df = spark.table(SCHEMES_TABLE)
print(f"Rows: {df.count()}")

# Add the new column
df = df.withColumn("embedding", embed_udf(col("embed_text")))

# Overwrite the table with the new schema
df.write.format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable(SCHEMES_TABLE)

print("✅ Embedding column added and table overwritten.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 2 — API key + imports

# COMMAND ----------

import os, sys

os.environ.setdefault("GROQ_API_KEY", "")   # set GROQ_API_KEY in your environment

# Make scheme_sahayak.py importable — adjust path to wherever you uploaded it
sys.path.insert(0, "/Workspace/Users/da24b007@smail.iitm.ac.in/Legal-Bot/scheme_bot/")
from scheme_sahayak import SchemeIngester, SchemeSahayak, ROUTER_SYSTEM_PROMPT, ATTRIBUTE_EXTRACTOR_PROMPT, ELIGIBILITY_RESPONDER_PROMPT

key = os.environ.get("GROQ_API_KEY", "")
print(f"✓ Key loaded: {key[:8]}... (len={len(key)})" if key.startswith("gsk_") else "✗ Key missing")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 6 — Load agent + smoke test all three modes
# MAGIC ▶️ **Start here every new session** (after cells 1 + 2).
# MAGIC
# MAGIC This builds the FAISS index in memory from the Delta tables — takes ~1-2 min.

# COMMAND ----------

import re, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ── Load tables ────────────────────────────────────────────
SCHEMES_TABLE    = "workspace.default.gov_schemes"
CATEGORIES_TABLE = "workspace.default.gov_schemes_categories"

print("📖 Loading schemes…")
df_schemes = spark.table(SCHEMES_TABLE).toPandas()
# Keep only rows with meaningful summaries
df_schemes = df_schemes[df_schemes["embed_text"].str.len() >= 50].reset_index(drop=True)

print("📖 Loading categories…")
df_categories = spark.table(CATEGORIES_TABLE).toPandas()
print(f"✅ {len(df_schemes)} schemes, {len(df_categories)} categories ready.\n")

# ── TF‑IDF setup (instant, no external vectors) ────────────
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df_schemes["embed_text"].tolist())

def tfidf_search(query: str, k: int = 15) -> list[dict]:
    """Return top‑k schemes by cosine similarity to the query."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = np.argpartition(scores, -k)[-k:]   # efficient top‑k
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # sort descending
    results = []
    for idx in top_idx:
        row = df_schemes.iloc[idx].to_dict()
        row["score"] = float(scores[idx])
        results.append(row)
    return results

# ── QuickSchemeBot (same as before, but uses tfidf_search) ──
class QuickSchemeBot:
    def __init__(self, df_schemes, df_categories):
        self.df_schemes = df_schemes
        self.df_categories = df_categories
        self.groq = Groq()

    def _llm_json(self, system, user, history=None, max_tokens=400):
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        resp = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    # … (handle_query, _format_mode_a, _format_mode_b exactly as before,
    #      but replace any call to keyword_search with tfidf_search) …

    def handle_query(self, query, history=None):
        router = self._llm_json(ROUTER_SYSTEM_PROMPT, query, history=history, max_tokens=120)
        mode = router.get("mode", "unknown")
        print(f"\n🎯 Mode: {mode}  |  {router.get('reasoning', '')}")

        if mode == "A_name_lookup":
            name = router.get("scheme_name") or query
            # exact match first
            hits = self.df_schemes[
                self.df_schemes["scheme_name"].str.lower().str.contains(name.lower(), na=False, regex=False)
            ]
            if hits.empty:
                hits = tfidf_search(name, k=3)
            else:
                hits = hits.head(3).to_dict(orient="records")
            if isinstance(hits, list):
                result_schemes = hits
            else:
                result_schemes = hits
            return {"mode": "A_name_lookup", "router": router,
                    "result": {"query": query, "schemes": result_schemes}}

        elif mode == "B_category_search":
            # … same category routing logic, but use tfidf_search for final ranking
            q_lower = query.lower()
            matched_cats = []
            for _, cat in self.df_categories.iterrows():
                if any(kw in cat["category_name"].lower() for kw in q_lower.split()):
                    matched_cats.append(cat["category_name"])
            if not matched_cats:
                matched_cats = self.df_categories["category_name"].head(2).tolist()
            # filter by category if possible
            filtered = self.df_schemes[self.df_schemes["category"].isin(matched_cats)]
            if len(filtered) < 3:
                filtered = self.df_schemes
            # tfidf search on the (filtered) embed_texts
            # For simplicity, we use the global tfidf_matrix but mask scores
            indices = filtered.index.tolist()
            sub_matrix = tfidf_matrix[indices]
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, sub_matrix).flatten()
            top_k = min(5, len(indices))
            top_local = np.argpartition(scores, -top_k)[-top_k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
            results = []
            for i in top_local:
                row = self.df_schemes.iloc[indices[i]].to_dict()
                row["score"] = float(scores[i])
                results.append(row)
            return {"mode": "B_category_search", "router": router,
                    "result": {"query": query, "categories_searched": matched_cats,
                               "schemes": results}}

        elif mode == "C_eligibility_check":
            attrs = self._llm_json(ATTRIBUTE_EXTRACTOR_PROMPT, query, history=history, max_tokens=250)
            concepts = attrs.pop("search_concepts", [query])
            print(f"   🔍 Profile : {attrs}")
            print(f"   💡 Concepts: {concepts}")
            candidates = []
            seen = set()
            for concept in concepts[:4]:
                for hit in tfidf_search(concept, k=5):
                    if hit["scheme_id"] not in seen:
                        seen.add(hit["scheme_id"])
                        candidates.append(hit)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            top = candidates[:10]
            context = "\n\n".join(
                f"Scheme {i+1}: {h['scheme_name']}\n"
                f"Category: {h.get('category')}\n"
                f"Eligibility: {str(h.get('eligibility', ''))[:400]}\n"
                f"Benefits: {str(h.get('benefits', ''))[:250]}"
                for i, h in enumerate(top)
            )
            verdict = self._llm_json(
                ELIGIBILITY_RESPONDER_PROMPT,
                f"User query: {query}\nProfile: {json.dumps(attrs)}\n{context}",
                history=history, max_tokens=1400
            )
            return {
                "mode": mode, "router": router,
                "result": {
                    "query": query,
                    "extracted_profile": attrs,
                    "search_concepts": concepts,
                    "retrieved_schemes": top,
                    "eligibility_result": verdict,
                }
            }
        else:
            return {"mode": mode, "router": router, "result": None}

# ── Prompts (same as before) ────────────────────────────────
# … copy ROUTER_SYSTEM_PROMPT, ATTRIBUTE_EXTRACTOR_PROMPT, ELIGIBILITY_RESPONDER_PROMPT

# ── Instantiate bot ───────────────────────────────────────
bot = QuickSchemeBot(df_schemes, df_categories)

# Optional smoke tests …

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 6b — Extended Mode B & C tests

# COMMAND ----------

# ---- Mode B: various categories ----
category_queries = [
    "schemes for farmers",
    "housing schemes for poor families",
    "disability pension",
    "skill development for unemployed youth",
    "maternity benefit scheme",
]

print("\n" + "=" * 65)
print("MODE B — Category Search Tests")
print("=" * 65)

for q in category_queries:
    r = bot.handle_query(q)
    mode = r["mode"]
    if mode == "A_name_lookup":
        schemes = r["result"] if isinstance(r["result"], list) else []
        cats = []
    else:
        schemes = r["result"].get("schemes", []) if r["result"] else []
        cats = r["result"].get("categories_searched", []) if r["result"] else []
    top = schemes[0] if schemes else {}
    print(f"\n  [{mode}] {q!r}")
    print(f"  Categories: {cats}")
    print(f"  Top result : {top.get('scheme_name', 'none')}  [{top.get('score', 0):.3f}]")


# ---- Mode C: profile scenarios ----
eligibility_scenarios = [
    "I am a small farmer with less than 2 hectares of land",
    "I am an SC student in class 12, looking for a scholarship",
    "My husband passed away and I have 3 children. We are very poor.",
    "I want to start a small business but don't have money for a loan",
    "I am a tribal person from Jharkhand, what welfare schemes exist for me?",
]

print("\n" + "=" * 65)
print("MODE C — Eligibility Check Tests")
print("=" * 65)

for q in eligibility_scenarios:
    print(f"\n{'=' * 65}")
    print(f">>> {q}")
    r = bot.handle_query(q)
    SchemeSahayak.pretty_print(r)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 7 — Interactive chat loop

# COMMAND ----------

def extract_assistant_text_for_history(result: dict) -> str:
    mode = result.get("mode")
    r = result.get("result")
    if mode == "A_name_lookup":
        schemes = r.get("schemes", [])
        return f"I found {len(schemes)} scheme(s). Ask me for details."
    elif mode == "B_category_search":
        cats = r.get("categories_searched", [])
        count = len(r.get("schemes", []))
        return f"Here are {count} schemes under {cats}."
    elif mode == "C_eligibility_check":
        verdict = r.get("eligibility_result", {})
        schemes = verdict.get("schemes", [])
        if schemes:
            names = [s["scheme_name"] for s in schemes[:3]]
            return f"You may be eligible for: {', '.join(names)}."
        return "I need more information to check eligibility."
    return "I didn't understand that."

# COMMAND ----------

print("🌾 Scheme Sahayak — Ready! Describe your situation.")
print("Type 'quit' to exit.\n")

history = []   # will hold {"role": "user", "content": ...} and {"role": "assistant", "content": ...}

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break

    # Pass history to the agent
    result = bot.handle_query(user_input, history=history)

    # Append user message to history
    history.append({"role": "user", "content": user_input})

    # Pretty print the result
    SchemeSahayak.pretty_print(result)

    # Extract assistant's readable response and append to history
    assistant_reply = extract_assistant_text_for_history(result)  # you need to implement this
    history.append({"role": "assistant", "content": assistant_reply})

    print("-" * 65)