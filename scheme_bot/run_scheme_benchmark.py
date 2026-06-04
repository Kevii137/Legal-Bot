# Databricks notebook source
# MAGIC %md
# MAGIC # Scheme Sahayak — Benchmark Evaluation
# MAGIC
# MAGIC Evaluates your RAG pipeline across 3 dimensions:
# MAGIC 1. **Retrieval** – TF‑IDF recall & MRR
# MAGIC 2. **Mode‑routing** accuracy (A / B / C)
# MAGIC 3. **Eligibility verdict** quality (Mode C)
# MAGIC
# MAGIC Uses a lightweight TF‑IDF + Groq agent – no FAISS, no Sentence‑Transformer downloads.
# MAGIC Requires `scheme_benchmark_questions.json` with ground‑truth labels.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 1 — Install dependencies (run once per cluster)

# COMMAND ----------

# MAGIC %pip install scikit-learn groq --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 2 — Setup (API key + imports)

# COMMAND ----------

import os, sys, json
from datetime import datetime
import numpy as np
import pandas as pd
import re

# ----- CONFIG -----
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")   # set via environment

# Path to benchmark questions
BENCHMARK_JSON = "scheme_benchmark_questions.json"   # adjust if needed

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run ID: {RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 3 — Load tables + build TF‑IDF + QuickSchemeBot

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Load tables
import os
if 'spark' in globals() or 'spark' in locals():
    print("📖 Loading schemes from Spark table…")
    df_schemes = spark.table(SCHEMES_TABLE).toPandas()
    print("📖 Loading categories from Spark table…")
    df_categories = spark.table(CATEGORIES_TABLE).toPandas()
else:
    data_dir = "data"
    schemes_path = os.path.join(data_dir, "gov_schemes.parquet")
    categories_path = os.path.join(data_dir, "gov_schemes_categories.parquet")
    print(f"📖 Loading local schemes from {schemes_path}…")
    df_schemes = pd.read_parquet(schemes_path)
    print(f"📖 Loading local categories from {categories_path}…")
    df_categories = pd.read_parquet(categories_path)

df_schemes = df_schemes[df_schemes["embed_text"].str.len() >= 50].reset_index(drop=True)
print(f"✅ {len(df_schemes)} schemes, {len(df_categories)} categories loaded.\n")

# ----- FAISS Search setup -----
from sentence_transformers import SentenceTransformer
import faiss

print("🔧 Building FAISS index on embed_text...")
vectorizer = SentenceTransformer("BAAI/bge-small-en-v1.5")
print(f"⏳ Embedding {len(df_schemes)} schemes…")
scheme_texts = df_schemes["embed_text"].tolist()
scheme_embs  = vectorizer.encode(
    scheme_texts,
    batch_size=64,
    show_progress_bar=False,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype(np.float32)
dim = scheme_embs.shape[1]
tfidf_matrix = faiss.IndexFlatIP(dim)
tfidf_matrix.add(scheme_embs)
print(f"   FAISS index ready. Dimension: {dim}")

def tfidf_search(query: str, k: int = 15) -> list[dict]:
    """Return top‑k schemes by cosine similarity."""
    q_emb = vectorizer.encode(
        [query.strip()], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, indices = tfidf_matrix.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            row = df_schemes.iloc[int(idx)].to_dict()
            row["score"] = float(score)
            results.append(row)
    return results

# ----- Prompts (same as scheme_sahayak_v3) -----
ROUTER_SYSTEM_PROMPT = """\
You are a query classifier for a Government Scheme Eligibility assistant for rural users in India.

Classify the user query into exactly ONE mode:

- "A_name_lookup": User is asking about a specific scheme by name.
  Examples: "tell me about PM Kisan", "what is Ayushman Bharat"

- "B_category_search": User gave a topic, category, or beneficiary type — no specific scheme name.
  Examples: "schemes for farmers", "education scholarships", "schemes for women"

- "C_eligibility_check": User described their personal situation and wants to know what they qualify for.
  Examples: "I am a farmer with 2 acres of land", "I am a widow with two children"

Respond ONLY with valid JSON (no markdown):
{"mode": "A_name_lookup" | "B_category_search" | "C_eligibility_check",
 "scheme_name": "<extracted scheme name or null>",
 "reasoning": "<one short sentence>"}
"""

ATTRIBUTE_EXTRACTOR_PROMPT = """\
Extract user profile attributes from the query. Return ONLY valid JSON (no markdown):
{
  "occupation": "<farmer | student | woman | disabled | elderly | unemployed | other | null>",
  "caste": "<SC | ST | OBC | General | Minority | null>",
  "gender": "<male | female | other | null>",
  "income_level": "<BPL | low | middle | null>",
  "state": "<state name or null>",
  "age_group": "<child | youth | adult | senior | null>",
  "search_concepts": ["<2-4 policy/welfare terms describing what the user needs>"]
}
"""

ELIGIBILITY_RESPONDER_PROMPT = """\
You are a friendly Government Scheme assistant for rural users in India.
Given the user's query and retrieved schemes, determine eligibility and explain clearly.

RULES:
- Base your answer ONLY on the scheme context provided. Do not invent schemes.
- Keep language simple. Assume low literacy.
- If unsure, ask for more details (income, caste, state, age).
- Respond in the same language the user uses (Hindi or English).

Return ONLY valid JSON (no markdown):
{
  "schemes": [
    {
      "scheme_name": "<name>",
      "eligibility_verdict": "Likely eligible | Possibly eligible | Need more info",
      "why": "<one sentence>",
      "key_benefits": "<2-3 short bullet points as one string>",
      "how_to_apply": "<brief steps>",
      "link": "<official link or null>"
    }
  ],
  "follow_up_question": "<ask for missing details or null>"
}
"""

# ----- QuickSchemeBot (same as working explore notebook) -----
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

    def _semantic_search(self, query, k=15):
        """Adapter so benchmark retrieval works."""
        return tfidf_search(query, k=k)

    def handle_query(self, query, history=None):
        router = self._llm_json(ROUTER_SYSTEM_PROMPT, query, history=history, max_tokens=120)
        mode = router.get("mode", "unknown")
        print(f"\n🎯 Mode: {mode}  |  {router.get('reasoning', '')}")

        if mode == "A_name_lookup":
            name = router.get("scheme_name") or query
            hits = self.df_schemes[
                self.df_schemes["scheme_name"].str.lower().str.contains(name.lower(), na=False, regex=False)
            ]
            if hits.empty:
                hits = tfidf_search(name, k=3)
            else:
                hits = hits.head(3).to_dict(orient="records")
            return {"mode": "A_name_lookup", "router": router,
                    "result": {"query": query, "schemes": hits}}

        elif mode == "B_category_search":
            q_lower = query.lower()
            matched_cats = []
            for _, cat in self.df_categories.iterrows():
                if any(kw in cat["category_name"].lower() for kw in q_lower.split()):
                    matched_cats.append(cat["category_name"])
            if not matched_cats:
                matched_cats = self.df_categories["category_name"].head(2).tolist()
            filtered = self.df_schemes[self.df_schemes["category"].isin(matched_cats)]
            if len(filtered) < 3:
                filtered = self.df_schemes
            # FAISS search on all schemes
            all_hits = tfidf_search(query, k=30)
            cat_hits = [h for h in all_hits if h.get("category") in matched_cats]
            results = cat_hits[:5] if len(cat_hits) >= 3 else all_hits[:5]
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

print("✅ QuickSchemeBot defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 4 — Instantiate the bot + helper mappings

# COMMAND ----------

bot = QuickSchemeBot(df_schemes, df_categories)
print("✅ QuickSchemeBot ready (TF‑IDF + Groq)")

# Build a name‑to‑id lookup for eligibility matching
name_to_id = dict(zip(bot.df_schemes["scheme_name"], bot.df_schemes["scheme_id"]))
print(f"Name→ID mapping: {len(name_to_id)} entries")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 5 — Load benchmark questions

# COMMAND ----------

try:
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
except FileNotFoundError:
    # Create a mini sample if not found
    benchmark = {
        "_meta": {"description": "Sample benchmark – replace with real questions"},
        "questions": [
            {
                "id": "NAME_SAMPLE",
                "mode": "A_name_lookup",
                "language": "en",
                "question": "Tell me about PM Kisan Samman Nidhi",
                "expected_scheme_ids": ["SCHEME_0"]
            },
            {
                "id": "CAT_SAMPLE",
                "mode": "B_category_search",
                "language": "en",
                "question": "Scholarships for SC ST students",
                "expected_categories": ["Education & Scholarships"],
                "expected_scheme_ids": []
            },
            {
                "id": "ELIG_SAMPLE",
                "mode": "C_eligibility_check",
                "language": "en",
                "question": "I am a 60 year old widow from BPL family in Rajasthan",
                "expected_eligible_schemes": [
                    {"scheme_id": "SCHEME_1", "verdict": "Likely eligible"}
                ],
                "expected_not_eligible_schemes": []
            }
        ]
    }
    with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    print("⚠️  Created sample benchmark file – replace with real questions.")

questions = benchmark["questions"]
print(f"Loaded {len(questions)} benchmark questions")
for mode in ["A_name_lookup", "B_category_search", "C_eligibility_check"]:
    cnt = sum(1 for q in questions if q.get("mode") == mode)
    print(f"  {mode}: {cnt}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 6 — Evaluation functions

# COMMAND ----------

def eval_retrieval(bot, questions, k=7):
    """For questions with expected_scheme_ids (or expected_scheme_names), compute recall@k and MRR.
       Expects 'expected_scheme_ids' (list of IDs) or 'expected_scheme_names' (list of names).
       Matches by scheme_name if IDs are unavailable."""
    rows = []
    for q in questions:
        expected_ids = set(q.get("expected_scheme_ids", []))
        expected_names = set(q.get("expected_scheme_names", []))  # new field for names
        if not expected_ids and not expected_names:
            continue

        sem = bot._semantic_search(q["question"], k=k)
        retrieved_names = [hit["scheme_name"] for hit in sem]
        retrieved_ids   = [hit["scheme_id"] for hit in sem]

        # Decide what to match on: names if available, else IDs
        if expected_names:
            retrieved_set = set(retrieved_names[:k])
            relevant_set = expected_names
        else:
            retrieved_set = set(retrieved_ids[:k])
            relevant_set = expected_ids

        # Recall
        recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 1.0

        # MRR (first match)
        mrr = 0.0
        for rank, val in enumerate(retrieved_names if expected_names else retrieved_ids):
            if val in relevant_set:
                mrr = 1.0 / (rank + 1)
                break

        rows.append({
            "question_id": q["id"],
            "recall": recall,
            "mrr": mrr,
            "retrieved": retrieved_names[:k],
            "expected": list(relevant_set),
        })
    df = pd.DataFrame(rows)
    if len(df):
        print(f"\nRetrieval (@{k}):")
        print(f"  Questions: {len(df)}")
        print(f"  Mean Recall: {df['recall'].mean():.3f}")
        print(f"  Mean MRR   : {df['mrr'].mean():.3f}")
    return df


def eval_eligibility(bot, questions):
    """Evaluate eligibility verdict against ground‑truth labels.
       Expects expected_eligible_schemes (list of {scheme_id, verdict}) 
       or expected_eligible_scheme_names (list of {scheme_name, verdict})."""
    rows = []
    for q in questions:
        if q.get("mode") != "C_eligibility_check":
            continue

        result = bot.handle_query(q["question"])["result"]
        verdict = result.get("eligibility_result", {})
        scheme_list = verdict.get("schemes", [])

        # Use names for matching if available
        expected_eligible = q.get("expected_eligible_schemes", [])  # must have "scheme_id"
        expected_eligible_names = q.get("expected_eligible_scheme_names", [])  # new field with "scheme_name"
        expected_not_eligible_ids = {e["scheme_id"] for e in q.get("expected_not_eligible_schemes", [])}
        expected_not_eligible_names = {e["scheme_name"] for e in q.get("expected_not_eligible_scheme_names", [])}

        # Build verdict dict using names (more reliable)
        verdict_by_name = {s.get("scheme_name", ""): s.get("eligibility_verdict", "") for s in scheme_list}

        # Determine which expected set to use (names preferred)
        if expected_eligible_names:
            # Match by name
            likely_eligible_expected = {exp["scheme_name"] for exp in expected_eligible_names
                                        if exp.get("verdict") == "Likely eligible"}
            likely_eligible_bot = {name for name, v in verdict_by_name.items() if v == "Likely eligible"}
            expected_all = {e["scheme_name"] for e in expected_eligible_names}
            fp_set = expected_not_eligible_names
            tp = likely_eligible_expected & likely_eligible_bot
            precision = len(tp) / len(likely_eligible_bot) if likely_eligible_bot else 0
            recall = len(tp) / len(likely_eligible_expected) if likely_eligible_expected else 1.0
            covered = len(expected_all & set(verdict_by_name.keys())) / len(expected_all) if expected_all else 1.0
            false_pos = sum(1 for name in fp_set if verdict_by_name.get(name) == "Likely eligible")
        else:
            # Old ID-based matching
            likely_eligible_expected = {exp["scheme_id"] for exp in expected_eligible
                                        if exp.get("verdict") == "Likely eligible"}
            likely_eligible_bot = set()
            for s in scheme_list:
                sid = name_to_id.get(s.get("scheme_name", ""))
                if sid and s.get("eligibility_verdict") == "Likely eligible":
                    likely_eligible_bot.add(sid)

            tp = likely_eligible_expected & likely_eligible_bot
            precision = len(tp) / len(likely_eligible_bot) if likely_eligible_bot else 0
            recall = len(tp) / len(likely_eligible_expected) if likely_eligible_expected else 1.0

            expected_ids = {e["scheme_id"] for e in expected_eligible}
            covered = len(expected_ids & {name_to_id.get(s["scheme_name"], "") for s in scheme_list}) / len(expected_ids) if expected_ids else 1.0
            false_pos = sum(1 for sid in expected_not_eligible_ids
                            if sid in [name_to_id.get(s["scheme_name"], "") for s in scheme_list
                                       if s.get("eligibility_verdict") == "Likely eligible"])

        rows.append({
            "question_id": q["id"],
            "precision_likely": precision,
            "recall_likely": recall,
            "coverage_eligible": covered,
            "false_positive_likely": false_pos,
            "bot_mentioned_schemes": list(verdict_by_name.keys()),
        })

    df = pd.DataFrame(rows)
    if len(df):
        print(f"\nEligibility Verdict (Likely eligible)")
        print(f"  Questions: {len(df)}")
        print(f"  Mean Precision: {df['precision_likely'].mean():.3f}")
        print(f"  Mean Recall   : {df['recall_likely'].mean():.3f}")
        print(f"  Mean Coverage : {df['coverage_eligible'].mean():.3f}")
        print(f"  False pos (likely): {df['false_positive_likely'].sum()}")
    return df

# COMMAND ----------

def eval_retrieval(bot, questions, k=7):
    """For questions with expected_scheme_ids or expected_scheme_names, compute recall@k and MRR."""
    rows = []
    for q in questions:
        expected_ids = set(q.get("expected_scheme_ids", []))
        expected_names = set(q.get("expected_scheme_names", []))
        if not expected_ids and not expected_names:
            continue

        sem = bot._semantic_search(q["question"], k=k)
        retrieved_names = [hit["scheme_name"] for hit in sem]
        retrieved_ids   = [hit["scheme_id"] for hit in sem]

        # Decide what to match on
        if expected_names:
            retrieved_set = set(retrieved_names[:k])
            relevant_set = expected_names
        else:
            retrieved_set = set(retrieved_ids[:k])
            relevant_set = expected_ids

        # Recall
        recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 1.0

        # MRR (first match)
        mrr = 0.0
        for rank, val in enumerate(retrieved_names if expected_names else retrieved_ids):
            if val in relevant_set:
                mrr = 1.0 / (rank + 1)
                break

        rows.append({
            "question_id": q["id"],
            "mode": q.get("mode", ""),
            "language": q.get("language", "en"),
            "recall": recall,
            "mrr": mrr,
            "retrieved": retrieved_names[:k],
            "expected": list(relevant_set),
        })

    df = pd.DataFrame(rows)
    if len(df):
        print(f"\nRetrieval (@{k}):")
        print(f"  Questions: {len(df)}")
        print(f"  Mean Recall: {df['recall'].mean():.3f}")
        print(f"  Mean MRR   : {df['mrr'].mean():.3f}")
    return df


def eval_routing(bot, questions):
    """Check router mode classification against expected mode."""
    correct = 0
    total = 0
    results = []
    for q in questions:
        routed = bot._llm_json(
            ROUTER_SYSTEM_PROMPT,
            q["question"],
            max_tokens=120
        )
        pred = routed.get("mode", "unknown")
        expected = q.get("mode", "")
        total += 1
        if pred == expected:
            correct += 1
        results.append({
            "question_id": q["id"],
            "expected": expected,
            "predicted": pred,
            "is_correct": pred == expected,
        })

    df = pd.DataFrame(results)
    accuracy = correct / total if total else 0
    print(f"\nRouter Accuracy: {accuracy:.3f} ({correct}/{total})")
    return df


def eval_eligibility(bot, questions):
    """Evaluate eligibility verdict against ground‑truth labels (scheme names)."""
    rows = []
    for q in questions:
        if q.get("mode") != "C_eligibility_check":
            continue

        result = bot.handle_query(q["question"])["result"]
        verdict = result.get("eligibility_result", {})
        scheme_list = verdict.get("schemes", [])

        # Support both ID-based and name-based expected lists
        expected_eligible = q.get("expected_eligible_schemes", [])
        expected_eligible_names = q.get("expected_eligible_scheme_names", [])
        expected_not_eligible_ids = {e["scheme_id"] for e in q.get("expected_not_eligible_schemes", [])}
        expected_not_eligible_names = {e["scheme_name"] for e in q.get("expected_not_eligible_scheme_names", [])}

        # Build verdict dict using names
        verdict_by_name = {s.get("scheme_name", ""): s.get("eligibility_verdict", "") for s in scheme_list}

        if expected_eligible_names:
            # Match by name
            likely_eligible_expected = {exp["scheme_name"] for exp in expected_eligible_names
                                        if exp.get("verdict") == "Likely eligible"}
            likely_eligible_bot = {name for name, v in verdict_by_name.items() if v == "Likely eligible"}
            expected_all = {e["scheme_name"] for e in expected_eligible_names}
            fp_set = expected_not_eligible_names
            tp = likely_eligible_expected & likely_eligible_bot
            precision = len(tp) / len(likely_eligible_bot) if likely_eligible_bot else 0
            recall = len(tp) / len(likely_eligible_expected) if likely_eligible_expected else 1.0
            covered = len(expected_all & set(verdict_by_name.keys())) / len(expected_all) if expected_all else 1.0
            false_pos = sum(1 for name in fp_set if verdict_by_name.get(name) == "Likely eligible")
        else:
            # Fallback to ID matching (less reliable)
            likely_eligible_expected = {exp["scheme_id"] for exp in expected_eligible
                                        if exp.get("verdict") == "Likely eligible"}
            likely_eligible_bot = set()
            for s in scheme_list:
                sid = name_to_id.get(s.get("scheme_name", ""))
                if sid and s.get("eligibility_verdict") == "Likely eligible":
                    likely_eligible_bot.add(sid)

            tp = likely_eligible_expected & likely_eligible_bot
            precision = len(tp) / len(likely_eligible_bot) if likely_eligible_bot else 0
            recall = len(tp) / len(likely_eligible_expected) if likely_eligible_expected else 1.0

            expected_ids = {e["scheme_id"] for e in expected_eligible}
            covered = len(expected_ids & {name_to_id.get(s["scheme_name"], "") for s in scheme_list}) / len(expected_ids) if expected_ids else 1.0
            false_pos = sum(1 for sid in expected_not_eligible_ids
                            if sid in [name_to_id.get(s["scheme_name"], "") for s in scheme_list
                                       if s.get("eligibility_verdict") == "Likely eligible"])

        rows.append({
            "question_id": q["id"],
            "precision_likely": precision,
            "recall_likely": recall,
            "coverage_eligible": covered,
            "false_positive_likely": false_pos,
            "bot_mentioned_schemes": list(verdict_by_name.keys()),
        })

    df = pd.DataFrame(rows)
    if len(df):
        print(f"\nEligibility Verdict (Likely eligible)")
        print(f"  Questions: {len(df)}")
        print(f"  Mean Precision: {df['precision_likely'].mean():.3f}")
        print(f"  Mean Recall   : {df['recall_likely'].mean():.3f}")
        print(f"  Mean Coverage : {df['coverage_eligible'].mean():.3f}")
        print(f"  False pos (likely): {df['false_positive_likely'].sum()}")
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 7 — Run Phase 1: Retrieval

# COMMAND ----------

ret_df = eval_retrieval(bot, questions, k=7)
display(ret_df[["question_id", "recall", "mrr", "retrieved"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 8 — Run Phase 2: Router accuracy

# COMMAND ----------

router_df = eval_routing(bot, questions)
display(router_df[["question_id", "expected", "predicted", "is_correct"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 9 — Run Phase 3: Eligibility verdict

# COMMAND ----------

elig_df = eval_eligibility(bot, questions)
display(elig_df[["question_id", "precision_likely", "recall_likely", "coverage_eligible", "false_positive_likely"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cell 10 — Save results (optional)

# COMMAND ----------

# Combine all metrics into one summary DataFrame
summary = []
if len(ret_df) > 0:
    summary.append({"phase": "retrieval", "metric": "mean_recall@7", "value": ret_df["recall"].mean()})
    summary.append({"phase": "retrieval", "metric": "mean_mrr", "value": ret_df["mrr"].mean()})
if len(router_df) > 0:
    summary.append({"phase": "router", "metric": "accuracy", "value": router_df["is_correct"].mean()})
if len(elig_df) > 0:
    summary.append({"phase": "eligibility", "metric": "precision_likely", "value": elig_df["precision_likely"].mean()})
    summary.append({"phase": "eligibility", "metric": "recall_likely", "value": elig_df["recall_likely"].mean()})
    summary.append({"phase": "eligibility", "metric": "coverage", "value": elig_df["coverage_eligible"].mean()})

summary_df = pd.DataFrame(summary)
summary_df["run_id"] = RUN_ID
display(summary_df)

# Optionally write to Delta or local CSV
try:
    if 'spark' in globals() or 'spark' in locals():
        spark.createDataFrame(summary_df).write.mode("append").saveAsTable("workspace.default.benchmark_summary_scheme")
        print("Saved to workspace.default.benchmark_summary_scheme")
    else:
        summary_path = "data/benchmark_summary_scheme.csv"
        if os.path.exists(summary_path):
            summary_df.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)
        print(f"Saved to local file {summary_path}")
except Exception as e:
    print(f"Delta save failed (non‑fatal): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Done
# MAGIC The benchmark results show:
# MAGIC - **Retrieval quality** (higher recall = the right schemes appear in top‑k).
# MAGIC - **Router correctness** (avoids mixing up name lookups with eligibility checks).
# MAGIC - **Eligibility accuracy** (precision/recall of “Likely eligible” verdicts).
# MAGIC
# MAGIC Use these numbers to tune your prompts, embedding model, or category keywords.

# COMMAND ----------

