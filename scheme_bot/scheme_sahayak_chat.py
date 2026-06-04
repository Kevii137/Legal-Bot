"""
scheme_sahayak_chat.py
======================
Standalone chat bot for Indian government schemes.

Requirements:
    pip install pyspark pandas numpy scikit-learn groq gradio

Assumptions:
    - Delta tables exist at:
        workspace.default.gov_schemes
        workspace.default.gov_schemes_categories
    - GROQ_API_KEY is set as environment variable, or pasted in this file.

Usage:
    python scheme_sahayak_chat.py            # terminal chat
    python scheme_sahayak_chat.py --gradio   # web UI (open browser)
"""

import os
import sys
import json
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local parquet loading and FAISS search


# ───────────────────────────────────────────────────────────────────────────
#  Configuration
# ───────────────────────────────────────────────────────────────────────────
SCHEMES_TABLE    = "workspace.default.gov_schemes"
CATEGORIES_TABLE = "workspace.default.gov_schemes_categories"

GROQ_MODEL       = "llama-3.3-70b-versatile"

# ───────────────────────────────────────────────────────────────────────────
#  Prompts (matched to scheme_sahayak.py)
# ───────────────────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────────────────
#  Load data and build TF‑IDF engine
# ───────────────────────────────────────────────────────────────────────────
_global_df_schemes = None

def init_spark():
    """Bypassed. Returns None since Spark is not used locally."""
    return None

def load_data(spark=None):
    """Read local Parquet files into Pandas DataFrames."""
    global _global_df_schemes
    import os
    data_dir = "data"
    schemes_path = os.path.join(data_dir, "gov_schemes.parquet")
    categories_path = os.path.join(data_dir, "gov_schemes_categories.parquet")

    print(f"📖 Loading {schemes_path}…")
    df_schemes = pd.read_parquet(schemes_path)
    # Keep only rows with a meaningful summary
    df_schemes = df_schemes[df_schemes["embed_text"].str.len() >= 50].reset_index(drop=True)

    print(f"📖 Loading {categories_path}…")
    df_categories = pd.read_parquet(categories_path)
    print(f"✅ {len(df_schemes)} schemes, {len(df_categories)} categories loaded.\n")
    _global_df_schemes = df_schemes
    return df_schemes, df_categories

def build_search_engine(df_schemes):
    """Create FAISS index from embed_text for fast retrieval."""
    print("🔧 Building FAISS index on embed_text...")
    from sentence_transformers import SentenceTransformer
    import faiss
    
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print(f"⏳ Embedding {len(df_schemes)} schemes…")
    scheme_texts = df_schemes["embed_text"].tolist()
    scheme_embs  = embedder.encode(
        scheme_texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    dim = scheme_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(scheme_embs)
    print(f"   FAISS index ready. Dimension: {dim}")
    return embedder, index

def tfidf_search(query, vectorizer, tfidf_matrix, k=15):
    """Return top-k schemes ranked by cosine similarity to the query using FAISS."""
    global _global_df_schemes
    dfs = _global_df_schemes if _global_df_schemes is not None else df_schemes
    q_emb = vectorizer.encode(
        [query.strip()], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, indices = tfidf_matrix.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            row = dfs.iloc[int(idx)].to_dict()
            row["score"] = float(score)
            results.append(row)
    return results

# ───────────────────────────────────────────────────────────────────────────
#  Chat bot class
# ───────────────────────────────────────────────────────────────────────────
class SchemeChatBot:
    def __init__(self, df_schemes, df_categories, vectorizer, tfidf_matrix, api_key=None):
        self.df_schemes = df_schemes
        self.df_categories = df_categories
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix

        from groq import Groq
        self.groq = Groq(api_key=api_key) if api_key else Groq()

    def _llm_json(self, system, user, history=None, max_tokens=400):
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    def handle_query(self, query, history=None):
        router = self._llm_json(ROUTER_SYSTEM_PROMPT, query, history=history, max_tokens=120)
        mode = router.get("mode", "unknown")
        print(f"🎯 Mode: {mode} | {router.get('reasoning', '')}")

        if mode == "A_name_lookup":
            return self._mode_a(router, query)
        elif mode == "B_category_search":
            return self._mode_b(router, query)
        elif mode == "C_eligibility_check":
            return self._mode_c(router, query, history)
        else:
            return {"mode": mode, "result": None, "message": "I didn't understand that. Please rephrase."}

    def _mode_a(self, router, query):
        name = router.get("scheme_name") or query
        # Exact match first
        mask = self.df_schemes["scheme_name"].str.lower().str.contains(name.lower(), na=False, regex=False)
        hits = self.df_schemes[mask]
        if not hits.empty:
            schemes = hits.head(3).to_dict(orient="records")
        else:
            schemes = tfidf_search(name, self.vectorizer, self.tfidf_matrix, k=3)
        return {"mode": "A_name_lookup", "schemes": schemes}

    def _mode_b(self, router, query):
        # Simple category routing by keyword matching in category names
        q_lower = query.lower()
        matched_cats = []
        for _, cat in self.df_categories.iterrows():
            cat_name = cat["category_name"].lower()
            if any(kw in cat_name for kw in q_lower.split()):
                matched_cats.append(cat["category_name"])
        if not matched_cats:
            matched_cats = self.df_categories["category_name"].head(2).tolist()

        # FAISS search on all schemes
        all_hits = tfidf_search(query, self.vectorizer, self.tfidf_matrix, k=30)
        cat_hits = [h for h in all_hits if h.get("category") in matched_cats]
        schemes = cat_hits[:5] if len(cat_hits) >= 3 else all_hits[:5]

        return {
            "mode": "B_category_search",
            "categories_searched": matched_cats,
            "schemes": schemes
        }

    def _mode_c(self, router, query, history=None):
        attrs = self._llm_json(ATTRIBUTE_EXTRACTOR_PROMPT, query, history=history, max_tokens=250)
        concepts = attrs.pop("search_concepts", [query])
        print(f"   Profile:  {attrs}")
        print(f"   Concepts: {concepts}")

        seen_ids = set()
        candidates = []
        for concept in concepts[:4]:
            for hit in tfidf_search(concept, self.vectorizer, self.tfidf_matrix, k=5):
                if hit["scheme_id"] not in seen_ids:
                    seen_ids.add(hit["scheme_id"])
                    candidates.append(hit)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:10]

        # Build compact context for LLM
        context = "\n\n".join(
            f"Scheme {i+1}: {h['scheme_name']}\n"
            f"Category: {h.get('category')}\n"
            f"Eligibility: {str(h.get('eligibility', ''))[:400]}\n"
            f"Benefits: {str(h.get('benefits', ''))[:250]}"
            for i, h in enumerate(top)
        )
        user_msg = f"User query: {query}\nProfile: {json.dumps(attrs)}\n{context}"
        verdict = self._llm_json(ELIGIBILITY_RESPONDER_PROMPT, user_msg, history=history, max_tokens=1400)

        return {
            "mode": "C_eligibility_check",
            "extracted_profile": attrs,
            "retrieved_schemes": top,
            "eligibility_result": verdict
        }

# ───────────────────────────────────────────────────────────────────────────
#  Pretty‑print for terminal
# ───────────────────────────────────────────────────────────────────────────
def pretty_print(result):
    mode = result.get("mode", "?")
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    if mode == "A_name_lookup":
        for s in result.get("schemes", []):
            print(f"\n📋 {s['scheme_name']}  [{s.get('category')}]")
            print(f"   Level      : {s.get('level')}")
            print(f"   Eligibility: {s.get('eligibility', '')[:250]}")
            print(f"   Benefits   : {s.get('benefits', '')[:250]}")
            print(f"   Link       : {s.get('official_link')}")

    elif mode == "B_category_search":
        print(f"Categories searched: {result.get('categories_searched')}")
        for s in result.get("schemes", [])[:5]:
            print(f"\n  [{s['score']:.3f}] {s['scheme_name']}  ({s.get('category')})")
            print(f"   {s.get('eligibility', '')[:150]}")

    elif mode == "C_eligibility_check":
        verdict = result.get("eligibility_result", {})
        for s in verdict.get("schemes", []):
            print(f"\n✅ {s['scheme_name']}  → {s['eligibility_verdict']}")
            print(f"   Why      : {s.get('why')}")
            print(f"   Benefits : {s.get('key_benefits')}")
            print(f"   Apply    : {s.get('how_to_apply')}")
            print(f"   Link     : {s.get('link')}")
        if verdict.get("follow_up_question"):
            print(f"\n❓ {verdict['follow_up_question']}")
    print()

# ───────────────────────────────────────────────────────────────────────────
#  Gradio chat interface (optional)
# ───────────────────────────────────────────────────────────────────────────
def create_gradio_ui(bot):
    import gradio as gr

    def respond(message, history_state):
        # history_state is a list of [user_msg, bot_msg] pairs (old Gradio format)
        # Convert to our history format: list of {"role": ..., "content": ...}
        history = []
        for user_msg, bot_msg in history_state:
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": bot_msg})

        result = bot.handle_query(message, history=history)

        # Convert result to a readable string
        output_lines = []
        mode = result.get("mode")
        if mode == "A_name_lookup":
            for s in result["schemes"]:
                output_lines.append(f"**{s['scheme_name']}** [{s.get('category')}]")
                output_lines.append(f"- Level: {s.get('level')}")
                output_lines.append(f"- Eligibility: {s.get('eligibility', '')[:200]}")
                output_lines.append(f"- Benefits: {s.get('benefits', '')[:200]}")
                output_lines.append(f"- Link: {s.get('official_link')}\n")
        elif mode == "B_category_search":
            output_lines.append(f"Categories searched: {', '.join(result['categories_searched'])}")
            for s in result["schemes"][:5]:
                output_lines.append(f"**{s['scheme_name']}** ({s.get('category')}) [score={s['score']:.3f}]")
                output_lines.append(f"- {s.get('eligibility', '')[:150]}")
        elif mode == "C_eligibility_check":
            verdict = result.get("eligibility_result", {})
            for s in verdict.get("schemes", []):
                output_lines.append(f"✅ **{s['scheme_name']}** → {s['eligibility_verdict']}")
                output_lines.append(f"- Why: {s.get('why')}")
                output_lines.append(f"- Benefits: {s.get('key_benefits')}")
                output_lines.append(f"- Apply: {s.get('how_to_apply')}")
                output_lines.append(f"- Link: {s.get('link')}\n")
            if verdict.get("follow_up_question"):
                output_lines.append(f"❓ {verdict['follow_up_question']}")
        else:
            output_lines.append("Sorry, I didn't understand that query.")

        reply = "\n".join(output_lines)
        return reply

    demo = gr.ChatInterface(
        fn=respond,
        title="🌾 Scheme Sahayak — Government Schemes Chat",
        description="Ask about schemes by name, category, or describe your situation.",
        theme="soft"
    )
    demo.launch(share=False)  # set share=True if you want public link

# ───────────────────────────────────────────────────────────────────────────
#  Main entry point
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio web UI instead of terminal chat")
    args = parser.parse_args()

    # API key check
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or not api_key.startswith("gsk_"):
        print("❌ GROQ_API_KEY environment variable not set or invalid.")
        print("   Set it before running: export GROQ_API_KEY='gsk_...'")
        sys.exit(1)

    # Start Spark
    spark = init_spark()
    df_schemes, df_categories = load_data(spark)
    vectorizer, tfidf_matrix = build_search_engine(df_schemes)

    bot = SchemeChatBot(df_schemes, df_categories, vectorizer, tfidf_matrix, api_key=api_key)

    if args.gradio:
        print("🚀 Starting Gradio web interface...")
        create_gradio_ui(bot)
    else:
        print("\n🌾 Scheme Sahayak — Terminal Chat Ready! Describe your situation.")
        print("   Type 'quit' to exit.\n")
        history = []
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            result = bot.handle_query(user_input, history=history)
            pretty_print(result)
            # Update history (simple version)
            history.append({"role": "user", "content": user_input})
            # We don't store the exact assistant string here, but the next turn won't suffer much