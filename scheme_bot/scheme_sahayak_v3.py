"""
scheme_sahayak.py
-----------------
Scheme Sahayak — Government Scheme Eligibility Agent.

Mirrors the ipc_bns_comparator.py framework exactly:
  - Delta tables in Unity Catalog for persistence
  - FAISS in-memory semantic index (built from Delta each session)
  - Groq LLM for routing and response generation
  - Clean class interface: SchemeIngester + SchemeSahayak

Data source: gov_myscheme.csv in Unity Catalog Volume
  /Volumes/workspace/default/raw_files/gov_myscheme.csv

Actual CSV columns (from dataset):
  scheme_name, slug, details, benefits, eligibility,
  application, documents, level, schemeCategory

Usage:
    # ---- INGESTION (run ONCE to build Delta tables) ----
    ingester = SchemeIngester()
    ingester.run_pipeline()

    # ---- EVERY SESSION (load from Delta + build FAISS) ----
    import os
    os.environ["GROQ_API_KEY"] = "gsk_..."

    bot = SchemeSahayak.from_delta()
    result = bot.handle_query("I am a poor farmer, what schemes can I get?")
    SchemeSahayak.pretty_print(result)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

CATALOG          = "workspace"
SCHEMA           = "default"
VOLUME           = "raw_files"
FILE_NAME        = "gov_myscheme.csv"
VOLUME_PATH      = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{FILE_NAME}"

SCHEMES_TABLE    = f"{CATALOG}.{SCHEMA}.gov_schemes"
CATEGORIES_TABLE = f"{CATALOG}.{SCHEMA}.gov_schemes_categories"

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_GROQ_MODEL  = "llama-3.3-70b-versatile"

MIN_TEXT_LEN = 50    # Drop rows whose embed_text is shorter than this
MAX_EMBED_LEN = 2000  # Truncate before encoding (model token limit)

# ---- Actual column names in gov_myscheme.csv ----
COL_NAME        = "scheme_name"
COL_SLUG        = "slug"
COL_DETAILS     = "details"
COL_BENEFITS    = "benefits"
COL_ELIGIBILITY = "eligibility"
COL_APP         = "application"
COL_DOCUMENTS   = "documents"
COL_LEVEL       = "level"         # "Central" / "State"
COL_CATEGORY    = "schemeCategory"

# ============================================================================
# Category keyword router (same pattern as ipc_bns_comparator)
# ============================================================================

CATEGORY_KEYWORDS = {
    "Agriculture & Farming":    ["agri", "farm", "crop", "kisan", "farmer",
                                 "irrigation", "soil", "horticulture", "fisheri"],
    "Education & Scholarships": ["scholar", "education", "school", "student",
                                 "college", "study", "fellowship", "literacy"],
    "Health & Medical":         ["health", "medical", "hospital", "disease",
                                 "insurance", "ayushman", "sanitation", "medicine"],
    "Women & Child Welfare":    ["women", "child", "maternity", "girl", "ladli",
                                 "beti", "widow", "mahila", "anganwadi"],
    "Housing & Shelter":        ["housing", "house", "awas", "shelter",
                                 "toilet", "swachh"],
    "Employment & Skill":       ["employment", "job", "skill", "mudra",
                                 "self-employ", "apprentice", "labour",
                                 "startup", "msme"],
    "Social Welfare & Pension": ["pension", "elderly", "disabled", "welfare",
                                 "senior citizen", "handicap", "divyang"],
    "Financial Assistance":     ["loan", "credit", "subsidy", "grant",
                                 "financial", "bank", "interest", "insurance"],
    "Rural Development":        ["rural", "village", "gram", "panchayat",
                                 "mnrega", "mgnrega", "pmgsy"],
    "Minority & SC/ST Welfare": ["sc", "st", "obc", "minority", "tribal",
                                 "dalit", "schedule", "backward"],
}

# ============================================================================
# Prompts
# ============================================================================

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

# ============================================================================
# Helpers
# ============================================================================

def _safe(val: Any) -> str:
    """Coerce a cell value to a clean string; return '' for nulls."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in {"nan", "none", ""} else s


def _build_embed_text(s: dict) -> str:
    """Build the rich text that gets embedded for a scheme row."""
    parts = [
        f"Scheme: {s.get('scheme_name', '')}",
        f"Category: {s.get('category', '')}",
        f"Level: {s.get('level', '')}",
        f"Eligibility: {s.get('eligibility', '')}",
        f"Details: {s.get('details', '')}",
        f"Benefits: {s.get('benefits', '')}",
        f"Application: {s.get('application', '')}",
    ]
    text = "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())
    return text[:MAX_EMBED_LEN]


def _infer_category(name: str, details: str, csv_category: str) -> str:
    """Use CSV schemeCategory first; fall back to keyword matching."""
    if csv_category and csv_category.lower() not in {"nan", "none", ""}:
        return csv_category.strip()
    combined = f"{name} {details}".lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return cat
    return "General / Other"


# ============================================================================
# SchemeIngester — run ONCE to build Delta tables
# ============================================================================

class SchemeIngester:
    """
    Reads gov_myscheme.csv from a Unity Catalog Volume, cleans it, and writes
    two persistent Delta tables:
      • gov_schemes           — one row per scheme with embed_text
      • gov_schemes_categories — one row per category with member scheme IDs

    Run ONCE. Every subsequent session just calls SchemeSahayak.from_delta().
    """

    def __init__(self, spark_session: Any | None = None) -> None:
        if spark_session:
            self.spark = spark_session
        else:
            from pyspark.sql import SparkSession
            self.spark = SparkSession.builder.getOrCreate()

    # ------------------------------------------------------------------
    def load_raw(self) -> pd.DataFrame:
        print(f"📦 Reading: {VOLUME_PATH}")
        df = (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "false")   # keep everything as string
            .option("multiLine", "true")
            .option("escape", '"')
            .csv(VOLUME_PATH)
            .toPandas()
        )
        df.columns = [c.strip() for c in df.columns]
        print(f"📄 Raw shape : {df.shape}")
        print(f"   Columns   : {list(df.columns)}")
        return df

    # ------------------------------------------------------------------
    def clean(self, df_raw: pd.DataFrame) -> list[dict]:
        schemes, dropped = [], 0
        for idx, row in df_raw.iterrows():
            name     = _safe(row.get(COL_NAME))
            details  = _safe(row.get(COL_DETAILS))
            slug     = _safe(row.get(COL_SLUG))
            category = _infer_category(name, details, _safe(row.get(COL_CATEGORY)))

            s = {
                "scheme_id":    f"SCHEME_{idx:05d}",
                "scheme_name":  name,
                "slug":         slug,
                "details":      details,
                "benefits":     _safe(row.get(COL_BENEFITS)),
                "eligibility":  _safe(row.get(COL_ELIGIBILITY)),
                "application":  _safe(row.get(COL_APP)),
                "documents":    _safe(row.get(COL_DOCUMENTS)),
                "level":        _safe(row.get(COL_LEVEL)),
                "category":     category,
                "official_link": (
                    f"https://www.myscheme.gov.in/schemes/{slug}" if slug else ""
                ),
            }
            s["embed_text"] = _build_embed_text(s)

            if not name or len(s["embed_text"]) < MIN_TEXT_LEN:
                dropped += 1
                continue
            schemes.append(s)

        print(f"✅ Clean: {len(schemes)} schemes ({dropped} noise rows dropped)")
        return schemes

    # ------------------------------------------------------------------
    def _to_delta(self, rows: list[dict], table: str) -> None:
        from pyspark.sql.types import StringType, StructField, StructType
        schema = StructType([StructField(c, StringType(), True) for c in rows[0]])
        (
            self.spark.createDataFrame(rows, schema=schema)
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table)
        )
        print(f"💾 {len(rows)} rows → {table}")

    # ------------------------------------------------------------------
    def run_pipeline(self) -> None:
        df_raw  = self.load_raw()
        schemes = self.clean(df_raw)

        # --- schemes table ---
        scheme_rows = [
            {k: s[k] for k in ["scheme_id", "scheme_name", "slug", "details",
                                "benefits", "eligibility", "application",
                                "documents", "level", "category",
                                "official_link", "embed_text"]}
            for s in schemes
        ]
        self._to_delta(scheme_rows, SCHEMES_TABLE)

        # --- categories table ---
        groups: dict[str, list[str]] = {}
        for s in schemes:
            groups.setdefault(s["category"], []).append(s["scheme_id"])

        cat_rows = [
            {
                "category_id":   re.sub(r"[^a-zA-Z0-9_-]", "_", cat),
                "category_name": cat,
                "scheme_count":  str(len(ids)),
                "scheme_ids":    ",".join(ids),
            }
            for cat, ids in groups.items()
        ]
        self._to_delta(cat_rows, CATEGORIES_TABLE)
        for r in cat_rows:
            print(f"   📂 {r['category_name']} — {r['scheme_count']} schemes")

        print("\n🎉 Ingestion complete.")
        print(f"   {SCHEMES_TABLE}")
        print(f"   {CATEGORIES_TABLE}")


# ============================================================================
# SchemeSahayak — the query agent (mirrors Comparator in ipc_bns_comparator)
# ============================================================================

class SchemeSahayak:
    """
    Government scheme eligibility agent.

    Three query modes (parallels Mode A / B / C of ipc_bns_comparator):
      A_name_lookup       — find a scheme by name
      B_category_search   — semantic search scoped to a category
      C_eligibility_check — extract user profile → RAG → LLM eligibility verdict

    Startup (every session after ingestion):
        bot = SchemeSahayak.from_delta()
        result = bot.handle_query("I am a farmer, what subsidies can I get?")
        SchemeSahayak.pretty_print(result)
    """

    def __init__(
        self,
        df_schemes: pd.DataFrame,
        df_categories: pd.DataFrame,
        embed_model: str = DEFAULT_EMBED_MODEL,
        groq_client: Any | None = None,
    ) -> None:
        self.df_schemes    = df_schemes[
            df_schemes["embed_text"].str.len() >= MIN_TEXT_LEN
        ].reset_index(drop=True)
        self.df_categories = df_categories.reset_index(drop=True)
        self._embed_model_name = embed_model
        self._groq         = groq_client
        self._embedder     = None
        self._index        = None      # FAISS over schemes
        self._cat_index    = None      # FAISS over categories

    # ------------------------------------------------------------------
    # Factory — normal startup path
    # ------------------------------------------------------------------

    @classmethod
    def from_delta(
        cls,
        spark_session: Any | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        groq_client: Any | None = None,
    ) -> "SchemeSahayak":
        if spark_session is None:
            from pyspark.sql import SparkSession
            spark_session = SparkSession.builder.getOrCreate()

        print(f"📖 Loading {SCHEMES_TABLE}…")
        df_schemes = spark_session.table(SCHEMES_TABLE).toPandas()

        print(f"📖 Loading {CATEGORIES_TABLE}…")
        df_categories = spark_session.table(CATEGORIES_TABLE).toPandas()

        print(f"✅ {len(df_schemes)} schemes, {len(df_categories)} categories loaded.")
        return cls(df_schemes, df_categories, embed_model=embed_model,
                   groq_client=groq_client)

    # ------------------------------------------------------------------
    # Lazy FAISS setup  (mirrors _ensure_index in comparator)
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Run: %pip install faiss-cpu sentence-transformers"
            ) from e

        print(f"⏳ Loading embedding model: {self._embed_model_name}…")
        self._embedder = SentenceTransformer(self._embed_model_name)

        # --- Scheme-level index ---
        print(f"⏳ Embedding {len(self.df_schemes)} schemes…")
        scheme_texts = self.df_schemes["embed_text"].tolist()
        scheme_embs  = self._embedder.encode(
            scheme_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        dim = scheme_embs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(scheme_embs)

        # --- Category-level index (mean-pool of member embed_texts) ---
        print("⏳ Building category-level index…")
        cat_vectors = []
        for _, cat_row in self.df_categories.iterrows():
            member_ids   = set(cat_row["scheme_ids"].split(","))
            member_texts = self.df_schemes[
                self.df_schemes["scheme_id"].isin(member_ids)
            ]["embed_text"].tolist() or [cat_row["category_name"]]
            embs = self._embedder.encode(
                member_texts, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
            mean_vec = np.mean(embs, axis=0)
            mean_vec /= max(np.linalg.norm(mean_vec), 1e-9)
            cat_vectors.append(mean_vec)

        cat_matrix = np.vstack(cat_vectors).astype(np.float32)
        self._cat_index = faiss.IndexFlatIP(dim)
        self._cat_index.add(cat_matrix)

        print(f"✅ FAISS ready — {self._index.ntotal} schemes, "
              f"{self._cat_index.ntotal} categories.")

    # ------------------------------------------------------------------
    # Groq helper
    # ------------------------------------------------------------------

    def _ensure_groq(self) -> None:
        if self._groq is None:
            from groq import Groq
            self._groq = Groq()   # reads GROQ_API_KEY from env

    def _llm_json(self, system: str, user: str, max_tokens: int = 400) -> dict:
        self._ensure_groq()
        try:
            resp = self._groq.chat.completions.create(
                model=DEFAULT_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------
    # Core search primitives
    # ------------------------------------------------------------------

    def _semantic_search(self, query: str, k: int = 15) -> list[dict]:
        self._ensure_index()
        q_emb = self._embedder.encode(
            [query.strip()], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores, indices = self._index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                row = self.df_schemes.iloc[int(idx)].to_dict()
                row["score"] = float(score)
                results.append(row)
        return results

    def _route_categories(self, query: str, top_k: int = 2) -> list[str]:
        self._ensure_index()
        q_emb = self._embedder.encode(
            [query.strip()], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        _, indices = self._cat_index.search(q_emb, top_k)
        return [
            self.df_categories.iloc[int(i)]["category_name"]
            for i in indices[0] if i >= 0
        ]

    # ------------------------------------------------------------------
    # Mode A — name lookup
    # ------------------------------------------------------------------

    def mode_a_name_lookup(self, query: str, k: int = 3) -> list[dict]:
        """Substring match on scheme_name; falls back to semantic search."""
        q_lower = query.lower()
        hits = self.df_schemes[
            self.df_schemes["scheme_name"].str.lower().str.contains(
                re.escape(q_lower), na=False, regex=True
            )
        ].head(k).to_dict(orient="records")

        if hits:
            for h in hits:
                h["score"] = 1.0
                h["match_type"] = "name_match"
            return hits

        sem = self._semantic_search(query, k=k)
        for h in sem:
            h["match_type"] = "semantic_fallback"
        return sem[:k]

    # ------------------------------------------------------------------
    # Mode B — category search
    # ------------------------------------------------------------------

    def mode_b_category_search(self, query: str, k: int = 5) -> dict:
        """Stage-1 category routing → Stage-2 filtered semantic retrieval."""
        target_cats = self._route_categories(query, top_k=2)
        print(f"   📂 Categories: {target_cats}")

        all_hits  = self._semantic_search(query, k=k * 4)
        cat_hits  = [h for h in all_hits if h.get("category") in target_cats]
        results   = cat_hits[:k] if len(cat_hits) >= 3 else all_hits[:k]

        return {
            "query":               query,
            "categories_searched": target_cats,
            "schemes":             results,
        }

    # ------------------------------------------------------------------
    # Mode C — eligibility check
    # ------------------------------------------------------------------

    def mode_c_eligibility_check(self, query: str, k: int = 5) -> dict:
        """
        1. LLM extracts user attributes + search concepts
        2. Multi-concept semantic retrieval
        3. LLM eligibility verdict per scheme
        """
        # Step 1: extract profile
        attrs    = self._llm_json(ATTRIBUTE_EXTRACTOR_PROMPT, query, max_tokens=250)
        concepts = attrs.pop("search_concepts", [query])
        print(f"   🔍 Profile  : {attrs}")
        print(f"   💡 Concepts : {concepts}")

        # Step 2: gather candidates across all concepts
        seen, candidates = set(), []
        for concept in concepts[:4]:
            for hit in self._semantic_search(concept, k=k * 2):
                if hit["scheme_id"] not in seen:
                    seen.add(hit["scheme_id"])
                    candidates.append(hit)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:k * 2]

        # Step 3: LLM eligibility verdict
        context = "\n\n---\n\n".join(
            f"Scheme {i+1}: {h['scheme_name']}\n"
            f"Category : {h.get('category', '')}\n"
            f"Level    : {h.get('level', '')}\n"
            f"Eligible : {str(h.get('eligibility', ''))[:400]}\n"
            f"Benefits : {str(h.get('benefits', ''))[:250]}\n"
            f"Apply    : {str(h.get('application', ''))[:200]}\n"
            f"Link     : {h.get('official_link', '')}"
            for i, h in enumerate(top)
        )
        user_msg = (
            f"User query: {query}\n\n"
            f"Extracted profile: {json.dumps(attrs)}\n\n"
            f"Retrieved schemes:\n{context}"
        )
        verdict = self._llm_json(ELIGIBILITY_RESPONDER_PROMPT, user_msg,
                                 max_tokens=1400)

        return {
            "query":              query,
            "extracted_profile":  attrs,
            "search_concepts":    concepts,
            "retrieved_schemes":  top,
            "eligibility_result": verdict,
        }

    # ------------------------------------------------------------------
    # Unified entry point — mirrors Comparator.handle_query()
    # ------------------------------------------------------------------

    def handle_query(self, query: str) -> dict:
        routed = self._llm_json(ROUTER_SYSTEM_PROMPT, query.strip(), max_tokens=120)
        mode   = routed.get("mode", "unknown")
        print(f"\n🎯 Mode: {mode}  |  {routed.get('reasoning', '')}")

        if mode == "A_name_lookup":
            name = routed.get("scheme_name") or query
            return {"mode": mode, "router": routed,
                    "result": self.mode_a_name_lookup(name)}

        if mode == "B_category_search":
            return {"mode": mode, "router": routed,
                    "result": self.mode_b_category_search(query)}

        if mode == "C_eligibility_check":
            return {"mode": mode, "router": routed,
                    "result": self.mode_c_eligibility_check(query)}

        return {"mode": mode, "router": routed, "result": None,
                "note": "Unclassified query — try rephrasing."}

    # ------------------------------------------------------------------
    # Pretty-print  (mirrors the notebook display blocks in the comparator)
    # ------------------------------------------------------------------

    @staticmethod
    def pretty_print(result: dict) -> None:
        mode = result.get("mode", "?")
        print(f"\n{'=' * 65}")
        print(f"Mode : {mode}")
        print(f"{'=' * 65}")

        r = result.get("result")
        if r is None:
            print("No result.")
            return

        if mode == "A_name_lookup":
            for h in (r if isinstance(r, list) else [r]):
                print(f"\n📋 {h.get('scheme_name')}  [{h.get('category')}]")
                print(f"   Level      : {h.get('level')}")
                print(f"   Eligibility: {str(h.get('eligibility', ''))[:250]}")
                print(f"   Benefits   : {str(h.get('benefits', ''))[:250]}")
                print(f"   Link       : {h.get('official_link')}")

        elif mode == "B_category_search":
            print(f"Categories searched: {r.get('categories_searched')}")
            for h in r.get("schemes", [])[:5]:
                print(f"\n  [{h['score']:.3f}] {h['scheme_name']}  ({h['category']})")
                print(f"   Eligibility: {str(h.get('eligibility', ''))[:150]}")

        elif mode == "C_eligibility_check":
            verdict = r.get("eligibility_result", {})
            for s in verdict.get("schemes", []):
                print(f"\n✅ {s.get('scheme_name')}  → {s.get('eligibility_verdict')}")
                print(f"   Why      : {s.get('why')}")
                print(f"   Benefits : {s.get('key_benefits')}")
                print(f"   Apply    : {s.get('how_to_apply')}")
                print(f"   Link     : {s.get('link')}")
            if verdict.get("follow_up_question"):
                print(f"\n❓ {verdict['follow_up_question']}")
