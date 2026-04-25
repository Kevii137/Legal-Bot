"""
scheme_sahayak.py
-----------------
Scheme Sahayak — Unified Government Scheme Eligibility Pipeline.

This module provides both the ingestion pipeline (for processing, embedding,
and indexing scheme data into Databricks Vector Search) and the retrieval/chat
pipeline (for semantic search and LLM interaction) in a single file.

PERSISTENCE: All vector data is stored in Databricks Vector Search (backed by
Delta tables in Unity Catalog). No re-ingestion is needed between sessions —
just run SchemeRetriever directly once the index is built.

Usage:
    from scheme_sahayak import SchemeIngester, SchemeRetriever

    # Run ONCE to build the persistent index
    ingester = SchemeIngester()
    ingester.run_pipeline()

    # Every subsequent session — just run this
    import os
    os.environ["GROQ_API_KEY"] = "your-api-key"
    bot = SchemeRetriever()
    print(bot.chat("I am a farmer looking for a tractor loan"))
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# ============================================================================
# Constants & Prompts
# ============================================================================

DEFAULT_EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM_MODEL    = "llama-3.3-70b-versatile"

CATEGORY_KEYWORDS = {
    "Agriculture & Farming": [
        "agri", "farm", "crop", "kisan", "farmer",
        "irrigation", "soil", "horticulture",
    ],
    "Education & Scholarships": [
        "scholar", "education", "school", "student",
        "college", "study", "fellowship",
    ],
    "Health & Medical": [
        "health", "medical", "hospital", "disease",
        "insurance", "ayushman", "sanitation",
    ],
    "Women & Child Welfare": [
        "women", "child", "maternity", "girl",
        "ladli", "beti", "widow", "mahila",
    ],
    "Housing & Shelter": [
        "housing", "house", "awas", "shelter", "pradhan mantri awas",
    ],
    "Employment & Skill": [
        "employment", "job", "skill", "mudra",
        "self-employ", "apprentice", "labour",
    ],
    "Social Welfare & Pension": [
        "pension", "elderly", "disabled", "welfare",
        "senior citizen", "handicap",
    ],
    "Financial Assistance": [
        "loan", "credit", "subsidy", "grant",
        "financial", "bank", "interest",
    ],
    "Rural Development": [
        "rural", "village", "gram", "panchayat", "mnrega", "mgnrega",
    ],
    "Minority & SC/ST Welfare": [
        "sc", "st", "obc", "minority", "tribal", "dalit", "schedule",
    ],
}

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

# ============================================================================
# Configurations
# ============================================================================


@dataclass
class IngestionConfig:
    catalog:          str = "workspace"
    schema:           str = "default"
    volume:           str = "raw_files"
    file_name:        str = "gov_myscheme.csv"
    vs_endpoint_name: str = "scheme_bot_endpoint"
    embed_model:      str = DEFAULT_EMBED_MODEL

    # Column mappings
    col_name:        str = "Scheme Name"
    col_desc:        str = "Description"
    col_eligibility: str = "Eligibility Criteria"
    col_benefits:    str = "Benefits"
    col_app_process: str = "Application Process"
    col_link:        str = "Official Link"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/{self.file_name}"

    @property
    def schemes_table(self) -> str:
        return f"{self.catalog}.{self.schema}.gov_schemes_embeddings"

    @property
    def categories_table(self) -> str:
        return f"{self.catalog}.{self.schema}.gov_schemes_categories_embeddings"

    @property
    def schemes_index(self) -> str:
        return f"{self.catalog}.{self.schema}.gov_schemes_index"

    @property
    def categories_index(self) -> str:
        return f"{self.catalog}.{self.schema}.gov_schemes_categories_index"


@dataclass
class RetrievalConfig:
    # Must match IngestionConfig values exactly
    catalog:          str = "workspace"
    schema:           str = "default"
    vs_endpoint_name: str = "scheme_bot_endpoint"
    embed_model:      str = DEFAULT_EMBED_MODEL
    rerank_model:     str = DEFAULT_RERANK_MODEL
    llm_model:        str = DEFAULT_LLM_MODEL
    max_tokens:       int = 1024
    temperature:     float = 0.2

    @property
    def schemes_index(self) -> str:
        return f"{self.catalog}.{self.schema}.gov_schemes_index"

    @property
    def categories_index(self) -> str:
        return f"{self.catalog}.{self.schema}.gov_schemes_categories_index"


# ============================================================================
# Ingester: Unity Catalog CSV -> Delta Table -> Databricks Vector Search
# ============================================================================


class SchemeIngester:
    """
    End-to-end ingestion pipeline.
    Run ONCE to build the persistent Databricks Vector Search index.
    After this completes, SchemeRetriever can be used in any future session
    without re-ingestion.
    """

    def __init__(
        self,
        config: IngestionConfig | None = None,
        spark_session: Any | None = None,
        vs_client: Any | None = None,
    ) -> None:
        self.config = config or IngestionConfig()

        if spark_session:
            self.spark = spark_session
        else:
            try:
                from pyspark.sql import SparkSession
                self.spark = SparkSession.builder.getOrCreate()
            except ImportError as e:
                raise ImportError("pyspark not installed.") from e

        self.vsc = vs_client
        self._embedder = None

    def _ensure_embedder(self) -> None:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"⏳ Loading embedding model: {self.config.embed_model}…")
            self._embedder = SentenceTransformer(self.config.embed_model)

    def _ensure_vsc(self) -> None:
        if self.vsc is None:
            from databricks.vector_search.client import VectorSearchClient
            print("⏳ Initializing Vector Search client…")
            self.vsc = VectorSearchClient()

    @staticmethod
    def infer_category(name: str, description: str) -> str:
        combined = f"{name} {description}".lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                return category
        return "General / Other"

    @staticmethod
    def build_embed_text(scheme: dict) -> str:
        parts = [
            f"Scheme: {scheme['scheme_name']}",
            f"Eligibility: {scheme['eligibility']}",
            f"Description: {scheme['description']}",
            f"Benefits: {scheme['benefits']}",
            f"Application Process: {scheme['application_process']}",
        ]
        return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())

    def load_schemes(self) -> list[dict]:
        print(f"📦 Reading file from Volume path: {self.config.volume_path}")
        spark_df = (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .csv(self.config.volume_path)
        )
        pandas_df = spark_df.toPandas()
        pandas_df.columns = [str(c).strip() for c in pandas_df.columns]
        print(f"📄 Loaded {len(pandas_df)} schemes.")

        schemes = []
        for idx, row in pandas_df.iterrows():
            def safe(col: str) -> str:
                val = row.get(col, "")
                return str(val).strip() if val and str(val).lower() != "nan" else ""

            scheme = {
                "scheme_id":           idx,
                "scheme_name":         safe(self.config.col_name),
                "description":         safe(self.config.col_desc),
                "eligibility":         safe(self.config.col_eligibility),
                "benefits":            safe(self.config.col_benefits),
                "application_process": safe(self.config.col_app_process),
                "official_link":       safe(self.config.col_link),
            }
            scheme["category"]   = self.infer_category(scheme["scheme_name"], scheme["description"])
            scheme["embed_text"] = self.build_embed_text(scheme)
            schemes.append(scheme)

        return schemes

    def ingest_schemes_table(self, schemes: list[dict]) -> None:
        from pyspark.sql.types import (
            ArrayType, FloatType, StringType, StructField, StructType,
        )
        self._ensure_embedder()
        self._ensure_vsc()

        print(f"🚀 Embedding and indexing {len(schemes)} schemes…")
        rows = []
        for scheme in schemes:
            vector = self._embedder.encode(scheme["embed_text"]).tolist()
            rows.append({
                "id":                  f"SCHEME_{scheme['scheme_id']}",
                "text":                scheme["embed_text"],
                "embedding":           vector,
                "scheme_name":         scheme["scheme_name"],
                "eligibility":         scheme["eligibility"][:500],
                "benefits":            scheme["benefits"][:300],
                "application_process": scheme["application_process"][:300],
                "official_link":       scheme["official_link"],
                "category":            scheme["category"],
            })

        schema = StructType([
            StructField("id",                  StringType(),            False),
            StructField("text",                StringType(),            True),
            StructField("embedding",           ArrayType(FloatType()),  True),
            StructField("scheme_name",         StringType(),            True),
            StructField("eligibility",         StringType(),            True),
            StructField("benefits",            StringType(),            True),
            StructField("application_process", StringType(),            True),
            StructField("official_link",       StringType(),            True),
            StructField("category",            StringType(),            True),
        ])

        df = self.spark.createDataFrame(rows, schema=schema)
        print(f"💾 Writing to Delta table: {self.config.schemes_table}")
        df.write.format("delta").mode("overwrite").saveAsTable(self.config.schemes_table)

        self._enable_cdf(self.config.schemes_table)
        self._create_or_sync_index(
            self.config.schemes_index,
            self.config.schemes_table,
            len(rows[0]["embedding"]),
        )

    def ingest_categories_table(self, schemes: list[dict]) -> None:
        from pyspark.sql.types import (
            ArrayType, FloatType, IntegerType, StringType, StructField, StructType,
        )
        self._ensure_embedder()
        self._ensure_vsc()

        print("📂 Building category-level embeddings…")
        category_groups: dict[str, list[str]] = {}
        for scheme in schemes:
            category_groups.setdefault(scheme["category"], []).append(scheme["embed_text"])

        rows = []
        for cat_name, texts in category_groups.items():
            embeddings  = self._embedder.encode(texts)
            mean_vector = np.mean(embeddings, axis=0).tolist()
            safe_id     = re.sub(r"[^a-zA-Z0-9_-]", "_", cat_name)
            rows.append({
                "id":            safe_id,
                "embedding":     mean_vector,
                "category_name": cat_name,
                "scheme_count":  len(texts),
            })

        schema = StructType([
            StructField("id",            StringType(),           False),
            StructField("embedding",     ArrayType(FloatType()), True),
            StructField("category_name", StringType(),           True),
            StructField("scheme_count",  IntegerType(),          True),
        ])

        df = self.spark.createDataFrame(rows, schema=schema)
        df.write.format("delta").mode("overwrite").saveAsTable(self.config.categories_table)

        self._enable_cdf(self.config.categories_table)
        self._create_or_sync_index(
            self.config.categories_index,
            self.config.categories_table,
            len(rows[0]["embedding"]),
        )

    def _enable_cdf(self, table: str) -> None:
        try:
            self.spark.sql(
                f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
            )
            print(f"✅ Change Data Feed enabled for {table}")
        except Exception as e:
            print(f"⚠️  Could not enable CDF for {table}: {e}")

    def _create_or_sync_index(self, index_name: str, source_table: str, dim: int) -> None:
        import time
        try:
            print(f"🔍 Creating Vector Search index: {index_name}")
            self.vsc.create_delta_sync_index(
                endpoint_name=self.config.vs_endpoint_name,
                index_name=index_name,
                source_table_name=source_table,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_dimension=dim,
                embedding_vector_column="embedding",
            )
            print("✅ Index created successfully.")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("⚠️  Index already exists, syncing…")
                index = self.vsc.get_index(
                    endpoint_name=self.config.vs_endpoint_name,
                    index_name=index_name,
                )
                index.sync()
                print("✅ Index synced successfully.")
            else:
                raise

        # Wait for index to be ready before returning
        print(f"⏳ Waiting for index {index_name} to be ready…")
        while True:
            idx = self.vsc.get_index(
                endpoint_name=self.config.vs_endpoint_name,
                index_name=index_name,
            )
            status = idx.describe().get("status", {}).get("ready", False)
            if status:
                print(f"✅ Index {index_name} is ready.")
                break
            time.sleep(10)

    def setup_endpoint(self) -> None:
        self._ensure_vsc()
        try:
            self.vsc.get_endpoint(self.config.vs_endpoint_name)
            print(f"✅ VS endpoint '{self.config.vs_endpoint_name}' exists.")
        except Exception:
            print(f"🔨 Creating VS endpoint '{self.config.vs_endpoint_name}' (takes a few mins)…")
            self.vsc.create_endpoint(
                name=self.config.vs_endpoint_name,
                endpoint_type="STANDARD",
            )

    def run_pipeline(self) -> None:
        self.setup_endpoint()
        schemes = self.load_schemes()
        self.ingest_schemes_table(schemes)
        self.ingest_categories_table(schemes)
        print("\n🎉 Ingestion complete! Data is persistent in Databricks Vector Search.")
        print("   You can now use SchemeRetriever() in any future session without re-ingesting.")


# ============================================================================
# Retriever: Databricks Vector Search -> Cross-Encoder -> Groq LLM
# ============================================================================


class SchemeRetriever:
    """
    Stateful retrieval and chat pipeline backed by Databricks Vector Search.

    Queries the PERSISTENT DVS index — no ChromaDB, no EphemeralClient,
    no re-ingestion required between Databricks sessions.

    Usage (after ingestion has been run at least once):
        bot = SchemeRetriever()
        print(bot.chat("I am a farmer, what schemes can I get?"))
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        vs_client: Any | None = None,
        groq_client: Any | None = None,
    ) -> None:
        self.config  = config or RetrievalConfig()
        self.history: list[dict] = []

        # Lazily loaded resources
        self._embedder   = None
        self._reranker   = None
        self._groq       = groq_client
        self._vsc        = vs_client
        self._scheme_idx = None
        self._cat_idx    = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        """Load embedding + reranking models on first use."""
        if self._embedder is None:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            print("⏳ Loading embedding & reranking models…")
            self._embedder = SentenceTransformer(self.config.embed_model)
            self._reranker = CrossEncoder(self.config.rerank_model)

    def _ensure_vsc(self) -> None:
        """Connect to Databricks Vector Search and cache index handles."""
        if self._vsc is None:
            from databricks.vector_search.client import VectorSearchClient
            print("🔗 Connecting to Databricks Vector Search…")
            self._vsc = VectorSearchClient()

        if self._scheme_idx is None:
            self._scheme_idx = self._vsc.get_index(
                endpoint_name=self.config.vs_endpoint_name,
                index_name=self.config.schemes_index,
            )
            self._cat_idx = self._vsc.get_index(
                endpoint_name=self.config.vs_endpoint_name,
                index_name=self.config.categories_index,
            )
            print(f"✅ Connected to indices:")
            print(f"   • {self.config.schemes_index}")
            print(f"   • {self.config.categories_index}")

    def _ensure_groq(self) -> None:
        """Initialise Groq client (picks up GROQ_API_KEY from env automatically)."""
        if self._groq is None:
            try:
                from groq import Groq
                self._groq = Groq()   # reads GROQ_API_KEY from os.environ
            except ImportError as e:
                raise ImportError("groq not installed. Run: %pip install groq") from e

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _dvs_query(self, index: Any, query_vector: list[float], n: int, filters: dict | None = None) -> list[dict]:
        """
        Query a Databricks Vector Search index and return a flat list of
        result dicts. DVS returns results under result.data_array with
        columns defined in result.manifest.columns.
        """
        kwargs: dict = {
            "query_vector": query_vector,
            "num_results":  n,
            "columns":      ["id", "text", "scheme_name", "eligibility",
                             "benefits", "application_process",
                             "official_link", "category", "category_name"],
        }
        if filters:
            kwargs["filters"] = filters

        result   = index.similarity_search(**kwargs)
        cols     = [c["name"] for c in result.get("manifest", {}).get("columns", [])]
        rows     = result.get("result", {}).get("data_array", [])
        return [dict(zip(cols, row)) for row in rows]

    def retrieve(
        self,
        query: str,
        top_k_categories: int = 2,
        top_k_candidates: int = 10,
        final_n: int = 3,
    ) -> list[dict]:
        """
        Three-stage retrieval against Databricks Vector Search:
          1. Embed the query → find top-k matching categories.
          2. Query scheme index filtered to those categories.
          3. Cross-encoder rerank → return top final_n results.
        """
        self._ensure_models()
        self._ensure_vsc()

        query_vector = self._embedder.encode(query).tolist()

        # Stage 1 — Category routing
        cat_hits = self._dvs_query(self._cat_idx, query_vector, n=top_k_categories)
        target_categories = [h["category_name"] for h in cat_hits if h.get("category_name")]
        print(f"📂 Routing to categories: {target_categories}")

        # Stage 2 — Filtered scheme retrieval
        # DVS filter syntax: {"column_name filter_op": value}
        scheme_hits = []
        if target_categories:
            try:
                scheme_hits = self._dvs_query(
                    self._scheme_idx,
                    query_vector,
                    n=top_k_candidates,
                    filters={"category": target_categories},  # DVS "IN" filter
                )
            except Exception as e:
                print(f"⚠️  Category filter failed ({e}), falling back to global search.")

        # Fallback: global search if filter returned nothing
        if not scheme_hits:
            print("⚠️  No candidates from category filter — running global search.")
            scheme_hits = self._dvs_query(self._scheme_idx, query_vector, n=top_k_candidates)

        # Stage 3 — Cross-encoder reranking
        docs   = [h.get("text", "") for h in scheme_hits]
        pairs  = [[query, doc] for doc in docs]
        scores = self._reranker.predict(pairs)

        reranked = sorted(
            zip(scores, scheme_hits),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            {
                "score":               float(score),
                "text":                hit.get("text", ""),
                "scheme_name":         hit.get("scheme_name", "Unknown Scheme"),
                "eligibility":         hit.get("eligibility", ""),
                "benefits":            hit.get("benefits", ""),
                "application_process": hit.get("application_process", ""),
                "official_link":       hit.get("official_link", ""),
                "category":            hit.get("category", ""),
            }
            for score, hit in reranked[:final_n]
        ]

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_context(results: list[dict]) -> str:
        blocks = []
        for i, r in enumerate(results):
            block = (
                f"[Scheme {i + 1}] {r['scheme_name']}\n"
                f"Category       : {r['category']}\n"
                f"Eligibility    : {r['eligibility']}\n"
                f"Benefits       : {r['benefits']}\n"
                f"How to Apply   : {r['application_process']}\n"
                f"Official Link  : {r['official_link']}"
            )
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, question: str) -> str:
        """
        Retrieve relevant schemes, inject context, call Groq LLM, and
        return the answer. Conversation history is maintained across calls.
        """
        self._ensure_groq()

        results = self.retrieve(question)
        context = self.build_context(results)

        messages = [{
            "role":    "system",
            "content": f"{SYSTEM_PROMPT}\n\nSCHEME CONTEXT:\n{context}",
        }]
        messages.extend(self.history)
        messages.append({"role": "user", "content": question})

        response = self._groq.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        answer = response.choices[0].message.content.strip()

        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

        return answer

    def clear_history(self) -> None:
        """Reset the conversation context."""
        self.history.clear()


# ============================================================================
# Interactive Loop
# ============================================================================

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if mode == "ingest":
        print("\n🚀 Starting Scheme Sahayak Ingestion Pipeline…")
        ingester = SchemeIngester()
        ingester.run_pipeline()
    else:
        bot = SchemeRetriever()
        print("\n🌾 Scheme Sahayak — Government Scheme Eligibility Assistant")
        print("Describe your situation and I'll find the right schemes for you.")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"quit", "exit"}:
                    break
                print(f"\nScheme Sahayak: {bot.chat(user_input)}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[Error] {e}\n")