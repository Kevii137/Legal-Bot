# Databricks notebook source
# MAGIC %md
# MAGIC # Scheme Sahayak Pipeline
# MAGIC This notebook runs the end-to-end ingestion and retrieval pipeline for the Government Scheme Eligibility assistant. 
# MAGIC
# MAGIC First, let's install the required dependencies.

# COMMAND ----------

# Install the necessary packages. 
# (If running in Databricks, %pip ensures the packages are installed on the cluster)
%pip install sentence-transformers databricks-vectorsearch groq --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup
# MAGIC Set up your API keys and restart the Python kernel if prompted after the installation step.

# COMMAND ----------

# ⚠️  SKIP THIS CELL IF YOU HAVE ALREADY RUN INGESTION BEFORE

from scheme_sahayak import SchemeIngester

ingester = SchemeIngester()
ingester.run_pipeline()   # builds persistent DVS index; takes ~10 mins for 3400 schemes

# COMMAND ----------

# DBTITLE 1,Fix SchemeRetriever to use Vector Search
# Patch SchemeRetriever to use Databricks Vector Search instead of ChromaDB
from dataclasses import dataclass
from typing import Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from databricks.vector_search.client import VectorSearchClient
from groq import Groq

@dataclass
class RetrievalConfig:
    catalog: str = "workspace"
    schema: str = "default"
    vs_endpoint_name: str = "scheme_bot_endpoint"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 1024
    temperature: float = 0.2

    @property
    def schemes_index(self) -> str:
        return f"{self.catalog}.{self.schema}.gov_schemes_index"

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

class FixedSchemeRetriever:
    """Retrieval and chat pipeline using Databricks Vector Search."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        vs_client: Any | None = None,
        groq_client: Any | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.history: list[dict] = []
        
        self._vsc = vs_client
        self._groq = groq_client
        self._embedder = None
        self._reranker = None

    def _ensure_models(self) -> None:
        if self._embedder is None:
            print("⏳ Loading embedding & reranking models…")
            self._embedder = SentenceTransformer(self.config.embed_model)
            self._reranker = CrossEncoder(self.config.rerank_model)

        if self._vsc is None:
            print("🔗 Connecting to Databricks Vector Search…")
            self._vsc = VectorSearchClient(disable_notice=True)

        if self._groq is None:
            self._groq = Groq()

    def retrieve(self, query: str, top_k_cand: int = 20, final_n: int = 5) -> list[dict]:
        self._ensure_models()
        query_emb = self._embedder.encode(query).tolist()

        # Query the schemes index directly
        scheme_index = self._vsc.get_index(
            endpoint_name=self.config.vs_endpoint_name,
            index_name=self.config.schemes_index
        )
        
        scheme_results = scheme_index.similarity_search(
            query_vector=query_emb,
            columns=["scheme_name", "eligibility", "benefits", "application_process", "official_link", "text"],
            num_results=top_k_cand
        )

        candidates = []
        if scheme_results and "result" in scheme_results and "data_array" in scheme_results["result"]:
            for row in scheme_results["result"]["data_array"]:
                if len(row) >= 6:
                    candidates.append({
                        "scheme_name": row[0],
                        "eligibility": row[1],
                        "benefits": row[2],
                        "application_process": row[3],
                        "official_link": row[4],
                        "text": row[5]
                    })

        # Rerank
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self._reranker.predict(pairs)
        
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:final_n]

    def chat(self, user_message: str) -> str:
        self._ensure_models()
        
        schemes = self.retrieve(user_message)
        
        if not schemes:
            context = "No relevant schemes found for your query."
        else:
            context_parts = []
            for i, scheme in enumerate(schemes, 1):
                context_parts.append(
                    f"[Scheme {i}] {scheme['scheme_name']}\n"
                    f"Eligibility: {scheme['eligibility']}\n"
                    f"Benefits: {scheme['benefits']}\n"
                    f"Application: {scheme['application_process']}\n"
                    f"Link: {scheme['official_link']}"
                )
            context = "\n\n".join(context_parts)

        self.history.append({"role": "user", "content": user_message})
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-10:])
        messages.append({
            "role": "system",
            "content": f"**Retrieved Scheme Context:**\n{context}"
        })

        completion = self._groq.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        assistant_reply = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_reply})
        
        return assistant_reply

    def clear_history(self) -> None:
        self.history = []

print("✅ Fixed SchemeRetriever loaded (simplified - no category routing)!")

# COMMAND ----------

import os
from scheme_sahayak import SchemeRetriever

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")   # 🔑 Set GROQ_API_KEY in your environment

bot = SchemeRetriever()
bot.clear_history()

print("🌾 Scheme Sahayak — Ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break
    print(f"\nScheme Sahayak:\n{bot.chat(user_input)}\n")
    print("-" * 60)

# COMMAND ----------



# COMMAND ----------

