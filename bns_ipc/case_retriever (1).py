"""
Case retriever — semantic search over Indian criminal case summaries.

Build-time:
    summaries (Delta) -> embed -> FAISS index in memory

Query-time:
    user query -> embed -> FAISS top-k -> LLM ranker/summarizer (optional)

Public API:
    cr = CaseRetriever(df_summaries=df_summaries, groq_client=client)
    cr.find_similar_cases(query="someone hacked my account", k=5,
                          rerank_with_llm=True) -> list[dict]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 70B for the on-demand reranker; can fall back to 8B if quotas hit
DEFAULT_RERANK_MODEL = "llama-3.3-70b-versatile"
RERANK_FALLBACK_MODEL = "llama-3.1-8b-instant"


# ----------------------------------------------------------------------------
# Embed text — what we actually feed into the index per case
# ----------------------------------------------------------------------------

def build_search_text(row: pd.Series | dict) -> str:
    """
    Compose the text we embed for a case row.

    The summaries Delta has columns:
      - case_summary       (4-sentence narrative)
      - sections_invoked_json  (list of section strings)
      - key_topics_json    (list of topic keywords)
      - title              (case name from court records)
      - court              (court name)
    """
    def _get(key, default=""):
        v = row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
        return "" if v is None else str(v)

    summary = _get("case_summary")
    title = _get("title")
    court = _get("court")

    try:
        sections = json.loads(_get("sections_invoked_json") or "[]")
    except Exception:
        sections = []
    try:
        topics = json.loads(_get("key_topics_json") or "[]")
    except Exception:
        topics = []

    parts = [
        f"Case: {title}",
        f"Court: {court}",
        f"Summary: {summary}",
    ]
    if sections:
        parts.append("Sections: " + ", ".join(str(s) for s in sections))
    if topics:
        parts.append("Topics: " + ", ".join(str(t) for t in topics))
    return ". ".join(p for p in parts if p)


# ----------------------------------------------------------------------------
# Reranker prompt — runs at query time on the top-k retrieved cases
# ----------------------------------------------------------------------------

RERANK_SYSTEM_PROMPT = """You are a legal research assistant ranking past Indian criminal cases by relevance to a current case description.

You will receive a user's case description and a list of retrieved precedent cases. For each retrieved case, decide:
  - similarity_score: 0-100, how relevant is this precedent to the user's case
  - why_relevant: one sentence explaining the connection
  - distinguishing_factors: one sentence on how the precedent differs (or null)

Respond with ONLY valid json in this schema (no markdown, no fences):
{
  "ranked_cases": [
    {
      "case_id": "<unchanged from input>",
      "similarity_score": <0-100 integer>,
      "why_relevant": "<one sentence>",
      "distinguishing_factors": "<one sentence or null>"
    }
  ]
}

Do NOT add cases that weren't in the input. Sort the output by similarity_score descending.
"""


# ----------------------------------------------------------------------------
# CaseRetriever
# ----------------------------------------------------------------------------

@dataclass
class CaseRetrieverConfig:
    embed_model: str = DEFAULT_EMBED_MODEL
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_fallback_model: str = RERANK_FALLBACK_MODEL
    max_text_len_for_rerank: int = 600  # chars per case in the rerank prompt


class CaseRetriever:
    """Lazy-built FAISS index over case summaries; LLM rerank on demand."""

    def __init__(
        self,
        df_summaries: pd.DataFrame,
        groq_client: Any | None = None,
        config: CaseRetrieverConfig | None = None,
    ) -> None:
        self.config = config or CaseRetrieverConfig()
        self.df = df_summaries.reset_index(drop=True).copy()
        self.df["search_text"] = self.df.apply(build_search_text, axis=1)
        self._embedder = None
        self._index = None
        self._groq = groq_client  # may be None if user never uses rerank

    # ------------------------------------------------------------------
    # Lazy index build
    # ------------------------------------------------------------------
    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Install: pip install faiss-cpu sentence-transformers"
            ) from e

        self._embedder = SentenceTransformer(self.config.embed_model)
        embeddings = self._embedder.encode(
            self.df["search_text"].tolist(),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    # ------------------------------------------------------------------
    # Plain semantic search
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> list[dict]:
        """Pure FAISS retrieval. Returns top-k case dicts with score + rank."""
        self._ensure_index()
        q_emb = self._embedder.encode(
            [query.strip()], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores, indices = self._index.search(q_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            row = self.df.iloc[int(idx)].to_dict()
            row.pop("search_text", None)  # internal, don't leak
            row["score"] = float(score)
            row["rank"] = rank
            # Promote some fields out of summary_json for easier UI access
            row["summary_obj"] = self._safe_load_json(row.get("summary_json"))
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # LLM rerank — uses one Groq call to rank top-k semantic hits
    # ------------------------------------------------------------------
    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        """
        Send top-k hits + the user's query to an LLM and have it produce
        similarity scores + reasoning per case. Returns the hits enriched
        with `llm_score`, `why_relevant`, `distinguishing_factors`.
        """
        if not self._groq or not hits:
            return hits

        # Build a compact, ordered list of cases for the LLM to score
        cases_block_parts = []
        for h in hits:
            summary = (h.get("case_summary") or "")[: self.config.max_text_len_for_rerank]
            cases_block_parts.append(
                f"case_id: {h['case_id']}\n"
                f"title: {h.get('title','')}\n"
                f"court: {h.get('court','')}\n"
                f"summary: {summary}\n"
            )
        cases_block = "\n---\n".join(cases_block_parts)

        user_msg = (
            f"User's case: {query}\n\n"
            f"Retrieved precedent cases:\n\n{cases_block}\n\n"
            f"Rank these cases by relevance and return json."
        )

        # Try 70B; fall back to 8B if it errors (most likely TPD/TPM)
        for model_id in (self.config.rerank_model, self.config.rerank_fallback_model):
            try:
                resp = self._groq.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                    response_format={"type": "json_object"},
                )
                ranked = json.loads(resp.choices[0].message.content)
                ranked_list = ranked.get("ranked_cases", [])
                # Build a quick lookup so we can merge LLM judgments back into hits
                judgments = {r["case_id"]: r for r in ranked_list if "case_id" in r}
                enriched = []
                for h in hits:
                    j = judgments.get(h["case_id"], {})
                    enriched.append({
                        **h,
                        "llm_score": j.get("similarity_score"),
                        "why_relevant": j.get("why_relevant"),
                        "distinguishing_factors": j.get("distinguishing_factors"),
                    })
                # Sort by llm_score where available, falling back to FAISS rank
                enriched.sort(
                    key=lambda x: (-(x.get("llm_score") or 0), x.get("rank", 999))
                )
                return enriched
            except Exception:
                continue
        # Both models failed — return the unranked hits
        return hits

    # ------------------------------------------------------------------
    # The main public method
    # ------------------------------------------------------------------
    def find_similar_cases(
        self,
        query: str,
        k: int = 5,
        rerank_with_llm: bool = True,
    ) -> list[dict]:
        """
        Top-level lawyer-facing search.
        Returns up to k cases ranked by semantic similarity (and LLM if requested).

        Each result dict has:
            case_id, title, court, court_type, doc_url,
            cites_count, cited_by_count, outcome,
            case_summary, summary_obj (full extracted dict),
            score (FAISS), rank,
            llm_score, why_relevant, distinguishing_factors  (if rerank=True)
        """
        # Over-fetch so the LLM has slightly more candidates to choose from
        over_fetch = min(k * 2, 15)
        hits = self.search(query, k=over_fetch)
        if rerank_with_llm and self._groq:
            hits = self.rerank(query, hits)
        return hits[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_load_json(s):
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}