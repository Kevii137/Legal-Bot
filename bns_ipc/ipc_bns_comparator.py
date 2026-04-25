"""
IPC-BNS Comparator — Shared library for the Nyaya-Sahayak hackathon team.

Provides a unified query interface (Mode A / B / C) for comparing Indian Penal
Code (IPC) sections with their Bharatiya Nyaya Sanhita (BNS) successors.

Usage (from a notebook or another agent like the FIR maker):

    import os
    os.environ["GROQ_API_KEY"] = "..."   # set before importing

    from ipc_bns_comparator import Comparator

    comp = Comparator(df_mapping=df_mapping, df_corpus=df_corpus)

    # High-level — auto-routes to the right mode
    result = comp.handle_query("someone hacked my account and stole money")

    # Or call specific modes directly
    row      = comp.lookup_ipc("302")
    reverse  = comp.lookup_bns("103")
    concept  = comp.mode_b_concept_search("murder")
    scenario = comp.mode_c_scenario("my neighbour threatened me with a knife")

For the FIR maker: `comp.mode_c_scenario(...)` returns the applicable IPC/BNS
sections grounded in the corpus — feed those sections directly into your
FIR-drafting prompt to guarantee citation accuracy.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ============================================================================
# Constants
# ============================================================================

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
MIN_TEXT_LEN = 100  # Corpus rows shorter than this are noise (definition stubs)
MAX_EMBED_TEXT_LEN = 2000  # Truncate long legal text before embedding

SPELLING_VARIANTS = {
    "organized": "organised",
    "authorized": "authorised",
    "recognized": "recognised",
    "criticize": "criticise",
    "analyze": "analyse",
    "offense": "offence",
    "offenses": "offences",
    "defense": "defence",
    "license": "licence",
    "labor": "labour",
}

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by",
    "is", "are", "was", "were", "be", "and", "or", "me", "my", "someone",
    "what", "which", "how", "about", "tell",
}

# ============================================================================
# Prompts
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are a query classifier for an Indian legal tool that compares IPC (old) and BNS (new) criminal law sections.

Classify the user's query into exactly ONE of three modes:

- "A_exact_lookup": User wants to look up a specific section by number.
  Examples: "302 IPC", "what is section 420", "tell me about IPC 498A", "BNS 103 meaning"

- "B_concept_search": User gave a short concept or crime type (no specific number).
  Examples: "murder", "cyber fraud", "sexual harassment", "organized crime", "defamation"

- "C_scenario": User described a real-life situation and wants to know what law applies.
  Examples: "someone hacked my account and stole money", "my neighbor is threatening me", "I was cheated by an online seller"

Also extract the section number if the query contains one.

Respond with ONLY valid JSON in this exact format (no markdown, no code fences, no extra text):
{"mode": "A_exact_lookup" | "B_concept_search" | "C_scenario", "section": "<number or null>", "code": "IPC" | "BNS" | null, "reasoning": "<one short sentence>"}
"""

EXPLAINER_SYSTEM_PROMPT = """You are a concise legal mapper. Compare the provided IPC and BNS texts.

Rules:
- NO conversational filler.
- NO legal advice.
- Focus strictly on the direct mapping and the primary change.

Respond with ONLY valid JSON:
{
  "summary": "<one-sentence description>",
  "substantive_change": "<Briefly state if the punishment or definition changed, else 'None'>",
  "severity": "low | medium | high | capital"
}
"""

SCENARIO_EXTRACTOR_PROMPT = """You analyze Indian criminal law scenarios and extract the underlying legal concepts.

Given a scenario, identify 2-4 distinct legal concepts or offense types that may apply. Be specific but concise. Use terminology that appears in Indian criminal statutes.

Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
{"concepts": ["<concept1>", "<concept2>", ...]}

Examples:
Scenario: "someone hacked my account and stole money"
{"concepts": ["cheating by personation", "identity theft", "dishonest misappropriation of property", "criminal breach of trust"]}

Scenario: "my neighbor is threatening me with a knife"
{"concepts": ["criminal intimidation", "assault with a deadly weapon", "threat to cause hurt"]}
"""

SCENARIO_REASONER_PROMPT = """You are a legal assistant helping a citizen understand what happened to them and what they can do about it.

The user described a real situation. Your job:
1. Tell them clearly what crimes were committed against them.
2. Cite the current applicable law (BNS section + number).
3. Only mention IPC if it is still relevant (e.g. for an ongoing case filed before 2024).
4. Never explain how IPC became BNS. Never compare old vs new law.

Respond with ONLY valid JSON:
{
  "what_happened_legally": "<One sentence: what crime(s) this situation constitutes>",
  "applicable_sections": [
    {
      "code": "BNS" | "IPC",
      "section": "<num>",
      "heading": "<text>",
      "relevance": "<One sentence: why this applies to the user's specific facts>"
    }
  ],
  "severity": "low | medium | high | capital",
  "what_you_should_do": ["<Step 1>", "<Step 2>", "<Step 3>"],
  "disclaimer": "Informational only, not legal advice."
}
"""

# ============================================================================
# Helpers
# ============================================================================

def normalize_section(s: str) -> str:
    """Strip 'IPC'/'BNS'/'Section' prefixes and suffixes from a section query."""
    s = str(s).strip().upper()
    s = re.sub(r"\s+OF\s+(IPC|BNS)$", "", s)
    s = re.sub(r"\s+(IPC|BNS)$", "", s)
    s = re.sub(r"^(IPC|BNS)\s+SECTION\s+", "", s)
    s = re.sub(r"^(IPC|BNS)\s+", "", s)
    s = re.sub(r"^SECTION\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_spelling(text: str) -> str:
    """American -> British (IPC/BNS use British spellings)."""
    t = text.lower()
    for us, uk in SPELLING_VARIANTS.items():
        t = re.sub(rf"\b{us}\b", uk, t)
    return t


# ============================================================================
# Comparator
# ============================================================================

@dataclass
class ComparatorConfig:
    embed_model: str = DEFAULT_EMBED_MODEL
    llm_model: str = DEFAULT_GROQ_MODEL
    min_text_len: int = MIN_TEXT_LEN
    max_embed_text_len: int = MAX_EMBED_TEXT_LEN


class Comparator:
    """
    End-to-end IPC-BNS comparator.

    Takes the mapping and corpus DataFrames (loaded from Delta by the caller)
    and exposes the three query modes plus a unified handle_query() entry point.
    """

    def __init__(
        self,
        df_mapping: pd.DataFrame,
        df_corpus: pd.DataFrame,
        config: ComparatorConfig | None = None,
        groq_client: Any | None = None,
    ) -> None:
        self.config = config or ComparatorConfig()
        self.df_mapping = df_mapping.copy()
        self.df_corpus = df_corpus[
            df_corpus["text"].str.len() >= self.config.min_text_len
        ].reset_index(drop=True)

        # LLM client — caller can inject one, else we build from GROQ_API_KEY
        if groq_client is None:
            try:
                from groq import Groq
                self._groq = Groq()  # reads GROQ_API_KEY from env
            except ImportError as e:
                raise ImportError(
                    "groq not installed. Run: pip install groq"
                ) from e
        else:
            self._groq = groq_client

        # FAISS + embedder are built lazily on first semantic search
        self._embedder = None
        self._index = None

    # ------------------------------------------------------------------
    # Lazy initialisation of embedding index
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Install FAISS + sentence-transformers: "
                "pip install faiss-cpu sentence-transformers"
            ) from e

        self._embedder = SentenceTransformer(self.config.embed_model)

        def build_search_text(row: pd.Series) -> str:
            heading = str(row.get("heading") or "").strip()
            text = str(row.get("text") or "").strip()[: self.config.max_embed_text_len]
            return f"{heading}. {text}" if heading else text

        texts = self.df_corpus.apply(build_search_text, axis=1).tolist()
        embeddings = self._embedder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    # ------------------------------------------------------------------
    # Level 1: exact section lookup
    # ------------------------------------------------------------------

    def lookup_ipc(self, query: str) -> dict | None:
        """Forward lookup: IPC number -> mapping row (dict) or None."""
        q = normalize_section(query)
        hits = self.df_mapping[self.df_mapping["ipc_section"] == q]
        return hits.iloc[0].to_dict() if not hits.empty else None

    def lookup_bns(self, query: str) -> list[dict]:
        """Reverse lookup: BNS number -> list of IPC sections that map to it."""
        q = normalize_section(query)
        hits = self.df_mapping[self.df_mapping["bns_section"] == q]
        return hits.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Level 2: LLM-enriched explanation
    # ------------------------------------------------------------------

    def explain_mapping(self, mapping: dict | None) -> dict:
        """Given a mapping row, produce structured LLM explanation."""
        if not mapping:
            return {"error": "No mapping provided"}

        max_len = 3000
        ipc_desc = (mapping.get("ipc_description") or "")[:max_len]
        bns_desc = (mapping.get("bns_description") or "")[:max_len]

        user_msg = f"""IPC Section {mapping['ipc_section']} — {mapping['ipc_heading']}

Full IPC text:
{ipc_desc}

---

BNS Section {mapping['bns_section']} — {mapping['bns_heading']}

Full BNS text:
{bns_desc}

---

Status of this mapping: {mapping['status']}

Generate the structured explanation now."""

        try:
            response = self._groq.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def llm_route(self, query: str) -> dict:
        """Classify a query into one of the three modes."""
        if not query or not query.strip():
            return {"mode": "invalid", "section": None, "code": None,
                    "reasoning": "empty query"}
        try:
            response = self._groq.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query.strip()},
                ],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            if parsed.get("mode") not in {"A_exact_lookup", "B_concept_search", "C_scenario"}:
                parsed["mode"] = "unknown"
            return parsed
        except Exception as e:
            return {"mode": "error", "section": None, "code": None,
                    "reasoning": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------
    # Mode B: semantic concept search
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, k: int = 10) -> list[dict]:
        """Low-level: embed query, FAISS search, return top-k rows with scores."""
        self._ensure_index()
        q_emb = self._embedder.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, indices = self._index.search(q_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            row = self.df_corpus.iloc[int(idx)].to_dict()
            row["score"] = float(score)
            row["rank"] = rank
            results.append(row)
        return results

    def _keyword_boost(
        self, query: str, hits: list[dict], boost: float = 0.15
    ) -> list[dict]:
        """Boost hits whose heading contains query words (spelling-normalized)."""
        query_normalized = normalize_spelling(query)
        query_words = [
            w for w in re.findall(r"[a-zA-Z]+", query_normalized)
            if len(w) > 2 and w not in STOPWORDS
        ]
        if not query_words:
            return hits

        out = []
        for h in hits:
            heading_norm = normalize_spelling(str(h.get("heading", "")))
            matches = sum(1 for w in query_words if w in heading_norm)
            if matches > 0:
                h = {**h, "score": min(1.0, h["score"] + boost * matches),
                     "_boosted": True, "_boost_matches": matches}
            else:
                h = {**h, "_boosted": False}
            out.append(h)
        return sorted(out, key=lambda x: x["score"], reverse=True)

    def _ensure_counterparts(self, hits: list[dict], k: int) -> list[dict]:
        """Add IPC/BNS counterparts for top-k hits using the mapping table."""
        ipc_in = {h["section"]: h for h in hits if h["source_code"] == "IPC"}
        bns_in = {h["section"]: h for h in hits if h["source_code"] == "BNS"}

        added = []

        for ipc_sec, h in list(ipc_in.items())[:k]:
            counterpart = h.get("counterpart_section")
            if counterpart and counterpart not in bns_in:
                paired = self.df_corpus[
                    (self.df_corpus["source_code"] == "BNS")
                    & (self.df_corpus["section"] == counterpart)
                ]
                if not paired.empty:
                    row = paired.iloc[0].to_dict()
                    row["score"] = h["score"] * 0.9
                    row["_paired_from"] = f"IPC {ipc_sec}"
                    added.append(row)

        for bns_sec, h in list(bns_in.items())[:k]:
            counterpart = h.get("counterpart_section")
            if counterpart and counterpart not in ipc_in:
                paired = self.df_corpus[
                    (self.df_corpus["source_code"] == "IPC")
                    & (self.df_corpus["section"] == counterpart)
                ]
                if not paired.empty:
                    row = paired.iloc[0].to_dict()
                    row["score"] = h["score"] * 0.9
                    row["_paired_from"] = f"BNS {bns_sec}"
                    added.append(row)

        return hits + added

    def mode_b_concept_search(self, query: str, k: int = 5) -> dict:
        """Concept/keyword search with boost + counterpart pairing."""
        raw = self.semantic_search(query, k=k * 3)
        boosted = self._keyword_boost(query, raw)
        with_pairs = self._ensure_counterparts(boosted, k=k)

        def _top(hits: list[dict], source: str) -> list[dict]:
            filtered = [h for h in hits if h["source_code"] == source]
            filtered.sort(key=lambda x: x["score"], reverse=True)
            seen, out = set(), []
            for h in filtered:
                key = h["section"]
                if key in seen:
                    continue
                seen.add(key)
                out.append(h)
            return out[:k]

        return {
            "query": query,
            "ipc_matches": _top(with_pairs, "IPC"),
            "bns_matches": _top(with_pairs, "BNS"),
        }

    # ------------------------------------------------------------------
    # Mode C: scenario reasoning with RAG grounding
    # ------------------------------------------------------------------

    def extract_concepts(self, scenario: str) -> list[str]:
        """LLM-extract 2-4 legal concepts from a natural-language scenario."""
        try:
            response = self._groq.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": SCENARIO_EXTRACTOR_PROMPT},
                    {"role": "user", "content": scenario.strip()},
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            concepts = parsed.get("concepts", [])
            return [str(c).strip() for c in concepts if c] if isinstance(concepts, list) else []
        except Exception:
            return []

    def mode_c_scenario(self, scenario: str, k_per_concept: int = 3) -> dict:
        """
        Full scenario reasoning pipeline:
          1. Extract legal concepts from scenario
          2. Retrieve candidate IPC+BNS sections per concept (RAG)
          3. LLM reasons over scenario + candidates (grounded, no hallucinations)

        Returns a dict with 'extracted_concepts', 'retrieved_candidates',
        and 'reasoning' (the LLM output with applicable sections, advice, etc.).

        For the FIR maker: pass the resulting `applicable_sections`
        (inside 'reasoning') into your FIR prompt.
        """
        concepts = self.extract_concepts(scenario)
        if not concepts:
            return {"error": "Could not extract legal concepts from the scenario"}

        all_candidates = []
        seen = set()
        for concept in concepts:
            hits = self.semantic_search(concept, k=k_per_concept * 2)
            for h in hits[: k_per_concept * 2]:
                key = (h["source_code"], h["section"])
                if key in seen:
                    continue
                seen.add(key)
                all_candidates.append({
                    "source": h["source_code"],
                    "section": h["section"],
                    "text": h.get("text", "")[:500],
                    "heading": h["heading"],
                    "matched_concept": concept,
                    "score": h["score"],
                })

        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        top = all_candidates[:20]

        candidate_lines = [
            f"  {c['source']} {c['section']} — {c['heading']}: {c['text']}"
            for c in top
        ]
        candidates_block = "Relevant legal sections:\n" + "\n".join(candidate_lines)

        user_msg = f"""Scenario: {scenario}

Extracted legal concepts: {', '.join(concepts)}

Candidate sections (retrieved via semantic search):
{candidates_block}

Now produce the structured legal analysis."""

        try:
            response = self._groq.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": SCENARIO_REASONER_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            reasoning = json.loads(response.choices[0].message.content)
            return {
                "scenario": scenario,
                "extracted_concepts": concepts,
                "retrieved_candidates": top,
                "reasoning": reasoning,
            }
        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "concepts": concepts,
                "candidates": top,
            }

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def handle_query(self, query: str, *, explain: bool = False) -> dict:
        """
        Unified pipeline: LLM router -> dispatch to Mode A / B / C.

        Returns a dict with:
          - mode:    which mode was chosen
          - router:  the router's raw output (section, code, reasoning)
          - result:  the mode-specific payload
          - (Mode A only) explanation: Level 2 LLM explanation
        """
        routed = self.llm_route(query)
        mode = routed.get("mode")
        section = routed.get("section")
        code = routed.get("code") or "IPC"

        if mode == "A_exact_lookup":
            if not section:
                return {"mode": mode, "router": routed, "result": None,
                        "error": "Router chose Mode A but extracted no section number"}
            if code == "BNS":
                return {"mode": mode, "router": routed,
                        "result_type": "reverse_bns_lookup",
                        "result": self.lookup_bns(section)}
            result = self.lookup_ipc(section)
            explanation = self.explain_mapping(result) if (result and explain) else None
            return {"mode": mode, "router": routed,
                    "result_type": "forward_ipc_lookup",
                    "result": result, "explanation": explanation}

        if mode == "B_concept_search":
            return {"mode": mode, "router": routed,
                    "result": self.mode_b_concept_search(query)}

        if mode == "C_scenario":
            return {"mode": mode, "router": routed,
                    "result": self.mode_c_scenario(query)}

        return {"mode": mode, "router": routed, "result": None,
                "note": "Unclassified query"}