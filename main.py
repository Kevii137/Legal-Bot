"""
Nyaya Sahayak — Main Orchestration Pipeline
Combines IPC/BNS Comparator + Case Retriever + Scheme Checker + FIR Drafter
Supports Tamil ↔ English via Sarvam AI

Data files (single-file parquet, NOT Spark directories) expected at:
  ./data/ipc_bns_mapping.parquet
  ./data/ipc_bns_corpus.parquet
  ./data/case_summaries.parquet

Override the directory via env var LEGAL_BOT_DATA_DIR if needed.
"""

from __future__ import annotations
import os
import sys
import json
import re
import pandas as pd
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY env var not set. Configure it in Apps → Environment.")

MODEL = "llama-3.3-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Parquet file paths — default to ./data alongside main.py
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get(
    "LEGAL_BOT_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
MAPPING_PATH        = os.path.join(DATA_DIR, "ipc_bns_mapping.parquet")
CORPUS_PATH         = os.path.join(DATA_DIR, "ipc_bns_corpus.parquet")
CASE_SUMMARIES_PATH = os.path.join(DATA_DIR, "case_summaries.parquet")

# ---------------------------------------------------------------------------
# Make local modules importable
# ---------------------------------------------------------------------------
_BNS_IPC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bns_ipc")
if _BNS_IPC_DIR not in sys.path:
    sys.path.insert(0, _BNS_IPC_DIR)

from ipc_bns_comparator import Comparator
from case_retriever import CaseRetriever

# ---------------------------------------------------------------------------
# IPC/BNS Comparator — lazy singleton
# ---------------------------------------------------------------------------
_comparator: Comparator | None = None


def _get_comparator() -> Comparator:
    global _comparator
    if _comparator is None:
        if not os.path.isfile(MAPPING_PATH):
            raise FileNotFoundError(f"Mapping parquet not found at: {MAPPING_PATH}")
        if not os.path.isfile(CORPUS_PATH):
            raise FileNotFoundError(f"Corpus parquet not found at: {CORPUS_PATH}")

        print(f"[Comparator] Loading parquet files from {DATA_DIR}…")
        df_mapping = pd.read_parquet(MAPPING_PATH)
        df_corpus  = pd.read_parquet(CORPUS_PATH)
        print(f"[Comparator] {len(df_mapping)} mapping rows, {len(df_corpus)} corpus rows.")

        _comparator = Comparator(
            df_mapping=df_mapping,
            df_corpus=df_corpus,
            groq_client=client,
        )
    return _comparator


# ---------------------------------------------------------------------------
# Case Retriever — lazy singleton
# ---------------------------------------------------------------------------
_case_retriever: CaseRetriever | None = None


def _get_case_retriever() -> CaseRetriever:
    global _case_retriever
    if _case_retriever is None:
        if not os.path.isfile(CASE_SUMMARIES_PATH):
            raise FileNotFoundError(f"Case summaries parquet not found at: {CASE_SUMMARIES_PATH}")

        print(f"[CaseRetriever] Loading {CASE_SUMMARIES_PATH}…")
        df_summaries = pd.read_parquet(CASE_SUMMARIES_PATH)
        print(f"[CaseRetriever] Building FAISS index over {len(df_summaries)} cases (this takes ~30s)…")
        _case_retriever = CaseRetriever(
            df_summaries=df_summaries,
            groq_client=client,
        )
        _case_retriever._ensure_index()
        print(f"[CaseRetriever] Ready.")
    return _case_retriever


# ---------------------------------------------------------------------------
# Sarvam Language Integration
# ---------------------------------------------------------------------------
try:
    from language import to_english, from_english, is_configured as sarvam_configured
    SARVAM_AVAILABLE = sarvam_configured()
except ImportError:
    SARVAM_AVAILABLE = False

    def to_english(text, source_lang="auto"):
        return text

    def from_english(text, target_lang):
        return text

    def sarvam_configured():
        return False


def translate_to_english(text: str, lang: str) -> str:
    if lang == "ta" and SARVAM_AVAILABLE:
        try:
            return to_english(text, source_lang="ta-IN")
        except Exception as e:
            print(f"Translation to English failed: {e}")
    return text


def translate_from_english(text: str, lang: str) -> str:
    if lang == "ta" and SARVAM_AVAILABLE:
        try:
            return from_english(text, target_lang="ta-IN")
        except Exception as e:
            print(f"Translation from English failed: {e}")
    return text


# ---------------------------------------------------------------------------
# Agent Router Prompt — UPDATED with `cases` agent
# ---------------------------------------------------------------------------
ROUTER_PROMPT = """You are an intelligent legal assistant router for an Indian legal aid system called Nyaya Sahayak.

Classify the user's query into ONE or MORE of these agents (multiple allowed):

- "ipc_bns": Queries about IPC/BNS sections, criminal offences, legal provisions, what law applies, penalties, legal definitions, or scenarios where the user is asking what crime occurred.
  Examples: "What is IPC 302?", "my neighbour stole my phone", "is hacking a crime?", "Section 420 IPC explanation"

- "scheme": Queries about government welfare schemes, eligibility, subsidies, benefits, entitlements.
  Examples: "Am I eligible for PM Awas Yojana?", "what schemes help farmers?", "Ayushman Bharat coverage"

- "cases": Queries asking for PAST COURT CASES, PRECEDENTS, or PRIOR JUDGMENTS similar to a current situation. The user (often a lawyer or law student) wants to research how courts have ruled on similar facts before. Look for words like: "past cases", "precedents", "previous judgments", "similar cases", "case law", "court rulings", "prior decisions", "find cases", "show me cases", "research cases".
  Examples: "find me past cases of theft at night", "show me precedents for online cheating", "any prior judgments on dowry harassment quashing?", "give me previous cases related to bail in narcotics"

IMPORTANT distinctions:
- "what is the punishment for theft?" → ipc_bns (asking about the law itself)
- "find past cases of theft" → cases (asking for prior judgments)
- "my friend was robbed last night" → ipc_bns (describing a crime, wants to know what law applies)
- "show me cases similar to night-time robbery" → cases (researching precedents)
- "show me past cases of robbery and what sections apply" → cases AND ipc_bns (both)

Also determine:
- "needs_fir": true ONLY if the user explicitly asks to file/draft/write an FIR.
- "language": "ta" if query is in Tamil, "en" otherwise.

Respond with ONLY valid JSON (no markdown):
{
  "agents": ["ipc_bns"] | ["scheme"] | ["cases"] | ["ipc_bns", "scheme"] | ["ipc_bns", "cases"] | ...,
  "needs_fir": true | false,
  "language": "en" | "ta",
  "reasoning": "<one short sentence>"
}
"""

FIR_DETECTION_PROMPT = """You are analyzing a legal AI response to determine if the user's situation warrants filing an FIR (First Information Report) with the police.

Return ONLY valid JSON (no markdown):
{"recommend_fir": true | false, "reason": "<one sentence why or why not>"}

recommend_fir = true if:
- A cognizable offence was described (theft, assault, fraud, cheating, harassment, etc.)
- The user is a victim or witness to a crime
- Specific IPC/BNS sections were cited that involve criminal acts

recommend_fir = false if:
- This is a purely informational/academic query
- No crime or victim is involved
- User is asking about penalties in general
- User is researching past cases (lawyer-mode query)
"""


def llm_call(system: str, user: str, max_tokens: int = 1000, temp: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def extract_json(text: str) -> dict | None:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text.strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
    return None


def route_query(query: str, history: list[dict]) -> dict:
    context = ""
    if history:
        last = history[-3:] if len(history) > 3 else history
        context = "\n".join([f"{m['role'].upper()}: {m['content'][:200]}" for m in last])
        context = f"Recent conversation:\n{context}\n\n"

    raw = llm_call(ROUTER_PROMPT, f"{context}Current query: {query}", max_tokens=200, temp=0)
    result = extract_json(raw)
    if not result:
        return {"agents": ["ipc_bns"], "needs_fir": False, "language": "en", "reasoning": "fallback"}

    valid_agents = {"ipc_bns", "scheme", "cases"}
    result["agents"] = [a for a in result.get("agents", []) if a in valid_agents]
    if not result["agents"]:
        result["agents"] = ["ipc_bns"]
    return result


# ---------------------------------------------------------------------------
# IPC/BNS Agent
# ---------------------------------------------------------------------------
def run_ipc_bns_agent(query: str, history: list[dict]) -> dict:
    comp = _get_comparator()
    result = comp.handle_query(query)
    mode = result.get("mode", "unknown")

    if mode == "A_exact_lookup":
        r = result.get("result")
        e = result.get("explanation") or {}
        if isinstance(r, dict):
            response_text = (
                f"**IPC {r['ipc_section']}** ({r['ipc_heading']})\n"
                f"→ **BNS {r['bns_section']}** ({r['bns_heading']})\n"
                f"Status: {r['status']}\n\n"
                f"**Summary:** {e.get('summary', '')}\n\n"
                f"**What changed:** {e.get('substantive_change', e.get('what_changed', ''))}\n\n"
                f"**Severity:** {e.get('severity', '')}"
            )
        elif isinstance(r, list):
            lines = [f"BNS {result['router']['section']} absorbs {len(r)} IPC section(s):"]
            for item in r[:10]:
                lines.append(f"- IPC {item['ipc_section']}: {item['ipc_heading']}")
            response_text = "\n".join(lines)
        else:
            response_text = "Section not found in the mapping table."

    elif mode == "B_concept_search":
        r = result.get("result", {})
        ipc_lines = [f"- IPC {h['section']}: {h['heading']}" for h in r.get("ipc_matches", [])]
        bns_lines = [f"- BNS {h['section']}: {h['heading']}" for h in r.get("bns_matches", [])]
        response_text = (
            "**Relevant IPC sections:**\n" + "\n".join(ipc_lines) +
            "\n\n**Relevant BNS sections:**\n" + "\n".join(bns_lines)
        )

    elif mode == "C_scenario":
        r = result.get("result", {})
        reasoning = r.get("reasoning", {})
        sections = reasoning.get("applicable_sections", [])
        ipc_apps = [s for s in sections if s.get("code") == "IPC"]
        bns_apps = [s for s in sections if s.get("code") == "BNS"]
        if not sections:
            ipc_apps = reasoning.get("applicable_ipc_sections", [])
            bns_apps = reasoning.get("applicable_bns_sections", [])

        ipc_lines = [
            f"- IPC {s['section']} ({s['heading']}): {s.get('why_applicable') or s.get('relevance', '')}"
            for s in ipc_apps
        ]
        bns_lines = [
            f"- BNS {s['section']} ({s['heading']}): {s.get('why_applicable') or s.get('relevance', '')}"
            for s in bns_apps
        ]
        response_text = (
            f"**{reasoning.get('what_happened_legally', '')}**\n\n"
            + ("**Applicable IPC sections:**\n" + "\n".join(ipc_lines) + "\n\n" if ipc_lines else "")
            + ("**Applicable BNS sections:**\n" + "\n".join(bns_lines) + "\n\n" if bns_lines else "")
            + f"**Severity:** {reasoning.get('severity', '')}\n\n"
            + "**What you should do:**\n"
            + "\n".join(f"- {s}" for s in reasoning.get("what_you_should_do", []))
            + f"\n\n_{reasoning.get('disclaimer', 'Informational only, not legal advice.')}_"
        )
    else:
        response_text = "Could not classify the query."

    ipc_secs, bns_secs = [], []
    crime_described = mode in ("B_concept_search", "C_scenario")

    if mode == "A_exact_lookup" and isinstance(result.get("result"), dict):
        ipc_secs = [result["result"].get("ipc_section", "")]
        bns_secs = [result["result"].get("bns_section", "")]
    elif mode == "B_concept_search":
        r = result.get("result", {})
        ipc_secs = [h["section"] for h in r.get("ipc_matches", [])[:3]]
        bns_secs = [h["section"] for h in r.get("bns_matches", [])[:3]]
    elif mode == "C_scenario":
        reasoning = result.get("result", {}).get("reasoning", {})
        sections = reasoning.get("applicable_sections", [])
        if sections:
            ipc_secs = [s["section"] for s in sections if s.get("code") == "IPC"]
            bns_secs = [s["section"] for s in sections if s.get("code") == "BNS"]
        else:
            ipc_secs = [s["section"] for s in reasoning.get("applicable_ipc_sections", [])]
            bns_secs = [s["section"] for s in reasoning.get("applicable_bns_sections", [])]

    return {
        "agent": "ipc_bns",
        "response": response_text,
        "sections": {"ipc": ipc_secs, "bns": bns_secs, "crime_described": crime_described},
        "comparator_mode": mode,
    }


# ---------------------------------------------------------------------------
# Case Retriever Agent (NEW — first-class agent)
# ---------------------------------------------------------------------------
def run_cases_agent(query: str, history: list[dict], k: int = 5) -> dict:
    """
    Lawyer-mode case research. Retrieves top-K similar past judgments.
    Returns:
      - response: markdown summary for the chat bubble
      - cases: structured list for the side drawer (frontend reads this)
    """
    cr = _get_case_retriever()
    hits = cr.find_similar_cases(query, k=k, rerank_with_llm=False)

    cases = []
    for h in hits:
        cases.append({
            "case_id": h.get("case_id"),
            "title": h.get("title"),
            "court": h.get("court"),
            "court_type": h.get("court_type"),
            "doc_url": h.get("doc_url"),
            "cited_by_count": int(h.get("cited_by_count") or 0),
            "outcome": h.get("outcome"),
            "case_summary": h.get("case_summary"),
            "score": round(float(h.get("score", 0)), 3),
        })

    if cases:
        lines = [f"Found **{len(cases)} similar past case(s)**. Full details in the side panel →\n"]
        for i, c in enumerate(cases, 1):
            title_short = c["title"][:80] + ("…" if len(c["title"]) > 80 else "")
            lines.append(f"{i}. *{title_short}* — {c['court']} — {int(c['score']*100)}% match")
        response_text = "\n".join(lines)
    else:
        response_text = "No similar past cases found in the corpus."

    return {
        "agent": "cases",
        "response": response_text,
        "cases": cases,
        "query": query,
    }


def find_similar_cases(query: str, language: str = "en", k: int = 5,
                       rerank: bool = False) -> dict:
    """Standalone API wrapper for /api/cases."""
    english_query = translate_to_english(query, language) if language == "ta" else query

    cr = _get_case_retriever()
    hits = cr.find_similar_cases(english_query, k=k, rerank_with_llm=rerank)

    cases = []
    for h in hits:
        cases.append({
            "case_id": h.get("case_id"),
            "title": h.get("title"),
            "court": h.get("court"),
            "court_type": h.get("court_type"),
            "doc_url": h.get("doc_url"),
            "cited_by_count": int(h.get("cited_by_count") or 0),
            "outcome": h.get("outcome"),
            "case_summary": h.get("case_summary"),
            "score": round(float(h.get("score", 0)), 3),
            "why_relevant": h.get("why_relevant"),
        })

    return {
        "query": english_query,
        "language": language,
        "results": cases,
    }


# ---------------------------------------------------------------------------
# Scheme Agent
# ---------------------------------------------------------------------------
_scheme_bot = None
SCHEME_BOT_AVAILABLE = False


def _init_scheme_bot() -> bool:
    global _scheme_bot, SCHEME_BOT_AVAILABLE
    if _scheme_bot is not None:
        return SCHEME_BOT_AVAILABLE
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scheme_bot"))
        from scheme_sahayak_chat import SchemeChatBot, load_data, build_search_engine

        df_schemes, df_categories = load_data()
        vectorizer, tfidf_matrix = build_search_engine(df_schemes)

        _scheme_bot = SchemeChatBot(
            df_schemes, df_categories, vectorizer, tfidf_matrix,
            api_key=GROQ_API_KEY,
        )
        SCHEME_BOT_AVAILABLE = True
        print("✅ SchemeChatBot initialised successfully locally.")
    except Exception as e:
        print(f"⚠️  SchemeChatBot init failed ({type(e).__name__}: {e}). Using LLM fallback.")
        SCHEME_BOT_AVAILABLE = False
    return SCHEME_BOT_AVAILABLE


def _format_scheme_result(result: dict) -> str:
    mode = result.get("mode", "")
    lines = []

    if mode == "A_name_lookup":
        for s in result.get("schemes", [])[:3]:
            lines.append(f"## {s.get('scheme_name', 'Scheme')}")
            if s.get("category"):
                lines.append(f"**Category:** {s['category']}")
            if s.get("level"):
                lines.append(f"**Level:** {s['level']}")
            if s.get("eligibility"):
                lines.append(f"**Eligibility:** {str(s['eligibility'])[:400]}")
            if s.get("benefits"):
                lines.append(f"**Benefits:** {str(s['benefits'])[:300]}")
            if s.get("official_link"):
                lines.append(f"**Apply:** {s['official_link']}")
            lines.append("")

    elif mode == "B_category_search":
        cats = result.get("categories_searched", [])
        if cats:
            lines.append(f"**Categories matched:** {', '.join(cats)}\n")
        for s in result.get("schemes", [])[:5]:
            lines.append(f"### {s.get('scheme_name', '')}")
            if s.get("eligibility"):
                lines.append(str(s["eligibility"])[:200])
            lines.append("")

    elif mode == "C_eligibility_check":
        verdict = result.get("eligibility_result", {})
        for s in verdict.get("schemes", []):
            verdict_label = s.get("eligibility_verdict", "")
            emoji = "✅" if "Likely" in verdict_label else ("⚠️" if "Possibly" in verdict_label else "ℹ️")
            lines.append(f"## {emoji} {s.get('scheme_name', '')}")
            lines.append(f"**Verdict:** {verdict_label}")
            if s.get("why"):
                lines.append(f"**Why:** {s['why']}")
            if s.get("key_benefits"):
                lines.append(f"**Benefits:** {s['key_benefits']}")
            if s.get("how_to_apply"):
                lines.append(f"**How to apply:** {s['how_to_apply']}")
            if s.get("link"):
                lines.append(f"**Link:** {s['link']}")
            lines.append("")
        fup = verdict.get("follow_up_question")
        if fup:
            lines.append(f"❓ {fup}")
    else:
        lines.append(result.get("message", "No results found."))

    return "\n".join(lines).strip()


def run_scheme_agent(query: str, history: list[dict]) -> dict:
    def _to_scheme_history(h):
        pairs, i = [], 0
        while i + 1 < len(h):
            if h[i]["role"] == "user" and h[i+1]["role"] == "assistant":
                pairs.append([h[i]["content"], h[i+1]["content"]])
                i += 2
            else:
                i += 1
        return pairs

    if _init_scheme_bot():
        try:
            scheme_history = _to_scheme_history(history[-8:])
            result = _scheme_bot.handle_query(query, history=scheme_history)
            response_text = _format_scheme_result(result)
            return {
                "agent": "scheme",
                "response": response_text,
                "scheme_mode": result.get("mode"),
                "raw_result": result,
            }
        except Exception as e:
            print(f"SchemeChatBot query failed: {e}. Falling back to LLM.")

    SCHEME_FALLBACK_SYSTEM = """You are a government scheme eligibility assistant for India.
Identify relevant schemes, explain eligibility, describe benefits, give official portal links.
Cover schemes like PM Awas Yojana, Ayushman Bharat, PM Kisan, Beti Bachao, Ujjwala, MGNREGA, Mudra.
Mention that users should verify at official portals."""

    messages = [{"role": "system", "content": SCHEME_FALLBACK_SYSTEM}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.3, max_tokens=800,
    )
    return {
        "agent": "scheme",
        "response": resp.choices[0].message.content.strip(),
        "scheme_mode": "llm_fallback",
    }


def should_recommend_fir(query: str, agent_response: str) -> dict:
    combined = f"User query: {query}\n\nAgent response summary: {agent_response[:500]}"
    raw = llm_call(FIR_DETECTION_PROMPT, combined, max_tokens=150, temp=0)
    result = extract_json(raw)
    if not result:
        return {"recommend_fir": False, "reason": "Could not determine"}
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_query(query: str, history: list[dict], language: str = "en") -> dict:
    forced_lang = language if language in ("ta",) else None

    english_query = query
    if forced_lang == "ta":
        english_query = translate_to_english(query, "ta")

    routing = route_query(english_query, history)
    agents = routing.get("agents", ["ipc_bns"])
    needs_fir = routing.get("needs_fir", False)
    lang = forced_lang or routing.get("language", language)

    responses = []
    crime_described = False

    for agent in agents:
        if agent == "ipc_bns":
            result = run_ipc_bns_agent(english_query, history)
            result["_english_response"] = result.get("response", "")
            responses.append(result)
            if result.get("sections", {}).get("crime_described"):
                crime_described = True
        elif agent == "scheme":
            result = run_scheme_agent(english_query, history)
            result["_english_response"] = result.get("response", "")
            responses.append(result)
        elif agent == "cases":
            result = run_cases_agent(english_query, history)
            result["_english_response"] = result.get("response", "")
            responses.append(result)

    if lang == "ta":
        for r in responses:
            if r.get("response"):
                r["response"] = translate_from_english(r["response"], "ta")

    recommend_fir = False
    fir_reason = ""

    if not needs_fir and crime_described and responses:
        ipc_response = next((r for r in responses if r["agent"] == "ipc_bns"), None)
        if ipc_response:
            en_response = ipc_response.get("_english_response", ipc_response["response"])
            fir_check = should_recommend_fir(english_query, en_response)
            recommend_fir = fir_check.get("recommend_fir", False)
            fir_reason = fir_check.get("reason", "")
            if lang == "ta" and fir_reason:
                fir_reason = translate_from_english(fir_reason, "ta")

    return {
        "responses": responses,
        "recommend_fir": recommend_fir,
        "fir_reason": fir_reason,
        "agents_used": agents,
        "needs_fir": needs_fir,
        "language": lang,
        "routing_reason": routing.get("reasoning", ""),
        "sarvam_active": SARVAM_AVAILABLE and lang == "ta",
    }