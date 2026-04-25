"""
Nyaya Sahayak — Main Orchestration Pipeline
Combines IPC/BNS Comparator + Scheme Checker + FIR Drafter
"""

from __future__ import annotations

import json
import os
import re

from groq import Groq


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Agent Router Prompt
# ---------------------------------------------------------------------------
ROUTER_PROMPT = """You are an intelligent legal assistant router for an Indian legal aid system called Nyaya Sahayak.

Classify the user's query into ONE or MORE of these agents (can be multiple):
- "ipc_bns": For any query about IPC/BNS sections, criminal offences, legal provisions, what law applies, penalties, legal definitions
- "scheme": For any query about government welfare schemes, eligibility, benefits, subsidies, entitlements

Also determine:
- "needs_fir": true ONLY if the user explicitly asks to file/draft/write an FIR. Not just describing a crime — they must ask for the FIR.
- "language": "hi" if query is in Hindi/Hinglish, else "en"

Respond with ONLY valid JSON (no markdown):
{
  "agents": ["ipc_bns"] | ["scheme"] | ["ipc_bns", "scheme"],
  "needs_fir": true | false,
  "language": "en" | "hi",
  "reasoning": "<one short sentence>"
}
"""

# ---------------------------------------------------------------------------
# IPC/BNS Agent (standalone LLM call — no embeddings for now)
# ---------------------------------------------------------------------------
IPC_BNS_SYSTEM = """You are a knowledgeable Indian legal assistant specializing in comparing the Indian Penal Code (IPC, 1860) with its successor the Bharatiya Nyaya Sanhita (BNS, 2023).

For every legal query:
1. Identify the relevant IPC section(s) and their BNS equivalents
2. Explain what the law says in plain English
3. Mention key changes between IPC and BNS
4. State the severity of punishment
5. Give practical advice (file FIR, consult lawyer, preserve evidence, etc.)

Format your response clearly with sections. Be accurate — cite only real sections.
End your response with a JSON block on its own line:
SECTIONS_JSON: {"ipc": ["302", "34"], "bns": ["103", "3(5)"], "crime_described": true}

crime_described should be true if the user described a crime or incident that could lead to an FIR.
"""

# ---------------------------------------------------------------------------
# Scheme Checker Agent (Placeholder with dummy data)
# ---------------------------------------------------------------------------
SCHEME_SYSTEM = """You are a government scheme eligibility assistant for India. 

You have knowledge of major central and state government schemes. For queries about schemes:
1. Identify relevant schemes
2. Explain eligibility criteria
3. Describe benefits
4. Explain how to apply

IMPORTANT: Since this is a demo, use these placeholder schemes as examples:
- PM Awas Yojana (housing for BPL families)
- Ayushman Bharat (health insurance up to 5 lakh)
- PM Kisan Samman Nidhi (6000/year for farmers)
- Beti Bachao Beti Padhao (girl child welfare)
- PM Ujjwala Yojana (LPG for BPL women)

Mention that scheme data is being updated and to verify at official portals.
"""

# ---------------------------------------------------------------------------
# FIR Need Detector
# ---------------------------------------------------------------------------
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
"""


# ---------------------------------------------------------------------------
# Helper: call LLM
# ---------------------------------------------------------------------------
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
        # Try to find JSON object in text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def route_query(query: str, history: list[dict]) -> dict:
    """Route the query to appropriate agents."""
    context = ""
    if history:
        last = history[-3:] if len(history) > 3 else history
        context = "\n".join(
            [f"{m['role'].upper()}: {m['content'][:200]}" for m in last]
        )
        context = f"Recent conversation:\n{context}\n\n"

    raw = llm_call(
        ROUTER_PROMPT, f"{context}Current query: {query}", max_tokens=200, temp=0
    )
    result = extract_json(raw)
    if not result:
        return {
            "agents": ["ipc_bns"],
            "needs_fir": False,
            "language": "en",
            "reasoning": "fallback",
        }
    return result


# ---------------------------------------------------------------------------
# IPC/BNS Agent
# ---------------------------------------------------------------------------
def run_ipc_bns_agent(query: str, history: list[dict]) -> dict:
    """Run the IPC/BNS comparator agent."""
    messages = [{"role": "system", "content": IPC_BNS_SYSTEM}]

    # Add relevant history
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content.strip()

    # Extract sections JSON if present
    sections_data = {"ipc": [], "bns": [], "crime_described": False}
    if "SECTIONS_JSON:" in raw:
        parts = raw.split("SECTIONS_JSON:")
        main_text = parts[0].strip()
        sections_raw = parts[1].strip() if len(parts) > 1 else "{}"
        parsed = extract_json(sections_raw)
        if parsed:
            sections_data = parsed
    else:
        main_text = raw

    return {
        "agent": "ipc_bns",
        "response": main_text,
        "sections": sections_data,
    }


# ---------------------------------------------------------------------------
# Scheme Agent (Placeholder)
# ---------------------------------------------------------------------------
def run_scheme_agent(query: str, history: list[dict]) -> dict:
    """Run the scheme eligibility checker (placeholder with dummy data)."""
    messages = [{"role": "system", "content": SCHEME_SYSTEM}]

    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )

    return {
        "agent": "scheme",
        "response": resp.choices[0].message.content.strip(),
        "note": "Scheme data is placeholder — full integration coming soon.",
    }


# ---------------------------------------------------------------------------
# FIR Recommendation Detector
# ---------------------------------------------------------------------------
def should_recommend_fir(query: str, agent_response: str) -> dict:
    """Determine if we should show the FIR drafting button."""
    combined = f"User query: {query}\n\nAgent response summary: {agent_response[:500]}"
    raw = llm_call(FIR_DETECTION_PROMPT, combined, max_tokens=150, temp=0)
    result = extract_json(raw)
    if not result:
        return {"recommend_fir": False, "reason": "Could not determine"}
    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def process_query(query: str, history: list[dict], language: str = "en") -> dict:
    """
    Main pipeline:
    1. Route query to agents
    2. Run appropriate agents
    3. Detect if FIR should be recommended
    4. Return structured response

    Returns:
    {
        "responses": [{"agent": str, "response": str, ...}],
        "recommend_fir": bool,
        "fir_reason": str,
        "agents_used": [str],
        "needs_fir": bool,  # user explicitly asked for FIR
        "language": str,
    }
    """
    # Step 1: Route
    routing = route_query(query, history)
    agents = routing.get("agents", ["ipc_bns"])
    needs_fir = routing.get("needs_fir", False)
    lang = routing.get("language", language)

    # Step 2: Run agents
    responses = []
    crime_described = False

    for agent in agents:
        if agent == "ipc_bns":
            result = run_ipc_bns_agent(query, history)
            responses.append(result)
            if result.get("sections", {}).get("crime_described"):
                crime_described = True
        elif agent == "scheme":
            result = run_scheme_agent(query, history)
            responses.append(result)

    # Step 3: FIR recommendation (only if not explicitly requested)
    recommend_fir = False
    fir_reason = ""

    if not needs_fir and crime_described and responses:
        # Use the first IPC/BNS response for detection
        ipc_response = next((r for r in responses if r["agent"] == "ipc_bns"), None)
        if ipc_response:
            fir_check = should_recommend_fir(query, ipc_response["response"])
            recommend_fir = fir_check.get("recommend_fir", False)
            fir_reason = fir_check.get("reason", "")

    return {
        "responses": responses,
        "recommend_fir": recommend_fir,
        "fir_reason": fir_reason,
        "agents_used": agents,
        "needs_fir": needs_fir,
        "language": lang,
        "routing_reason": routing.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Nyaya Sahayak — Pipeline Test")
    test_queries = [
        "My neighbor stole my mobile phone yesterday. What sections apply?",
        "What is IPC 302?",
        "Am I eligible for PM Awas Yojana?",
    ]

    history = []
    for q in test_queries:
        print(f"\nQ: {q}")
        result = process_query(q, history)
        print(f"Agents: {result['agents_used']}")
        for r in result["responses"]:
            print(f"\n[{r['agent'].upper()}]: {r['response'][:300]}...")
        print(f"Recommend FIR: {result['recommend_fir']}")
        history.append({"role": "user", "content": q})
        if result["responses"]:
            history.append(
                {"role": "assistant", "content": result["responses"][0]["response"]}
            )
