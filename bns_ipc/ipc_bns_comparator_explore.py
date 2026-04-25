# Databricks notebook source
import pandas as pd
import requests
from io import StringIO

# Same source as Nyaya Dhwani's ingestion notebook — Apache-2.0 licensed
HF_MAPPING_CSV = (
    "https://huggingface.co/datasets/nandhakumarg/IPC_and_BNS_transformation/"
    "resolve/main/IPC%20and%20BNS%20transformation%20.csv"
)

r = requests.get(HF_MAPPING_CSV, timeout=60)
r.raise_for_status()
df_raw = pd.read_csv(StringIO(r.text))

print(f"Rows: {len(df_raw)}")
print(f"Columns: {list(df_raw.columns)}")
df_raw.head(5)

# COMMAND ----------

import ast
import pandas as pd

def _parse_hf_row(raw_dict_str):
    """Convert the stringified dict in the 'response' column into real fields."""
    try:
        d = ast.literal_eval(raw_dict_str)
    except Exception:
        return None
    return {
        "ipc_section":     str(d.get("IPC Section", "")).strip(),
        "ipc_heading":     str(d.get("IPC Heading", "")).strip(),
        "ipc_description": str(d.get("IPC Descriptions", "") or d.get("IPC Description", "")).strip(),
        "bns_section":     str(d.get("BNS Section", "")).strip(),
        "bns_heading":     str(d.get("BNS Heading", "")).strip(),
        "bns_description": str(d.get("BNS description", "") or d.get("BNS Description", "")).strip(),
    }

# Parse every row
parsed = [_parse_hf_row(r) for r in df_raw["response"].tolist()]
parsed = [p for p in parsed if p is not None]
df_map = pd.DataFrame(parsed)

# Derive a status column so we can filter Mapped vs Repealed vs New
def _derive_status(row):
    ipc = row["ipc_section"]
    bns = row["bns_section"]
    ipc_h = row["ipc_heading"].lower()
    bns_h = row["bns_heading"].lower()
    if "repealed" in bns.lower() or "repealed" in bns_h:
        return "Repealed"
    if not ipc or ipc.lower() in ("none", "nan", ""):
        return "New"      # New BNS provision, no IPC equivalent
    if not bns or bns.lower() in ("none", "nan", ""):
        return "Dropped"  # IPC section with no BNS equivalent
    return "Mapped"

df_map["status"] = df_map.apply(_derive_status, axis=1)

print(f"Total rows: {len(df_map)}")
print(f"\nStatus breakdown:")
print(df_map["status"].value_counts())
print(f"\nSample (Mapped):")
display(df_map[df_map["status"] == "Mapped"].head(3))
print(f"\nSample (New, if any):")


# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "default"
MAPPING_TABLE = f"{CATALOG}.{SCHEMA}.ipc_bns_mapping"

# Verify the table exists and is healthy
print(f"Table: {MAPPING_TABLE}")
print(f"Row count: {spark.table(MAPPING_TABLE).count()}")
print(f"\nSchema:")
spark.table(MAPPING_TABLE).printSchema()
print(f"\nStatus breakdown:")
display(spark.sql(f"SELECT status, COUNT(*) as count FROM {MAPPING_TABLE} GROUP BY status"))

# COMMAND ----------

# MAGIC %md
# MAGIC Step 4: Level 1 — exact section lookup (no LLM, just pandas)
# MAGIC Time to build your first working feature. This takes an IPC section like "302" and returns the full mapping. Instant, deterministic, no LLM tokens burned, no hallucination risk.

# COMMAND ----------

# ============================================================
# Exhaustive test suite for Level 1 lookup
# ============================================================

def run_test(name, fn, expected_fn=None):
    """Pretty-print a test. expected_fn(result) returns (bool, message)."""
    try:
        result = fn()
        if expected_fn:
            ok, msg = expected_fn(result)
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {status}  {name}")
            if not ok:
                print(f"         → {msg}")
                print(f"         got: {result}")
        else:
            print(f"  •       {name}: {result}")
    except Exception as e:
        print(f"  ✗ ERROR {name}: {type(e).__name__}: {e}")


# ============================================================
# SUITE 1: Landmark sections — sanity check well-known mappings
# ============================================================
print("\n" + "=" * 70)
print("SUITE 1: Landmark IPC sections")
print("=" * 70)

# Reference table: well-known IPC→BNS mappings from official MHA sources
landmarks = [
    ("302", "103", "Murder"),
    ("307", "109", "Attempt to murder"),
    ("375", "63",  "Rape (definition)"),
    ("376", "64",  "Punishment for rape"),
    ("378", "303", "Theft (definition)"),
    ("379", "303", "Punishment for theft"),
    ("420", "318", "Cheating"),
    ("498A", "85", "Cruelty by husband/in-laws"),
    ("124A", None, "Sedition (contentious — may be replaced by BNS 152 or removed)"),
    ("499", "356", "Defamation"),
]

for ipc, expected_bns, what in landmarks:
    r = lookup_ipc(ipc)
    if r is None:
        print(f"  ✗ NOT FOUND  IPC {ipc} ({what})")
        continue
    got_bns = r["bns_section"]
    # Use startswith because dataset has "318(4)" while we expect "318"
    match = expected_bns is None or got_bns.startswith(expected_bns)
    marker = "✓" if match else "✗"
    expected_str = expected_bns or "?"
    print(f"  {marker}  IPC {ipc:5} → BNS {got_bns:10} (expected ~{expected_str:5}) — {what}")


# ============================================================
# SUITE 2: Input-format robustness
# ============================================================
print("\n" + "=" * 70)
print("SUITE 2: Input-format normalization (all should resolve to IPC 302)")
print("=" * 70)

format_variants = [
    "302",
    "IPC 302",
    "ipc 302",
    "Ipc 302",
    "IPC  302",          # double space
    "Section 302",
    "section 302",
    "SECTION 302",
    "302 IPC",           # trailing IPC (not yet supported)
    "302 of IPC",
    "302 OF IPC",
    "  302  ",           # surrounding whitespace
    "\t302\n",           # tab/newline
    "IPC Section 302",
    "Section 302 IPC",
]

expected_bns = lookup_ipc("302")["bns_section"]
for variant in format_variants:
    r = lookup_ipc(variant)
    if r is None:
        print(f"  ✗ FAIL   {variant!r:25} → not found")
    elif r["bns_section"] == expected_bns:
        print(f"  ✓        {variant!r:25} → BNS {r['bns_section']}")
    else:
        print(f"  ✗ WRONG  {variant!r:25} → BNS {r['bns_section']} (expected {expected_bns})")


# ============================================================
# SUITE 3: Edge cases — invalid / weird inputs
# ============================================================
print("\n" + "=" * 70)
print("SUITE 3: Invalid inputs (should gracefully return None)")
print("=" * 70)

edge_cases = [
    "",                  # empty string
    " ",                 # whitespace only
    "9999",              # valid format, nonexistent section
    "abc",               # non-numeric garbage
    "302.5",             # decimal (not a real section)
    "IPC",               # prefix only
    "murder",            # a concept, not a section number
    "-1",                # negative
    "0",                 # zero
    "302;DROP TABLE",    # basic injection attempt (we're pandas, but still)
]

for q in edge_cases:
    try:
        r = lookup_ipc(q)
        status = "None ✓" if r is None else f"unexpectedly matched → {r['bns_section']}"
        print(f"  {q!r:25} → {status}")
    except Exception as e:
        print(f"  {q!r:25} → EXCEPTION {type(e).__name__}: {e}")


# ============================================================
# SUITE 4: Sub-sections (IPC 354A, 354B, etc.)
# ============================================================
print("\n" + "=" * 70)
print("SUITE 4: Sub-section handling (letters after number)")
print("=" * 70)

subsection_tests = ["354A", "354B", "354C", "354D", "498A", "509"]
for q in subsection_tests:
    r = lookup_ipc(q)
    if r:
        print(f"  ✓  IPC {q:6} → BNS {r['bns_section']:10} — {r['ipc_heading'][:50]}")
    else:
        print(f"  ✗  IPC {q:6} → NOT FOUND")


# ============================================================
# SUITE 5: Reverse lookup — find consolidations
# ============================================================
print("\n" + "=" * 70)
print("SUITE 5: Reverse lookup — which BNS sections absorbed multiple IPC sections?")
print("=" * 70)

# Find BNS sections that appear as the target of multiple IPC mappings
bns_counts = df_mapping[df_mapping["status"] == "Mapped"]["bns_section"].value_counts()
consolidations = bns_counts[bns_counts > 1].head(10)

if consolidations.empty:
    print("  (No BNS section absorbs multiple IPC sections — this is a strictly 1:1 dataset)")
else:
    print(f"  Top BNS sections absorbing multiple IPC sections:\n")
    for bns_sec, count in consolidations.items():
        sources = lookup_bns(bns_sec)
        ipc_list = ", ".join(s["ipc_section"] for s in sources[:5])
        more = f" (+{len(sources)-5} more)" if len(sources) > 5 else ""
        print(f"  BNS {bns_sec:8} ← {count} IPC sections: {ipc_list}{more}")


# ============================================================
# SUITE 6: Coverage summary
# ============================================================
print("\n" + "=" * 70)
print("SUITE 6: Dataset coverage summary")
print("=" * 70)

total = len(df_mapping)
mapped = (df_mapping["status"] == "Mapped").sum()
repealed = (df_mapping["status"] == "Repealed").sum()
unique_ipc = df_mapping["ipc_section"].nunique()
unique_bns = df_mapping[df_mapping["status"] == "Mapped"]["bns_section"].nunique()

print(f"  Total rows:              {total}")
print(f"  Mapped:                  {mapped}")
print(f"  Repealed:                {repealed}")
print(f"  Unique IPC sections:     {unique_ipc}")
print(f"  Unique BNS sections:     {unique_bns}")
print(f"  Consolidation ratio:     {mapped / unique_bns:.2f} IPC sections per BNS section")

# COMMAND ----------

import re

def normalize_section(s: str) -> str:
    """Strip whitespace, remove 'IPC'/'BNS'/'Section' prefixes/suffixes."""
    s = str(s).strip().upper()
    
    # Remove trailing suffixes (ORDER MATTERS: longest/most-specific first)
    s = re.sub(r"\s+OF\s+(IPC|BNS)$", "", s)   # "302 OF IPC" → "302"
    s = re.sub(r"\s+(IPC|BNS)$", "", s)         # "302 IPC"    → "302"
    
    # Remove leading prefixes
    s = re.sub(r"^(IPC|BNS)\s+SECTION\s+", "", s)  # "IPC SECTION 302" → "302"
    s = re.sub(r"^(IPC|BNS)\s+", "", s)             # "IPC 302"         → "302"
    s = re.sub(r"^SECTION\s+", "", s)               # "SECTION 302"     → "302"
    
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- Retest the full format suite ----------
print("Variants that should all resolve to IPC 302 → BNS 103:\n")
test_variants = [
    "302",
    "IPC 302",
    "ipc 302",
    "Ipc 302",
    "IPC  302",
    "Section 302",
    "section 302",
    "SECTION 302",
    "302 IPC",                # was failing
    "302 of IPC",
    "302 OF IPC",
    "  302  ",
    "\t302\n",
    "IPC Section 302",
    "Section 302 IPC",        # was failing
    "Section 302 of IPC",     # was failing
]

for q in test_variants:
    normalized = normalize_section(q)
    r = lookup_ipc(q)
    status = f"BNS {r['bns_section']}" if r else "NOT FOUND"
    marker = "✓" if r and r["bns_section"] == "103" else "✗"
    print(f"  {marker}  {q!r:30} → {normalized!r:12} → {status}")

# COMMAND ----------

# Test what the starter notebook showed — Databricks-hosted Claude Sonnet 4
%pip install -q databricks-langchain
dbutils.library.restartPython()

# COMMAND ----------

import os

# Paste your Groq key here — then clear this cell's output after running
os.environ.setdefault("GROQ_API_KEY", "")  # set GROQ_API_KEY in your environment

# Verify it's set (without printing the key itself)
key = os.environ.get("GROQ_API_KEY", "")
if key.startswith("gsk_") and len(key) > 20:
    print(f"✓ Key loaded (prefix: {key[:8]}..., length: {len(key)})")
else:
    print("✗ Key doesn't look right — should start with 'gsk_'")

# COMMAND ----------

# MAGIC %pip install -q groq
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import re
import ast
import pandas as pd
import requests
from io import StringIO

# ---- Re-set the Groq key ----
# GROQ_API_KEY already set in environment

# ---- Re-load mapping from Delta (already saved earlier) ----
CATALOG = "workspace"
SCHEMA = "default"
MAPPING_TABLE = f"{CATALOG}.{SCHEMA}.ipc_bns_mapping"
df_mapping = spark.table(MAPPING_TABLE).toPandas()
print(f"Loaded {len(df_mapping)} mapping rows from {MAPPING_TABLE}")

# ---- Re-define normalize_section and lookup functions ----
def normalize_section(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"\s+OF\s+(IPC|BNS)$", "", s)
    s = re.sub(r"\s+(IPC|BNS)$", "", s)
    s = re.sub(r"^(IPC|BNS)\s+SECTION\s+", "", s)
    s = re.sub(r"^(IPC|BNS)\s+", "", s)
    s = re.sub(r"^SECTION\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lookup_ipc(query: str):
    q = normalize_section(query)
    hits = df_mapping[df_mapping["ipc_section"] == q]
    return hits.iloc[0].to_dict() if not hits.empty else None

def lookup_bns(query: str):
    q = normalize_section(query)
    hits = df_mapping[df_mapping["bns_section"] == q]
    return hits.to_dict(orient="records")

print("✓ Functions redefined")

# COMMAND ----------

from groq import Groq

client = Groq()  # reads GROQ_API_KEY from env automatically

# Minimal ping to confirm end-to-end connectivity
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    temperature=0,
    max_tokens=10,
)
print("Response:", response.choices[0].message.content)
print("Latency (approx):", response.usage.total_time if hasattr(response.usage, 'total_time') else 'n/a')
print("Tokens:", response.usage.total_tokens)

# COMMAND ----------

import json
from groq import Groq

# Reuse one client across all calls
_groq_client = Groq()
ROUTER_MODEL = "llama-3.3-70b-versatile"

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


def llm_route(query: str) -> dict:
    """Classify a query using Groq. Returns dict with mode/section/code/reasoning."""
    if not query or not query.strip():
        return {"mode": "invalid", "section": None, "code": None, "reasoning": "empty query"}
    
    try:
        response = _groq_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query.strip()},
            ],
            temperature=0,           # deterministic — same query → same classification
            max_tokens=150,
            response_format={"type": "json_object"},  # force JSON output
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        
        # Sanity-check the fields
        if parsed.get("mode") not in {"A_exact_lookup", "B_concept_search", "C_scenario"}:
            parsed["mode"] = "unknown"
        return parsed
    
    except json.JSONDecodeError as e:
        return {"mode": "error", "section": None, "code": None, "reasoning": f"JSON parse failed: {e}"}
    except Exception as e:
        return {"mode": "error", "section": None, "code": None, "reasoning": f"{type(e).__name__}: {e}"}


# ============================================================
# Test suite
# ============================================================

test_queries = [
    # Mode A — clean section lookups
    "302",
    "IPC 302",
    "what is section 420",
    "tell me about IPC 498A",
    "BNS 103",
    "explain 354A IPC",
    
    # Mode B — concept searches
    "murder",
    "theft",
    "cyber fraud",
    "sexual harassment",
    "organized crime",
    "defamation",
    
    # Mode C — scenarios
    "someone hacked my account and stole money",
    "my neighbor is threatening me with a knife",
    "I was cheated by an online seller who took my money and disappeared",
    "what happens if a police officer arrests me without telling me the reason",
    
    # Tricky / ambiguous
    "what is IPC 302",              # has section + is a sentence (should be A)
    "tell me about cheating",       # concept disguised as sentence (should be B)
    "is 302 the same as BNS 103",   # comparison — could be A
    "",                              # empty
]

print(f"{'Mode':<22} {'Section':<10} {'Code':<6} Query")
print("=" * 100)
for q in test_queries:
    r = llm_route(q)
    mode = r.get("mode", "?")
    sec = str(r.get("section") or "-")
    code = str(r.get("code") or "-")
    print(f"  {mode:<20} {sec:<10} {code:<6} {q!r}")

# COMMAND ----------

def handle_query(query: str, *, explain: bool = True) -> dict:
    """
    Full pipeline: LLM router → dispatch → (optionally) LLM explanation.
    """
    routed = llm_route(query)
    mode = routed.get("mode")
    section = routed.get("section")
    code = routed.get("code") or "IPC"
    
    if mode == "A_exact_lookup":
        if not section:
            return {"mode": mode, "router": routed, "result": None,
                    "error": "Router chose Mode A but extracted no section number"}
        
        if code == "BNS":
            result = lookup_bns(section)
            result_type = "reverse_bns_lookup"
            explanation = None  # Reverse lookup returns a list — we'd need to explain each, skip for now
        else:
            result = lookup_ipc(section)
            result_type = "forward_ipc_lookup"
            explanation = explain_mapping(result) if (result and explain) else None
        
        return {
            "mode": mode,
            "router": routed,
            "result_type": result_type,
            "result": result,
            "explanation": explanation,
        }
    
    return {"mode": mode, "router": routed, "result": None,
            "note": f"Mode {mode} handler not yet implemented"}


# Quick smoke test
r = handle_query("IPC 498A")
print(f"Mode: {r['mode']}")
print(f"IPC {r['result']['ipc_section']} → BNS {r['result']['bns_section']}")
print(f"\nExplanation summary: {r['explanation']['summary']}")
print(f"Changed: {r['explanation']['what_changed']}")


# ============================================================
# End-to-end tests
# ============================================================

test_queries = [
    "302",                                  # Mode A — forward
    "what is section 420",                  # Mode A — forward, sentence form
    "IPC 498A",                             # Mode A — forward, sub-section
    "BNS 103",                              # Mode A — reverse
    "BNS 127",                              # Mode A — reverse, should return multiple
    "9999",                                 # Mode A but nonexistent
    "murder",                               # Mode B stub
    "someone hacked my account",            # Mode C stub
]

# for q in test_queries:
#     print(f"\n{'=' * 70}")
#     print(f"Query: {q!r}")
#     print('=' * 70)
#     r = handle_query(q)
    
#     print(f"Mode:    {r['mode']}")
#     print(f"Router:  section={r['router'].get('section')}, code={r['router'].get('code')}")
#     print(f"Reason:  {r['router'].get('reasoning', '-')}")
    
#     if r.get("result"):
#         result = r["result"]
#         if isinstance(result, dict):
#             print(f"\nIPC {result['ipc_section']}: {result['ipc_heading']}")
#             print(f"  → BNS {result['bns_section']}: {result['bns_heading']}")
#             print(f"  Status: {result['status']}")
#         elif isinstance(result, list):
#             print(f"\nBNS {r['router']['section']} absorbs {len(result)} IPC section(s):")
#             for item in result[:5]:
#                 print(f"  - IPC {item['ipc_section']}: {item['ipc_heading']}")
#             if len(result) > 5:
#                 print(f"  ... and {len(result) - 5} more")
#     elif r.get("note"):
#         print(f"Note:    {r['note']}")
#     elif r.get("error"):
#         print(f"Error:   {r['error']}")
#     else:
#         print("Result:  None (not found)")

# COMMAND ----------

# import json

# EXPLAINER_SYSTEM_PROMPT = """You are a legal explainer for Indian criminal law. You compare IPC (old, 1860) and BNS (new, 2023) sections for citizens and junior lawyers.

# Given the IPC and BNS texts for a section pair, produce a structured explanation.

# Rules:
# - Ground every claim in the provided texts. Do NOT invent section numbers, punishments, or provisions not in the texts.
# - Keep language clear and avoid legalese where possible.
# - If the texts are identical in substance, say so plainly — do not invent differences.
# - Be concise: 2-4 sentences per field.
# - For severity: "low" = fine or short imprisonment; "medium" = up to 7 years; "high" = 10+ years; "capital" = life imprisonment or death.

# Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
# {
#   "summary": "<one-sentence plain-English description of what this section is about>",
#   "what_changed": "<what's different between IPC and BNS, or 'No substantive change' if identical>",
#   "what_stayed": "<what's preserved between IPC and BNS>",
#   "practical_impact": "<what this means for someone facing this charge today>",
#   "severity": "low | medium | high | capital"
# }
# """


# def explain_mapping(mapping: dict) -> dict:
#     """Take a mapping row with full IPC + BNS text, return a structured LLM explanation."""
#     if not mapping:
#         return {"error": "No mapping provided"}
    
#     # Truncate descriptions to keep prompt size reasonable (legal text can be huge)
#     MAX_DESC_LEN = 3000
#     ipc_desc = (mapping.get("ipc_description") or "")[:MAX_DESC_LEN]
#     bns_desc = (mapping.get("bns_description") or "")[:MAX_DESC_LEN]
    
#     prompt = f"""IPC Section {mapping['ipc_section']} — {mapping['ipc_heading']}

# Full IPC text:
# {ipc_desc}

# ---

# BNS Section {mapping['bns_section']} — {mapping['bns_heading']}

# Full BNS text:
# {bns_desc}

# ---

# Status of this mapping: {mapping['status']}

# Generate the structured explanation now."""

#     try:
#         response = _groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[
#                 {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0.2,
#             max_tokens=800,
#             response_format={"type": "json_object"},
#         )
#         raw = response.choices[0].message.content
#         parsed = json.loads(raw)
#         return parsed
#     except json.JSONDecodeError as e:
#         return {"error": f"JSON parse failed: {e}", "raw": raw if 'raw' in dir() else None}
#     except Exception as e:
#         return {"error": f"{type(e).__name__}: {e}"}


# # ---------- Test on three contrasting cases ----------
# test_cases = [
#     ("302", "Murder — classic case, same punishment, renumbered"),
#     ("420", "Cheating — may have restructured definitions"),
#     ("124A", "Sedition — politically contentious, BNS 152 redefines it"),
# ]

# for section, what in test_cases:
#     print(f"\n{'=' * 70}")
#     print(f"IPC {section} — {what}")
#     print('=' * 70)
    
#     mapping = lookup_ipc(section)
#     if not mapping:
#         print(f"  Mapping not found.")
#         continue
    
#     print(f"IPC {mapping['ipc_section']}: {mapping['ipc_heading']}")
#     print(f"  → BNS {mapping['bns_section']}: {mapping['bns_heading']}")
#     print(f"  Status: {mapping['status']}")
    
#     explanation = explain_mapping(mapping)
    
#     if "error" in explanation:
#         print(f"\nERROR: {explanation['error']}")
#         continue
    
#     print(f"\nSUMMARY:           {explanation.get('summary', '-')}")
#     print(f"WHAT CHANGED:      {explanation.get('what_changed', '-')}")
#     print(f"WHAT STAYED:       {explanation.get('what_stayed', '-')}")
#     print(f"PRACTICAL IMPACT:  {explanation.get('practical_impact', '-')}")
#     print(f"SEVERITY:          {explanation.get('severity', '-')}")

# COMMAND ----------

# Build the search corpus from the mapping Delta table
corpus_rows = []

for _, row in df_mapping.iterrows():
    # Skip rows that are entirely empty (defensive)
    if not (row["ipc_section"] or row["bns_section"]):
        continue
    
    # IPC-angle row (only if IPC section + text exist)
    if row["ipc_section"] and row["ipc_description"]:
        corpus_rows.append({
            "chunk_id": f"IPC_{row['ipc_section']}",
            "source_code": "IPC",
            "section": row["ipc_section"],
            "heading": row["ipc_heading"],
            "text": row["ipc_description"],
            "counterpart_code": "BNS",
            "counterpart_section": row["bns_section"],
            "counterpart_heading": row["bns_heading"],
            "status": row["status"],
        })
    
    # BNS-angle row (only if BNS section + text exist)
    if row["bns_section"] and row["bns_description"]:
        corpus_rows.append({
            "chunk_id": f"BNS_{row['bns_section']}_from_IPC_{row['ipc_section']}",
            "source_code": "BNS",
            "section": row["bns_section"],
            "heading": row["bns_heading"],
            "text": row["bns_description"],
            "counterpart_code": "IPC",
            "counterpart_section": row["ipc_section"],
            "counterpart_heading": row["ipc_heading"],
            "status": row["status"],
        })

df_corpus = pd.DataFrame(corpus_rows)

print(f"Total corpus rows: {len(df_corpus)}")
print(f"\nBreakdown by source:")
print(df_corpus["source_code"].value_counts())
print(f"\nText length stats (chars):")
print(df_corpus["text"].str.len().describe())
print(f"\nSample rows:")
display(df_corpus[["chunk_id", "source_code", "section", "heading"]].head(5))

# Save to Delta as our second artifact
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.ipc_bns_corpus"
spark.createDataFrame(df_corpus.astype(str)).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(CORPUS_TABLE)

print(f"\n✓ Saved corpus to {CORPUS_TABLE}")
print(f"✓ Row count from Delta: {spark.table(CORPUS_TABLE).count()}")

# COMMAND ----------

# MAGIC %pip install -q faiss-cpu sentence-transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ============================================================
# Post-restart restore cell — rebuilds everything from Delta + functions
# ============================================================

import os
import re
import json
import pandas as pd
from groq import Groq

# ---- Re-set Groq key ----
# GROQ_API_KEY already set in environment

# ---- Table config ----
CATALOG = "workspace"
SCHEMA = "default"
MAPPING_TABLE = f"{CATALOG}.{SCHEMA}.ipc_bns_mapping"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.ipc_bns_corpus"

# ---- Load both Delta tables into pandas ----
df_mapping = spark.table(MAPPING_TABLE).toPandas()
df_corpus = spark.table(CORPUS_TABLE).toPandas()
print(f"✓ Mapping: {len(df_mapping)} rows")
print(f"✓ Corpus:  {len(df_corpus)} rows")

# ---- Lookup functions ----
def normalize_section(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"\s+OF\s+(IPC|BNS)$", "", s)
    s = re.sub(r"\s+(IPC|BNS)$", "", s)
    s = re.sub(r"^(IPC|BNS)\s+SECTION\s+", "", s)
    s = re.sub(r"^(IPC|BNS)\s+", "", s)
    s = re.sub(r"^SECTION\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lookup_ipc(query):
    q = normalize_section(query)
    hits = df_mapping[df_mapping["ipc_section"] == q]
    return hits.iloc[0].to_dict() if not hits.empty else None

def lookup_bns(query):
    q = normalize_section(query)
    hits = df_mapping[df_mapping["bns_section"] == q]
    return hits.to_dict(orient="records")

# ---- Groq client ----
_groq_client = Groq()
ROUTER_MODEL = "llama-3.3-70b-versatile"

# ---- Router ----
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

def llm_route(query):
    if not query or not query.strip():
        return {"mode": "invalid", "section": None, "code": None, "reasoning": "empty query"}
    try:
        response = _groq_client.chat.completions.create(
            model=ROUTER_MODEL,
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
        return {"mode": "error", "section": None, "code": None, "reasoning": f"{type(e).__name__}: {e}"}

# ---- Level 2 explainer ----
EXPLAINER_SYSTEM_PROMPT = """You are a legal explainer for Indian criminal law. You compare IPC (old, 1860) and BNS (new, 2023) sections for citizens and junior lawyers.

Given the IPC and BNS texts for a section pair, produce a structured explanation.

Rules:
- Ground every claim in the provided texts. Do NOT invent section numbers, punishments, or provisions not in the texts.
- Keep language clear and avoid legalese where possible.
- If the texts are identical in substance, say so plainly — do not invent differences.
- Be concise: 2-4 sentences per field.
- For severity: "low" = fine or short imprisonment; "medium" = up to 7 years; "high" = 10+ years; "capital" = life imprisonment or death.

Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "summary": "<one-sentence plain-English description of what this section is about>",
  "what_changed": "<what's different between IPC and BNS, or 'No substantive change' if identical>",
  "what_stayed": "<what's preserved between IPC and BNS>",
  "practical_impact": "<what this means for someone facing this charge today>",
  "severity": "low | medium | high | capital"
}
"""

def explain_mapping(mapping):
    if not mapping:
        return {"error": "No mapping provided"}
    MAX_DESC_LEN = 3000
    ipc_desc = (mapping.get("ipc_description") or "")[:MAX_DESC_LEN]
    bns_desc = (mapping.get("bns_description") or "")[:MAX_DESC_LEN]
    prompt = f"""IPC Section {mapping['ipc_section']} — {mapping['ipc_heading']}

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
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

# ---- Verify FAISS + sentence-transformers import cleanly ----
import faiss
from sentence_transformers import SentenceTransformer
print(f"✓ FAISS: {faiss.__version__ if hasattr(faiss, '__version__') else 'imported'}")
print(f"✓ sentence-transformers imported")

print("\n✓ All state restored. Ready for Mode B index build.")

# COMMAND ----------

import numpy as np

# ---- Load the embedding model ----
# all-MiniLM-L6-v2: 384-dim, fast on CPU, strong for short legal text
# This is the same model Nyaya Dhwani uses — good precedent
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading {EMBED_MODEL_NAME}... (first time ~60s, cached after)")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
print(f"✓ Model loaded. Embedding dim: {embedder.get_sentence_embedding_dimension()}")

# ---- Build search text for each row ----
# We combine heading + text so both keyword ("murder") and nuanced ("forcing sexual act") queries work
def build_search_text(row):
    heading = str(row.get("heading") or "").strip()
    text = str(row.get("text") or "").strip()
    # Truncate long text — embedding models have token limits (~512 tokens ≈ 2000 chars)
    text = text[:2000]
    return f"{heading}. {text}" if heading else text

search_texts = df_corpus.apply(build_search_text, axis=1).tolist()
print(f"\n✓ Prepared {len(search_texts)} texts for embedding")
print(f"  Sample: {search_texts[0][:120]}...")

# ---- Encode all corpus rows ----
print(f"\nEmbedding {len(search_texts)} rows (this takes ~30-60 seconds)...")
embeddings = embedder.encode(
    search_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # L2-normalize so inner product = cosine similarity
)
print(f"✓ Embeddings shape: {embeddings.shape}")  # should be (1126, 384)

# ---- Build FAISS index ----
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product (= cosine, because we normalized)
index.add(embeddings.astype(np.float32))
print(f"\n✓ FAISS index built: {index.ntotal} vectors, {dim} dims")

# COMMAND ----------

def semantic_search(query: str, k: int = 10) -> list[dict]:
    """
    Embed the query, find top-k similar corpus rows via FAISS,
    and return enriched results with scores.
    """
    # Embed query using same model + normalization as corpus
    q_emb = embedder.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    
    # FAISS search: returns scores (cosine since normalized) + row indices
    scores, indices = index.search(q_emb, k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        row = df_corpus.iloc[int(idx)].to_dict()
        row["score"] = float(score)
        row["rank"] = rank
        results.append(row)
    
    return results


def mode_b_concept_search(query: str, k: int = 6) -> dict:
    """
    Mode B handler: given a concept/keyword, find the most relevant IPC and BNS
    sections semantically. Return a structured result with both sides.
    """
    raw_hits = semantic_search(query, k=k * 2)  # over-fetch, then rebalance
    
    # Separate IPC and BNS hits so we return a balanced view
    ipc_hits = [h for h in raw_hits if h["source_code"] == "IPC"][:k]
    bns_hits = [h for h in raw_hits if h["source_code"] == "BNS"][:k]
    
    # De-duplicate by section number (a section might appear in top-k multiple times
    # if it has a long text that splits into similar chunks — not our case, but defensive)
    def dedupe(hits):
        seen, out = set(), []
        for h in hits:
            key = (h["source_code"], h["section"])
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out
    
    return {
        "query": query,
        "ipc_matches": dedupe(ipc_hits),
        "bns_matches": dedupe(bns_hits),
        "total_raw_hits": len(raw_hits),
    }


# ============================================================
# Test Mode B on concept queries
# ============================================================

test_concepts = [
    "murder",
    "cyber fraud",
    "organized crime",
    "dowry",
    "sexual harassment at workplace",
    "forcing someone into a sexual act",  # nuanced — no "rape" keyword
    "stealing a mobile phone",             # should retrieve theft
    "sedition",
]

for query in test_concepts:
    print(f"\n{'=' * 70}")
    print(f"Query: {query!r}")
    print('=' * 70)
    
    result = mode_b_concept_search(query, k=3)
    
    print(f"\nTop IPC matches:")
    for h in result["ipc_matches"]:
        print(f"  [{h['score']:.3f}] IPC {h['section']:6} — {h['heading'][:60]}")
    
    print(f"\nTop BNS matches:")
    for h in result["bns_matches"]:
        print(f"  [{h['score']:.3f}] BNS {h['section']:6} — {h['heading'][:60]}")

# COMMAND ----------

# Filter out short/empty text rows that pollute search results
MIN_TEXT_LEN = 100

df_corpus_clean = df_corpus[df_corpus["text"].str.len() >= MIN_TEXT_LEN].reset_index(drop=True)

dropped = len(df_corpus) - len(df_corpus_clean)
print(f"Original corpus: {len(df_corpus)} rows")
print(f"Filtered corpus: {len(df_corpus_clean)} rows ({dropped} noise rows dropped)")
print(f"\nNew text-length stats:")
print(df_corpus_clean["text"].str.len().describe())

# Save the cleaned corpus back to Delta (overwrite)
spark.createDataFrame(df_corpus_clean.astype(str)).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(CORPUS_TABLE)
print(f"\n✓ Saved filtered corpus to {CORPUS_TABLE}")

# Rebuild search texts and embeddings on filtered corpus
def build_search_text(row):
    heading = str(row.get("heading") or "").strip()
    text = str(row.get("text") or "").strip()[:2000]
    return f"{heading}. {text}" if heading else text

search_texts_clean = df_corpus_clean.apply(build_search_text, axis=1).tolist()

print(f"\nRe-embedding {len(search_texts_clean)} rows...")
embeddings_clean = embedder.encode(
    search_texts_clean,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# Rebuild FAISS index
index = faiss.IndexFlatIP(embeddings_clean.shape[1])
index.add(embeddings_clean.astype(np.float32))

# Update df_corpus pointer so semantic_search uses the cleaned data
df_corpus = df_corpus_clean

print(f"\n✓ Rebuilt FAISS index: {index.ntotal} vectors")

# COMMAND ----------

# Map American → British for legal text (IPC/BNS uses British)
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
    "colour": "color",  # reverse-tolerant
}

def _normalize_spelling(text: str) -> str:
    """Convert American spellings to British (what legal text uses)."""
    t = text.lower()
    for us, uk in SPELLING_VARIANTS.items():
        t = re.sub(rf"\b{us}\b", uk, t)
        t = re.sub(rf"\b{uk}\b", uk, t)  # idempotent
    return t


def _keyword_boost(query: str, hits: list[dict], boost: float = 0.15) -> list[dict]:
    """Boost hits whose heading contains query words (spelling-normalized)."""
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by",
                 "is", "are", "was", "were", "be", "and", "or", "me", "my", "someone",
                 "what", "which", "how", "about", "tell"}
    
    query_normalized = _normalize_spelling(query)
    query_words = [
        w for w in re.findall(r"[a-zA-Z]+", query_normalized)
        if len(w) > 2 and w not in stopwords
    ]
    if not query_words:
        return hits
    
    boosted = []
    for h in hits:
        heading_normalized = _normalize_spelling(str(h.get("heading", "")))
        matches = sum(1 for w in query_words if w in heading_normalized)
        if matches > 0:
            new_score = min(1.0, h["score"] + boost * matches)
            h = {**h, "score": new_score, "_boosted": True, "_boost_matches": matches}
        else:
            h = {**h, "_boosted": False}
        boosted.append(h)
    
    return sorted(boosted, key=lambda x: x["score"], reverse=True)


def _ensure_counterparts(hits: list[dict], k: int) -> list[dict]:
    """
    For every IPC hit in top-k, make sure its BNS counterpart is present
    (and vice versa). This fixes cases like 'sedition' where IPC 124A is found
    but BNS 152 isn't semantically close.
    """
    # Separate by source
    ipc_in = {h["section"]: h for h in hits if h["source_code"] == "IPC"}
    bns_in = {h["section"]: h for h in hits if h["source_code"] == "BNS"}
    
    added_ipc, added_bns = [], []
    
    # For each top IPC hit, ensure its paired BNS section is in results
    for ipc_sec, h in list(ipc_in.items())[:k]:
        counterpart_bns = h.get("counterpart_section")
        if counterpart_bns and counterpart_bns not in bns_in:
            # Look it up in df_corpus
            paired = df_corpus[
                (df_corpus["source_code"] == "BNS") &
                (df_corpus["section"] == counterpart_bns)
            ]
            if not paired.empty:
                paired_row = paired.iloc[0].to_dict()
                paired_row["score"] = h["score"] * 0.9   # slight penalty — it rode in on its pair
                paired_row["_paired_from"] = f"IPC {ipc_sec}"
                added_bns.append(paired_row)
    
    # Same in reverse: for each top BNS hit, add its paired IPC
    for bns_sec, h in list(bns_in.items())[:k]:
        counterpart_ipc = h.get("counterpart_section")
        if counterpart_ipc and counterpart_ipc not in ipc_in:
            paired = df_corpus[
                (df_corpus["source_code"] == "IPC") &
                (df_corpus["section"] == counterpart_ipc)
            ]
            if not paired.empty:
                paired_row = paired.iloc[0].to_dict()
                paired_row["score"] = h["score"] * 0.9
                paired_row["_paired_from"] = f"BNS {bns_sec}"
                added_ipc.append(paired_row)
    
    return hits + added_ipc + added_bns


def mode_b_concept_search(query: str, k: int = 6) -> dict:
    """
    Improved Mode B with keyword boost + counterpart pairing.
    """
    # 1. Get raw FAISS hits (over-fetch for reranking headroom)
    raw_hits = semantic_search(query, k=k * 3)
    
    # 2. Apply keyword boost
    boosted = _keyword_boost(query, raw_hits)
    
    # 3. Ensure IPC↔BNS counterparts are included
    with_pairs = _ensure_counterparts(boosted, k=k)
    
    # 4. Split back into IPC/BNS, re-sort, dedupe, take top-k per side
    ipc_hits = sorted(
        [h for h in with_pairs if h["source_code"] == "IPC"],
        key=lambda x: x["score"], reverse=True
    )
    bns_hits = sorted(
        [h for h in with_pairs if h["source_code"] == "BNS"],
        key=lambda x: x["score"], reverse=True
    )
    
    def dedupe(hits):
        seen, out = set(), []
        for h in hits:
            key = (h["source_code"], h["section"])
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out
    
    return {
        "query": query,
        "ipc_matches": dedupe(ipc_hits)[:k],
        "bns_matches": dedupe(bns_hits)[:k],
    }


# ============================================================
# Retest the previously-failing queries + sanity check good ones
# ============================================================

retest = [
    "murder",                    # sanity
    "sedition",                  # was fixed by pairing
    "organized crime",           # target of Fix 1 (spelling)
    "cyber fraud",               # target of Fix 2 (noise filtering)
    "stealing a mobile phone",   # sanity
]

for query in retest:
    print(f"\n{'=' * 70}")
    print(f"Query: {query!r}")
    print('=' * 70)
    
    result = mode_b_concept_search(query, k=3)
    
    print(f"\nTop IPC matches:")
    for h in result["ipc_matches"]:
        tags = []
        if h.get("_boosted") and h.get("_boost_matches", 0) > 0:
            tags.append(f"kw+{h['_boost_matches']}")
        if h.get("_paired_from"):
            tags.append(f"paired←{h['_paired_from']}")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  [{h['score']:.3f}] IPC {h['section']:8} — {h['heading'][:55]}{tag_str}")
    
    print(f"\nTop BNS matches:")
    for h in result["bns_matches"]:
        tags = []
        if h.get("_boosted") and h.get("_boost_matches", 0) > 0:
            tags.append(f"kw+{h['_boost_matches']}")
        if h.get("_paired_from"):
            tags.append(f"paired←{h['_paired_from']}")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  [{h['score']:.3f}] BNS {h['section']:8} — {h['heading'][:55]}{tag_str}")

# COMMAND ----------

# Sanity check: is the spelling normalizer live?
print(f"'organized crime' normalized → {_normalize_spelling('organized crime')!r}")
print(f"Expected: 'organised crime'")
print()

# Direct test: does BNS 111 exist in the filtered corpus?
bns_111 = df_corpus[(df_corpus["source_code"] == "BNS") & (df_corpus["section"].str.startswith("111"))]
print(f"BNS 111 rows in filtered corpus: {len(bns_111)}")
if not bns_111.empty:
    for _, r in bns_111.iterrows():
        print(f"  {r['section']:10} — {r['heading']}")
        print(f"    text starts with: {str(r['text'])[:200]}...")
else:
    print("  ⚠️  BNS 111 got filtered OUT — that's why it's not in results!")

# COMMAND ----------

# Check if BNS 111 exists in the mapping table (upstream)
bns_111_in_mapping = df_mapping[df_mapping["bns_section"].str.contains("111", na=False, regex=False)]
print(f"BNS 111 entries in mapping table: {len(bns_111_in_mapping)}")
if not bns_111_in_mapping.empty:
    display(bns_111_in_mapping[["ipc_section", "ipc_heading", "bns_section", "bns_heading", "status"]].head())
else:
    print("Confirmed: BNS 111 is not in the dataset at all.")
    print("This is a dataset limitation, not a filter problem.")

# While we're at it, let's see what new BNS sections we're missing
# (BNS sections that have no IPC predecessor)
print("\nNew BNS sections with no IPC mapping (if any):")
new_bns = df_mapping[
    df_mapping["ipc_section"].isin(["", "None", "nan"]) |
    df_mapping["ipc_section"].isna()
]
print(f"Count: {len(new_bns)}")

# COMMAND ----------

# ============================================================
# Mode C: Scenario reasoning with RAG grounding
# ============================================================

SCENARIO_EXTRACTOR_PROMPT = """You analyze Indian criminal law scenarios and extract the underlying legal concepts.

Given a scenario, identify 2-4 distinct legal concepts or offense types that may apply. Be specific but concise. Use terminology that appears in Indian criminal statutes.

Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
{"concepts": ["<concept1>", "<concept2>", ...]}

Examples:
Scenario: "someone hacked my account and stole money"
{"concepts": ["cheating by personation", "identity theft", "dishonest misappropriation of property", "criminal breach of trust"]}

Scenario: "my neighbor is threatening me with a knife"
{"concepts": ["criminal intimidation", "assault with a deadly weapon", "threat to cause hurt"]}

Scenario: "I was cheated by an online seller who took my money and disappeared"
{"concepts": ["cheating", "dishonest inducement to deliver property", "cheating by personation"]}
"""


def extract_concepts(scenario: str) -> list[str]:
    """LLM-extract 2-4 legal concepts from a natural-language scenario."""
    try:
        response = _groq_client.chat.completions.create(
            model=ROUTER_MODEL,
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
        if isinstance(concepts, list):
            return [str(c).strip() for c in concepts if c]
    except Exception as e:
        print(f"[extract_concepts] error: {e}")
    return []


SCENARIO_REASONER_PROMPT = """You are a legal reasoning assistant for Indian criminal law. You compare IPC (old) and BNS (new) to help users understand what laws apply to their situation.

You will receive:
1. A real-life scenario from the user
2. A set of candidate IPC and BNS sections retrieved via semantic search

Your job: identify which sections are ACTUALLY applicable, explain why, and compare IPC vs BNS.

CRITICAL RULES:
- Only cite section numbers that appear in the "Candidate sections" below. Do NOT invent sections.
- If no candidate section clearly applies, say "no clear match in retrieved sections" rather than inventing one.
- Rank applicable sections by relevance (most relevant first).
- Note what changed between IPC and BNS for these sections.
- Add a clear disclaimer that this is informational, not legal advice.

Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "applicable_ipc_sections": [
    {"section": "<number>", "heading": "<from candidates>", "why_applicable": "<one sentence>"}
  ],
  "applicable_bns_sections": [
    {"section": "<number>", "heading": "<from candidates>", "why_applicable": "<one sentence>"}
  ],
  "what_changed": "<summary of IPC→BNS changes for this scenario>",
  "severity": "low | medium | high | capital",
  "practical_advice": "<2-3 sentence advice, e.g. file an FIR, preserve evidence>",
  "disclaimer": "This is general information, not legal advice. Consult a qualified lawyer for your specific situation."
}
"""


def mode_c_scenario(scenario: str, k_per_concept: int = 3) -> dict:
    """
    Mode C: scenario reasoning with RAG grounding.
    Pipeline: concept extraction → retrieval per concept → reasoning.
    """
    # Step 1: extract concepts
    concepts = extract_concepts(scenario)
    if not concepts:
        return {"error": "Could not extract legal concepts from the scenario"}
    
    # Step 2: retrieve candidate sections for each concept
    all_candidates = []
    seen_keys = set()
    for concept in concepts:
        hits = semantic_search(concept, k=k_per_concept * 2)
        for h in hits[:k_per_concept * 2]:
            key = (h["source_code"], h["section"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_candidates.append({
                "source": h["source_code"],
                "section": h["section"],
                "heading": h["heading"],
                "matched_concept": concept,
                "score": h["score"],
            })
    
    # Sort by score, keep top N overall
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = all_candidates[:20]  # cap context size
    
    # Step 3: format candidates for the LLM
    ipc_lines = [
        f"  IPC {c['section']} — {c['heading']}"
        for c in top_candidates if c["source"] == "IPC"
    ]
    bns_lines = [
        f"  BNS {c['section']} — {c['heading']}"
        for c in top_candidates if c["source"] == "BNS"
    ]
    
    candidates_block = (
        f"IPC candidates:\n" + "\n".join(ipc_lines) +
        f"\n\nBNS candidates:\n" + "\n".join(bns_lines)
    )
    
    user_msg = f"""Scenario: {scenario}

Extracted legal concepts: {', '.join(concepts)}

Candidate sections (retrieved via semantic search):
{candidates_block}

Now produce the structured legal analysis."""
    
    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
            "retrieved_candidates": top_candidates,
            "reasoning": reasoning,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "concepts": concepts, "candidates": top_candidates}


# ============================================================
# Test Mode C on realistic scenarios
# ============================================================

test_scenarios = [
    "someone hacked my account and stole money from my bank",
    "my neighbor is threatening me with a knife and says he will kill me",
    "I paid for a product online but the seller never sent it and blocked me",
]

for scenario in test_scenarios:
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {scenario!r}")
    print('=' * 70)
    
    result = mode_c_scenario(scenario)
    
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        continue
    
    print(f"\nExtracted concepts: {result['extracted_concepts']}")
    
    r = result["reasoning"]
    
    print(f"\nAPPLICABLE IPC SECTIONS:")
    for s in r.get("applicable_ipc_sections", []):
        print(f"  IPC {s['section']} — {s['heading']}")
        print(f"    Why: {s['why_applicable']}")
    
    print(f"\nAPPLICABLE BNS SECTIONS:")
    for s in r.get("applicable_bns_sections", []):
        print(f"  BNS {s['section']} — {s['heading']}")
        print(f"    Why: {s['why_applicable']}")
    
    print(f"\nWHAT CHANGED: {r.get('what_changed', '-')}")
    print(f"SEVERITY:     {r.get('severity', '-')}")
    print(f"ADVICE:       {r.get('practical_advice', '-')}")

# COMMAND ----------

def handle_query(query: str, *, explain: bool = True) -> dict:

    """

    Full pipeline: LLM router → dispatch to Mode A / B / C → return result.

    This is THE entry point for the UI tomorrow.

    """

    routed = llm_route(query)

    mode = routed.get("mode")

    section = routed.get("section")

    code = routed.get("code") or "IPC"

    

    # ---- Mode A ----

    if mode == "A_exact_lookup":

        if not section:

            return {"mode": mode, "router": routed, "result": None,

                    "error": "Router chose Mode A but extracted no section number"}

        

        if code == "BNS":

            result = lookup_bns(section)

            return {"mode": mode, "router": routed,

                    "result_type": "reverse_bns_lookup", "result": result}

        else:

            result = lookup_ipc(section)

            explanation = explain_mapping(result) if (result and explain) else None

            return {"mode": mode, "router": routed,

                    "result_type": "forward_ipc_lookup",

                    "result": result, "explanation": explanation}

    

    # ---- Mode B ----

    if mode == "B_concept_search":

        return {"mode": mode, "router": routed,

                "result": mode_b_concept_search(query, k=5)}

    

    # ---- Mode C ----

    if mode == "C_scenario":

        return {"mode": mode, "router": routed,

                "result": mode_c_scenario(query)}

    

    return {"mode": mode, "router": routed, "result": None,

            "note": "Unclassified query"}

# ============================================================

# End-to-end smoke test

# ============================================================

demo_queries = [

    "302",                                           # Mode A number

    "What section is IPC 498A under BNS",            # Mode A sentence

    "murder",                                         # Mode B concept

    "someone hacked my account and stole money",     # Mode C scenario

]

for q in demo_queries:

    print(f"\n{'=' * 60}")

    print(f">>> Query: {q!r}")

    print('=' * 60)

    r = handle_query(q)

    print(f"Mode chosen: {r['mode']}")

    if r["mode"] == "A_exact_lookup" and r.get("result"):

        if isinstance(r["result"], dict):

            print(f"IPC {r['result']['ipc_section']} → BNS {r['result']['bns_section']}")

            if r.get("explanation"):

                print(f"Summary: {r['explanation'].get('summary', '-')}")

    elif r["mode"] == "B_concept_search":

        top_ipc = r["result"]["ipc_matches"][0] if r["result"]["ipc_matches"] else None

        top_bns = r["result"]["bns_matches"][0] if r["result"]["bns_matches"] else None

        if top_ipc:

            print(f"Top IPC: {top_ipc['section']} — {top_ipc['heading']}")

        if top_bns:

            print(f"Top BNS: {top_bns['section']} — {top_bns['heading']}")

    elif r["mode"] == "C_scenario":

        concepts = r["result"].get("extracted_concepts", [])

        print(f"Concepts: {concepts}")

        reasoning = r["result"].get("reasoning", {})

        ipc = reasoning.get("applicable_ipc_sections", [])

        bns = reasoning.get("applicable_bns_sections", [])

        print(f"Applicable IPC: {[s['section'] for s in ipc]}")

        print(f"Applicable BNS: {[s['section'] for s in bns]}")

print(f"\n{'=' * 60}")

print("✅ End-to-end pipeline test complete")

# COMMAND ----------

# ============================================================
# Persist FAISS index + embedder state to disk for tomorrow
# ============================================================
import pickle

# Save FAISS index
FAISS_PATH = "/tmp/ipc_bns_faiss.index"
faiss.write_index(index, FAISS_PATH)
print(f"✓ Saved FAISS index → {FAISS_PATH}")

# Save embeddings (so we can rebuild even if FAISS file breaks)
EMB_PATH = "/tmp/ipc_bns_embeddings.npy"
np.save(EMB_PATH, embeddings_clean)
print(f"✓ Saved embeddings → {EMB_PATH}")

# Verify everything can be reloaded
test_index = faiss.read_index(FAISS_PATH)
assert test_index.ntotal == index.ntotal, "FAISS reload mismatch!"
print(f"✓ Reload verified: {test_index.ntotal} vectors")

print(f"\n📦 State saved. Your Delta tables are already persisted in Unity Catalog.")
print(f"    Mapping table: {MAPPING_TABLE}")
print(f"    Corpus table:  {CORPUS_TABLE}")

# COMMAND ----------

# Upload ipc_bns_comparator.py to your Databricks workspace,
# then in a notebook cell:
import sys
sys.path.insert(0, "/Workspace/Users/da24b007@smail.iitm.ac.in/Legal-Bot/bns_ipc/ipc_bns_comparator.py")  # wherever you uploaded it

from ipc_bns_comparator import Comparator

df_mapping = spark.table("workspace.default.ipc_bns_mapping").toPandas()
df_corpus  = spark.table("workspace.default.ipc_bns_corpus").toPandas()

comp = Comparator(df_mapping=df_mapping, df_corpus=df_corpus)

# Quick test all three modes
print(comp.handle_query("302")["mode"])           # A_exact_lookup
print(comp.handle_query("murder")["mode"])        # B_concept_search
print(comp.handle_query("someone hacked me")["mode"])  # C_scenario

# COMMAND ----------

import os, sys
os.environ["SARVAM_API_KEY"] = "sk_ay2lvnsu_2akm7JIIYCW8gkP7pahO2qEe"   # paste, clear output
sys.path.insert(0, "/Workspace/Users/<your-email>/")  # wherever language.py lives

import language as lg

# Text translation both ways
print(lg.to_english("मेरे पड़ोसी ने मुझे चाकू से धमकी दी", "hi"))
print(lg.from_english("what is your name", "hi"))

# Test TTS by length only — don't need to play the audio in notebook
audio = lg.english_to_speech("Theft is punishable under BNS Section 303.", "hi")
print(f"TTS audio: {len(audio):,} bytes WAV")

# COMMAND ----------



# COMMAND ----------

# In a Databricks notebook cell (not the app):
import os

# Pick a path inside the App's working directory.
# Apps deployed via `databricks apps deploy` typically have a `data/` folder.
EXPORT_DIR = "/Workspace/Users/da24b007@smail.iitm.ac.in/Legal-Bot/data"
os.makedirs(EXPORT_DIR, exist_ok=True)

spark.table("workspace.default.ipc_bns_mapping").toPandas().to_parquet(
    f"{EXPORT_DIR}/ipc_bns_mapping.parquet"
)
spark.table("workspace.default.ipc_bns_corpus").toPandas().to_parquet(
    f"{EXPORT_DIR}/ipc_bns_corpus.parquet"
)
spark.table("workspace.default.indian_criminal_case_summaries_light").toPandas().to_parquet(
    f"{EXPORT_DIR}/case_summaries.parquet"
)

# Verify
import os
for f in os.listdir(EXPORT_DIR):
    size_mb = os.path.getsize(f"{EXPORT_DIR}/{f}") / 1e6
    print(f"  {f}: {size_mb:.1f} MB")

# COMMAND ----------

EXPORT_DIR = "dbfs:/FileStore/legal_bot_data"

# COMMAND ----------

EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"

# COMMAND ----------

spark.table("workspace.default.indian_criminal_case_summaries_light") \
    .write.mode("overwrite") \
    .parquet(f"{EXPORT_DIR}/indian_criminal_case_summaries_light")

print("Done")

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT READ VOLUME ON VOLUME workspace.default.legal_bot_data TO `sivasubramanians2006@gmail.com`;

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT WRITE VOLUME ON VOLUME workspace.default.legal_bot_data TO `sivasubramanians2006@gmail.com`;

# COMMAND ----------

# In a Databricks notebook
import os
EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"
os.makedirs(EXPORT_DIR, exist_ok=True)

# pandas .to_parquet writes ONE file at the exact path
spark.table("workspace.default.ipc_bns_mapping").toPandas() \
    .to_parquet(f"{EXPORT_DIR}/ipc_bns_mapping.parquet", index=False)

spark.table("workspace.default.ipc_bns_corpus").toPandas() \
    .to_parquet(f"{EXPORT_DIR}/ipc_bns_corpus.parquet", index=False)

spark.table("workspace.default.indian_criminal_case_summaries_light").toPandas() \
    .to_parquet(f"{EXPORT_DIR}/case_summaries.parquet", index=False)

# Verify they're real single files, not directories
import os
for f in ["ipc_bns_mapping.parquet", "ipc_bns_corpus.parquet", "case_summaries.parquet"]:
    p = f"{EXPORT_DIR}/{f}"
    if os.path.isfile(p):
        print(f"✓ {f}: {os.path.getsize(p)/1e6:.1f} MB (file)")
    elif os.path.isdir(p):
        print(f"✗ {f}: DIRECTORY (wrong — this is the bug)")
    else:
        print(f"✗ {f}: not found")

# COMMAND ----------

import os
EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"

if os.path.exists(EXPORT_DIR):
    for entry in sorted(os.listdir(EXPORT_DIR)):
        full = f"{EXPORT_DIR}/{entry}"
        if os.path.isfile(full):
            print(f"FILE  {entry}  {os.path.getsize(full)/1e6:.1f} MB")
        elif os.path.isdir(full):
            inner = sorted(os.listdir(full))[:5]
            print(f"DIR   {entry}/  contains: {inner}")
else:
    print(f"EXPORT_DIR doesn't exist: {EXPORT_DIR}")

# COMMAND ----------

import shutil, os
EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"

for d in ["ipc_bns_mapping", "ipc_bns_corpus", "indian_criminal_case_summaries_light"]:
    p = f"{EXPORT_DIR}/{d}"
    if os.path.isdir(p):
        shutil.rmtree(p)
        print(f"✓ deleted dir {p}")

print("\nRemaining:")
print(os.listdir(EXPORT_DIR))

# COMMAND ----------

import os
EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"

df1 = spark.table("workspace.default.ipc_bns_mapping").toPandas()
df1.to_parquet(f"{EXPORT_DIR}/ipc_bns_mapping.parquet", index=False)
print(f"✓ mapping: {len(df1)} rows")

df2 = spark.table("workspace.default.ipc_bns_corpus").toPandas()
df2.to_parquet(f"{EXPORT_DIR}/ipc_bns_corpus.parquet", index=False)
print(f"✓ corpus: {len(df2)} rows")

df3 = spark.table("workspace.default.indian_criminal_case_summaries_light").toPandas()
df3.to_parquet(f"{EXPORT_DIR}/case_summaries.parquet", index=False)
print(f"✓ summaries: {len(df3)} rows")

# Verify
print("\nFinal listing:")
for entry in sorted(os.listdir(EXPORT_DIR)):
    full = f"{EXPORT_DIR}/{entry}"
    if os.path.isfile(full):
        print(f"  FILE  {entry}  {os.path.getsize(full)/1e6:.1f} MB")
    else:
        print(f"  DIR   {entry}")

# COMMAND ----------

import os
EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"
for entry in sorted(os.listdir(EXPORT_DIR)):
    full = f"{EXPORT_DIR}/{entry}"
    if os.path.isfile(full):
        print(f"FILE  {entry}  {os.path.getsize(full)/1e6:.2f} MB")
    elif os.path.isdir(full):
        print(f"DIR   {entry}")
    else:
        print(f"???   {entry}")

# COMMAND ----------

import os
EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"

# Run each in its own block, swallow display errors
for table_name, out_name in [
    ("ipc_bns_mapping", "ipc_bns_mapping.parquet"),
    ("ipc_bns_corpus", "ipc_bns_corpus.parquet"),
    ("indian_criminal_case_summaries_light", "case_summaries.parquet"),
]:
    try:
        df = spark.table(f"workspace.default.{table_name}").toPandas()
        out_path = f"{EXPORT_DIR}/{out_name}"
        df.to_parquet(out_path, index=False)
        n = len(df)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"✓ {out_name}: {n} rows, {size_mb:.2f} MB")
        del df  # release memory
    except Exception as e:
        print(f"✗ {out_name}: {type(e).__name__}: {str(e)[:200]}")

# Force the cell to return None so the display layer doesn't try to JSON-encode anything
None

# COMMAND ----------

import os
import pandas as pd
EXPORT_DIR = "/Volumes/workspace/default/legal_bot_data"

for table_name, out_name in [
    ("ipc_bns_mapping", "ipc_bns_mapping.parquet"),
    ("ipc_bns_corpus", "ipc_bns_corpus.parquet"),
    ("indian_criminal_case_summaries_light", "case_summaries.parquet"),
]:
    try:
        spark_df = spark.table(f"workspace.default.{table_name}")
        # Convert to pandas, then make a CLEAN copy with no Spark metadata attached
        raw = spark_df.toPandas()
        # Rebuild the dataframe from scratch — this strips any Spark-attached metadata
        clean = pd.DataFrame({col: raw[col].tolist() for col in raw.columns})
        out_path = f"{EXPORT_DIR}/{out_name}"
        clean.to_parquet(out_path, index=False, engine="pyarrow")
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"✓ {out_name}: {len(clean)} rows, {size_mb:.2f} MB")
        del spark_df, raw, clean
    except Exception as e:
        print(f"✗ {out_name}: {type(e).__name__}: {str(e)[:300]}")

None