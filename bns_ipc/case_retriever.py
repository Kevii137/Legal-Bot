# Databricks notebook source
import os

# Paste HF token, then clear this cell's output after running
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# Verify
tok = os.environ.get("HF_TOKEN", "")
if tok.startswith("hf_") and len(tok) > 20:
    print(f"✓ Token loaded (prefix: {tok[:8]}..., length: {len(tok)})")
else:
    print("✗ Token doesn't look right — should start with 'hf_'")

# COMMAND ----------

# Install HuggingFace datasets library
%pip install -q datasets
dbutils.library.restartPython()

# COMMAND ----------

import os
from datasets import load_dataset

# Re-load HF token (env vars wiped on kernel restart)
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # ← set HF_TOKEN in your environment

# Load the dataset
print("Loading opennyaiorg/InJudgements_dataset...")
ds = load_dataset("opennyaiorg/InJudgements_dataset", token=os.environ["HF_TOKEN"])

print(f"\nDataset structure:")
print(ds)

# Look at the splits and schema
for split_name, split_data in ds.items():
    print(f"\n--- Split: {split_name} ---")
    print(f"Rows: {len(split_data):,}")
    print(f"Columns: {split_data.column_names}")
    print(f"First row preview:")
    first = split_data[0]
    for key, val in first.items():
        val_str = str(val)
        preview = val_str[:200] + "..." if len(val_str) > 200 else val_str
        print(f"  {key}: {preview}")

# COMMAND ----------

import os

# Re-confirm token (env vars wipe across kernel restarts)
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # ← set HF_TOKEN in your environment

# Approach 1: Explicit login via huggingface_hub (more reliable than env var)
from huggingface_hub import login, whoami
login(token=os.environ["HF_TOKEN"])

# Verify who we are
me = whoami()
print(f"✓ Logged in as: {me['name']} ({me.get('email', 'no email')})")

# Approach 2: Check if we can actually access the dataset card via API
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.dataset_info("opennyaiorg/InJudgements_dataset", token=os.environ["HF_TOKEN"])
    print(f"✓ Have access to dataset: {info.id}")
    print(f"  Private: {info.private}, Gated: {info.gated}")
except Exception as e:
    print(f"✗ Cannot access dataset: {type(e).__name__}: {e}")

# COMMAND ----------

import os
from datasets import load_dataset

# Fix the cache-path warning by pointing at a UC Volume path (persists across restarts)
# Free Edition gives us /Volumes/workspace/default/... — let's use that
# If the path doesn't exist or fails, we fall back to /tmp
try:
    cache_dir = "/Volumes/workspace/default/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    print(f"✓ Cache dir: {cache_dir}")
except Exception as e:
    print(f"⚠ Couldn't use UC Volume cache ({e}), falling back to /tmp")
    os.environ["HF_DATASETS_CACHE"] = "/tmp/.hf.data.cache"

# Load with explicit token (safer than relying on env var pickup)
print(f"\nLoading opennyaiorg/InJudgements_dataset...")
ds = load_dataset(
    "opennyaiorg/InJudgements_dataset",
    token=os.environ["HF_TOKEN"],
)

print(f"\n{'=' * 60}")
print(f"Dataset structure:")
print(f"{'=' * 60}")
print(ds)

# Inspect each split
for split_name, split_data in ds.items():
    print(f"\n--- Split: {split_name} ---")
    print(f"Rows: {len(split_data):,}")
    print(f"Columns: {split_data.column_names}")
    print(f"\nFirst row (all fields, truncated):")
    first = split_data[0]
    for key, val in first.items():
        val_str = str(val)
        preview = val_str[:300] + "..." if len(val_str) > 300 else val_str
        print(f"  {key}: {preview}")

# COMMAND ----------

# Upgrade datasets + huggingface_hub to compatible versions
# The Databricks runtime ships older versions that conflict with each other
%pip install -q --upgrade "datasets>=3.0" "huggingface_hub>=0.24" "fsspec>=2024.9.0"
dbutils.library.restartPython()

# COMMAND ----------

import os

# Re-set token after kernel restart
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # ← set HF_TOKEN in your environment

# Keep cache in /tmp for now — UC Volume needs a different approach we'll address later
os.environ["HF_DATASETS_CACHE"] = "/tmp/.hf.data.cache"

# Explicit HF login (more reliable than env var alone)
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# Load the dataset
from datasets import load_dataset
print("Loading opennyaiorg/InJudgements_dataset...")
ds = load_dataset(
    "opennyaiorg/InJudgements_dataset",
    token=os.environ["HF_TOKEN"],
)

print(f"\n{'=' * 60}")
print(f"Dataset structure:")
print(f"{'=' * 60}")
print(ds)

# Inspect
for split_name, split_data in ds.items():
    print(f"\n--- Split: {split_name} ---")
    print(f"Rows: {len(split_data):,}")
    print(f"Columns: {split_data.column_names}")
    print(f"\nFirst row (all fields, truncated):")
    first = split_data[0]
    for key, val in first.items():
        val_str = str(val)
        preview = val_str[:300] + "..." if len(val_str) > 300 else val_str
        print(f"  {key}: {preview}")

# COMMAND ----------

import pandas as pd

# Convert the HF dataset to pandas
df_cases_raw = ds["train"].to_pandas()
print(f"Total cases: {len(df_cases_raw):,}")

# Quick sanity check
print(f"\nCase types breakdown:")
print(df_cases_raw["Case_Type"].value_counts())

print(f"\nCourt types breakdown:")
print(df_cases_raw["Court_Type"].value_counts())

print(f"\nText size stats (chars):")
print(df_cases_raw["Doc_size"].describe())

print(f"\nCitation count stats:")
print(df_cases_raw["Cited_by"].describe())

# Clean up column names to lowercase + snake_case (Delta-friendly, easier to query in SQL)
df_cases = df_cases_raw.rename(columns={
    "Titles": "title",
    "Court_Name": "court_name_raw",
    "Cites": "cites_count",
    "Cited_by": "cited_by_count",
    "Doc_url": "doc_url",
    "Text": "text",
    "Doc_size": "doc_size",
    "Case_Type": "case_type",
    "Court_Type": "court_type",
    "Court_Name_Normalized": "court",
})

# Add a stable case_id column for primary-key-ish lookups later
df_cases = df_cases.reset_index().rename(columns={"index": "case_id"})
df_cases["case_id"] = df_cases["case_id"].apply(lambda i: f"CASE_{i:05d}")

print(f"\n✓ Renamed {len(df_cases.columns)} columns")
print(f"  Final columns: {list(df_cases.columns)}")

# Save to Delta
CATALOG = "workspace"
SCHEMA = "default"
CASES_TABLE = f"{CATALOG}.{SCHEMA}.indian_court_cases"

print(f"\nSaving to Delta table {CASES_TABLE}...")
spark.createDataFrame(df_cases.astype(str)).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(CASES_TABLE)

print(f"✓ Saved {spark.table(CASES_TABLE).count():,} rows to {CASES_TABLE}")

# COMMAND ----------

# Re-filter to Criminal cases only
df_cases_criminal = df_cases[df_cases["case_type"] == "Criminal"].reset_index(drop=True)

# Also filter out pathologically short or empty judgments (same noise filter we used for IPC-BNS)
df_cases_criminal = df_cases_criminal[
    df_cases_criminal["text"].str.len() >= 500
].reset_index(drop=True)

# Re-assign case_ids cleanly after filtering
df_cases_criminal["case_id"] = df_cases_criminal.index.map(lambda i: f"CRIM_{i:05d}")

print(f"Criminal cases (after noise filter): {len(df_cases_criminal):,}")
print(f"\nCourt breakdown:")
print(df_cases_criminal["court_type"].value_counts())
print(f"\nTop 10 courts by case count:")
print(df_cases_criminal["court"].value_counts().head(10))
print(f"\nText length stats:")
print(df_cases_criminal["text"].str.len().describe().round(0))
print(f"\nTop 5 most-cited criminal cases (Cited_by):")
top_cited = df_cases_criminal.nlargest(5, "cited_by_count")[["title", "court", "cited_by_count"]]
display(top_cited)

# Re-save Delta table (overwrite the previous one)
CATALOG = "workspace"
SCHEMA = "default"
CASES_TABLE = f"{CATALOG}.{SCHEMA}.indian_criminal_cases"

print(f"\nSaving to {CASES_TABLE}...")
spark.createDataFrame(df_cases_criminal.astype(str)).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(CASES_TABLE)

print(f"✓ Saved {spark.table(CASES_TABLE).count():,} criminal cases to {CASES_TABLE}")

# Optionally drop the previous (all-cases) table to keep Unity Catalog clean
# Uncomment if you want to:
# spark.sql(f"DROP TABLE IF EXISTS workspace.default.indian_court_cases")

# COMMAND ----------

import os
from groq import Groq

# Re-set Groq key (env wiped on the kernel restart earlier)
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")  # ← set GROQ_API_KEY in your environment

_groq_client = Groq()

# Smoke-ping to confirm
test = _groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "ok"}],
    max_tokens=5,
    temperature=0,
)
print(f"✓ Groq client ready. Test response: {test.choices[0].message.content!r}")

# COMMAND ----------

import json
import time

# ============================================================
# Case Summary Extractor — schema, prompt, and runner
# ============================================================

CASE_EXTRACTOR_SYSTEM_PROMPT = """You are a legal analyst extracting structured information from Indian court judgments.

Given the full text of a judgment, extract the following fields. If a field cannot be confidently determined from the text, return null or an empty list — DO NOT invent content. Be precise and concise.

Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "case_name": "<plaintiff vs defendant>",
  "citation": "<court name + year if available, else null>",
  "year": <integer or null>,
  "court": "<court name as appears>",
  "court_tier": "Supreme Court | High Court | District Court | Tribunal | Other",

  "facts": "<2-3 sentence factual narrative of what happened>",

  "legal_issues": ["<each issue as one sentence>"],

  "parties": {
    "appellant": "<name + role, e.g. 'X (private individual)' or 'State of Maharashtra'>",
    "respondent": "<name + role>"
  },

  "sections_invoked": [
    {"act": "IPC | CrPC | Constitution | Other", "section": "<number>", "context": "<short why-cited>"}
  ],

  "arguments": {
    "appellant": "<1-2 sentence main argument>",
    "respondent": "<1-2 sentence main argument>"
  },

  "held": "<1-2 sentences: the court's binding holding (ratio decidendi)>",

  "reasoning": "<2-3 sentences on why the court decided this way>",

  "outcome": "appeal_allowed | appeal_dismissed | conviction_upheld | conviction_set_aside | acquittal | remanded | bail_granted | bail_denied | writ_allowed | writ_dismissed | other",

  "sentence_or_relief": "<actual outcome for the parties, e.g. 'life imprisonment confirmed', 'acquittal', '5 years RI reduced to 2 years', 'bail granted on ₹50,000 surety'>",

  "precedents_cited": ["<case name + brief note on what principle it stands for>"],

  "key_principles": ["<1-2 short legal principles this case establishes or applies>"],

  "topical_tags": ["<short keywords, e.g. 'murder', 'common intention', 'circumstantial evidence'>"],

  "lawyer_summary": "<1-2 sentence elevator pitch a lawyer would scan in a search result>"
}

Rules:
- Truncate or summarize where the source text is verbose; do not pad with speculation.
- For sections_invoked: only include sections you can verify appear in the text.
- For precedents_cited: only include cases explicitly mentioned in the judgment.
- If facts are unclear (procedural-only judgments), return facts: "Procedural matter; facts not detailed in this judgment".
"""


def extract_case_summary(case_text: str, client) -> dict:
    """Single-case extractor. Handles long cases by trimming if needed."""
    # Outlier handling: cases > 200K chars get trimmed (first 100K + last 50K)
    # Llama 3.3 70B has 131K token context (~500K chars). 200K is our safety threshold.
    MAX_LEN = 200_000
    text = case_text or ""
    if len(text) > MAX_LEN:
        text = text[:100_000] + "\n\n[...middle truncated for length...]\n\n" + text[-50_000:]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": CASE_EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract structured information from this judgment:\n\n{text}"},
        ],
        temperature=0.1,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ============================================================
# Smoke test on 5 representative cases
# ============================================================

# Pick a diverse sample: 1 Supreme Court, 2 High Court, 1 District, 1 long-text outlier
sample = pd.concat([
    df_cases_criminal[df_cases_criminal["court_type"] == "Supreme_Court"].head(1),
    df_cases_criminal[df_cases_criminal["court_type"] == "High_Court"].head(2),
    df_cases_criminal[df_cases_criminal["court_type"] == "District_And_Tribunals"].head(1),
    df_cases_criminal.nlargest(1, "doc_size"),  # the longest case
]).drop_duplicates(subset=["case_id"]).reset_index(drop=True)

print(f"Smoke test sample: {len(sample)} cases")
print(f"Char lengths: {[len(t) for t in sample['text'].tolist()]}")
print()

results = []
for i, row in sample.iterrows():
    print(f"[{i+1}/{len(sample)}] {row['title'][:80]}")
    print(f"        Court: {row['court']}, text length: {len(row['text']):,} chars")
    t0 = time.time()
    try:
        summary = extract_case_summary(row["text"], _groq_client)
        elapsed = time.time() - t0
        results.append({"case_id": row["case_id"], "summary": summary, "elapsed": elapsed})
        print(f"        ✓ Extracted in {elapsed:.1f}s")
        # Print a key field as a sanity check
        print(f"        Outcome: {summary.get('outcome', '?')}")
        print(f"        Held: {summary.get('held', '?')[:120]}...")
        print(f"        Sections invoked: {[s.get('section') for s in summary.get('sections_invoked', [])][:5]}")
        print(f"        Topical tags: {summary.get('topical_tags', [])[:5]}")
    except Exception as e:
        print(f"        ✗ FAILED: {type(e).__name__}: {e}")
    print()

# Show one full extraction prettily
print("=" * 70)
print("FULL EXTRACTION OF FIRST CASE (for quality review):")
print("=" * 70)
if results:
    print(json.dumps(results[0]["summary"], indent=2, ensure_ascii=False))

# COMMAND ----------

import os
from groq import Groq

# Paste 4 Groq keys
GROQ_KEYS = [
    os.environ.get("GROQ_API_KEY_1", ""),
    os.environ.get("GROQ_API_KEY_2", ""),
    os.environ.get("GROQ_API_KEY_3", ""),
    os.environ.get("GROQ_API_KEY_4", ""),
]

# Verify each key works with a tiny ping
working_clients = []
for i, key in enumerate(GROQ_KEYS):
    try:
        c = Groq(api_key=key)
        r = c.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=5,
            temperature=0,
        )
        working_clients.append(c)
        print(f"✓ Key {i+1}: working ({r.choices[0].message.content!r})")
    except Exception as e:
        print(f"✗ Key {i+1}: FAILED — {type(e).__name__}: {e}")

print(f"\nTotal working clients: {len(working_clients)}/4")

# COMMAND ----------

# Install tiktoken for accurate token counting (it works for any tokenizer family,
# Llama tokenizes similarly to cl100k_base for ratio purposes)
%pip install -q tiktoken tqdm

# COMMAND ----------

# ============================================================================
# CONSOLIDATED: token-aware chunking + atomic per-key counter +
#               metadata-threaded process_case
# Run this once. Then re-run the smoke test cell.
# ============================================================================

import json
import time
import threading
import concurrent.futures
from collections import defaultdict
import tqdm as tqdm

import tiktoken

# ----------------------------------------------------------------------------
# Token counting
# ----------------------------------------------------------------------------
# cl100k_base is OpenAI's tokenizer; ratio is close enough to Llama 3 family
# for budgeting purposes (within ~5%, well inside our safety margin).
_enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


# ----------------------------------------------------------------------------
# Chunking config
# ----------------------------------------------------------------------------
# Groq llama-3.3-70b-versatile free tier: 12,000 TPM per key.
# Budget per request: input + system prompt + output ≤ 12,000.
# Conservative target: 8,000 input tokens per chunk + ~1,500 sys prompt
#                    + 2,500 output tokens = ~12,000 total. Tight but fits.
TARGET_CHUNK_TOKENS = 8000
TARGET_OVERLAP_TOKENS = 600


def chunk_text_by_tokens(
    text: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = TARGET_OVERLAP_TOKENS,
) -> list[str]:
    """Token-aware chunking: encode → slice → decode back to text."""
    tokens = _enc.encode(text, disallowed_special=())
    if len(tokens) <= target_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    return chunks


# ----------------------------------------------------------------------------
# Prompts (lowercase 'json' included so Groq's response_format guard accepts)
# ----------------------------------------------------------------------------
CHUNK_EXTRACTOR_PROMPT = """You are a legal analyst extracting structured information from PART of an Indian court judgment.

This is chunk {chunk_num} of {total_chunks} from the same judgment. The user message will include CASE METADATA at the top — use it for case_name / court / court_tier so you don't have to guess.

Extract whatever fields you can confidently determine from THIS chunk only. Other chunks will be processed separately and merged later. If a field cannot be determined from this chunk, return null or an empty list — DO NOT invent content.

Respond with ONLY valid json in this exact schema (no markdown, no code fences):
{{
  "case_name": "<from metadata header>",
  "citation": "<court name + year if available, else null>",
  "year": <integer or null>,
  "court": "<from metadata header>",
  "court_tier": "Supreme Court | High Court | District Court | Tribunal | Other",
  "facts": "<factual narrative if present in this chunk, else null>",
  "legal_issues": ["<each issue as one sentence>"],
  "parties": {{"appellant": "<...>", "respondent": "<...>"}},
  "sections_invoked": [{{"act": "IPC | CrPC | Constitution | Other", "section": "<number>", "context": "<short why-cited>"}}],
  "arguments": {{"appellant": "<...>", "respondent": "<...>"}},
  "held": "<court's holding if found in this chunk, else null>",
  "reasoning": "<reasoning if found in this chunk, else null>",
  "outcome": "appeal_allowed | appeal_dismissed | conviction_upheld | conviction_set_aside | acquittal | remanded | bail_granted | bail_denied | writ_allowed | writ_dismissed | other | null",
  "sentence_or_relief": "<actual outcome for parties if found, else null>",
  "precedents_cited": ["<case name + brief note>"],
  "key_principles": ["<principles found>"],
  "topical_tags": ["<keywords>"],
  "lawyer_summary": "<short summary of THIS chunk's content>"
}}
"""

MERGE_PROMPT = """You are merging structured json extractions from multiple chunks of the SAME Indian court judgment into one unified summary.

Below are partial json extractions. Produce one coherent merged json summary, preferring non-null values, deduplicating list items, and resolving contradictions in favor of the most specific/complete information.

CRITICAL:
- Do NOT invent content. If no chunk had a clear "held", return whatever the chunks said (even if vague) rather than fabricating.
- Combine lists by union (deduplicate semantically similar entries).
- For prose fields (facts/held/reasoning), prefer the longest/most detailed version, or synthesize if multiple chunks have complementary info.
- For outcome: if chunks disagree, pick the most specific one.
- Preserve case_name and court from any chunk that has them (they came from court records).

Respond with ONLY valid json using this schema:
{
  "case_name": "...", "citation": "...", "year": <int>, "court": "...", "court_tier": "...",
  "facts": "...", "legal_issues": [...], "parties": {"appellant": "...", "respondent": "..."},
  "sections_invoked": [...], "arguments": {"appellant": "...", "respondent": "..."},
  "held": "...", "reasoning": "...", "outcome": "...", "sentence_or_relief": "...",
  "precedents_cited": [...], "key_principles": [...], "topical_tags": [...],
  "lawyer_summary": "..."
}
"""


# ----------------------------------------------------------------------------
# Per-call extraction (single chunk + merge)
# ----------------------------------------------------------------------------

def extract_chunk(client, chunk_text_str: str, chunk_num: int, total_chunks: int) -> dict:
    sys_prompt = CHUNK_EXTRACTOR_PROMPT.format(chunk_num=chunk_num, total_chunks=total_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Chunk {chunk_num} of {total_chunks}:\n\n{chunk_text_str}"},
        ],
        temperature=0.1,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def merge_extractions(client, partials: list[dict]) -> dict:
    user_msg = "Partial json extractions to merge:\n\n" + "\n\n---\n\n".join(
        f"CHUNK {i+1}:\n{json.dumps(p, ensure_ascii=False, indent=2)}"
        for i, p in enumerate(partials)
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": MERGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ----------------------------------------------------------------------------
# Per-key call counter with hard cap
# ----------------------------------------------------------------------------
PER_KEY_CAP = 800
_call_counts = defaultdict(int)
_counter_lock = threading.Lock()


class AllKeysExhausted(Exception):
    """Raised when every key has hit PER_KEY_CAP."""


def pick_least_loaded_key() -> int:
    """Return the index of the least-loaded available key. Atomically increments."""
    with _counter_lock:
        available = [i for i in range(len(working_clients)) if _call_counts[i] < PER_KEY_CAP]
        if not available:
            raise AllKeysExhausted(
                f"All {len(working_clients)} keys at cap of {PER_KEY_CAP}. "
                f"Counts: {dict(_call_counts)}"
            )
        chosen = min(available, key=lambda i: (_call_counts[i], i))
        _call_counts[chosen] += 1
        return chosen


def reset_call_counts():
    with _counter_lock:
        _call_counts.clear()


def get_call_distribution() -> dict:
    with _counter_lock:
        return dict(_call_counts)


# ----------------------------------------------------------------------------
# process_case: single-chunk fast path, multi-chunk parallel + merge
# Metadata is threaded into every chunk so the LLM never has to guess.
# ----------------------------------------------------------------------------

def process_case(
    case_text: str,
    case_id: str,
    title: str = "",
    court: str = "",
    court_type: str = "",
) -> dict:
    chunks = chunk_text_by_tokens(case_text)
    n = len(chunks)

    metadata_header = (
        "=== CASE METADATA (from court records) ===\n"
        f"Title: {title}\n"
        f"Court: {court}\n"
        f"Court tier: {court_type}\n"
        f"Case ID: {case_id}\n"
        "=== END METADATA ===\n\n"
    )

    if n == 1:
        client = working_clients[pick_least_loaded_key()]
        return extract_chunk(client, metadata_header + chunks[0], 1, 1)

    def _run_chunk(i: int, ch: str):
        client = working_clients[pick_least_loaded_key()]
        return i, extract_chunk(client, metadata_header + ch, i + 1, n)

    partials = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(working_clients)) as exe:
        futures = [exe.submit(_run_chunk, i, ch) for i, ch in enumerate(chunks)]
        for fut in concurrent.futures.as_completed(futures):
            i, partial = fut.result()
            partials[i] = partial

    merge_client = working_clients[pick_least_loaded_key()]
    return merge_extractions(merge_client, partials)


print("✓ All extraction functions redefined.")
print(f"  Chunk size: {TARGET_CHUNK_TOKENS} tokens, overlap: {TARGET_OVERLAP_TOKENS} tokens")
print(f"  Per-key cap: {PER_KEY_CAP}")
print(f"  Working clients: {len(working_clients)}")

# COMMAND ----------

reset_call_counts()
print("Reset counters. Re-running smoke test with token-aware chunking + metadata.\n")

# Recompute chunks-per-case using the new tokenizer for accurate planning
print("Smoke test sample with token counts:")
for _, row in sample.iterrows():
    n_tokens = count_tokens(row["text"])
    n_chunks = (n_tokens // TARGET_CHUNK_TOKENS) + (0 if n_tokens % TARGET_CHUNK_TOKENS == 0 else 1)
    print(f"  {row['case_id']}: {len(row['text']):>7,} chars / {n_tokens:>6,} tokens → {n_chunks} chunk(s)")

print()
results = []
errors = []

for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Smoke (token-chunked)"):
    n_tokens = count_tokens(row["text"])
    n_chunks = (n_tokens // TARGET_CHUNK_TOKENS) + (0 if n_tokens % TARGET_CHUNK_TOKENS == 0 else 1)
    expected_calls = n_chunks + (1 if n_chunks > 1 else 0)
    t0 = time.time()
    try:
        summary = process_case(
            row["text"], row["case_id"],
            title=row["title"], court=row["court"], court_type=row["court_type"],
        )
        elapsed = time.time() - t0
        results.append({"case_id": row["case_id"], "n_chunks": n_chunks,
                        "elapsed": elapsed, "summary": summary})
        print(f"  ✓ {row['case_id']}: {n_chunks} chunk(s), {elapsed:.1f}s, "
              f"case_name={summary.get('case_name', '?')[:50]!r}, "
              f"distribution: {get_call_distribution()}")
    except Exception as e:
        errors.append({"case_id": row["case_id"], "n_chunks": n_chunks, "error": str(e)})
        print(f"  ✗ {row['case_id']}: {type(e).__name__}: {e}")

print(f"\n{'=' * 70}")
print(f"Smoke test summary:")
print(f"  Successful: {len(results)}/{len(sample)}")
print(f"  Errors: {len(errors)}")
print(f"  Per-key distribution: {get_call_distribution()}")

if results:
    multi = next((r for r in results if r["n_chunks"] > 1), results[0])
    print(f"\n{'=' * 70}")
    print(f"FULL EXTRACTION OF A MULTI-CHUNK CASE (verify metadata preserved):")
    print(f"{'=' * 70}")
    print(f"Original title: {sample[sample['case_id'] == multi['case_id']].iloc[0]['title']}")
    print(f"Extracted case_name: {multi['summary'].get('case_name', '?')}")
    print(f"Extracted court: {multi['summary'].get('court', '?')}")
    print(json.dumps(multi["summary"], indent=2, ensure_ascii=False))

# COMMAND ----------

# ============================================================
# Lightweight 8B extraction — fast, single-pass, raw-truncated input
# Budget: ~1.3K tokens × 1474 cases = ~1.92M tokens (fits 2M TPD)
# ============================================================

import json
import time
import threading
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm

# Truncate input to first 4000 chars (~1000 tokens)
# Catches case header + parties + sections + opening facts — what matters for retrieval
INPUT_TRUNC_CHARS = 4000

LIGHT_EXTRACTOR_PROMPT = """You are a legal analyst extracting key fields from the start of an Indian court judgment.

The user message contains case metadata and the opening portion of the judgment text. Extract a compact structured summary for use in a search index.

Be brief and factual. Do NOT invent content. If a field cannot be determined, return null or an empty list.

Respond with ONLY valid json in this exact schema (no markdown, no code fences):
{
  "case_summary": "<3-4 sentence narrative: who, what happened, the legal issue, and outcome if visible>",
  "sections_invoked": ["<short form like 'IPC 302' or 'CrPC 482'>"],
  "key_topics": ["<short keywords like 'murder', 'common intention', 'circumstantial evidence'>"],
  "outcome": "appeal_allowed | appeal_dismissed | conviction_upheld | conviction_set_aside | acquittal | remanded | bail_granted | bail_denied | writ_allowed | writ_dismissed | other | null"
}
"""


def extract_light(client, case_text: str, title: str, court: str, court_type: str) -> dict:
    """Single 8B call. Returns compact dict."""
    truncated = (case_text or "")[:INPUT_TRUNC_CHARS]
    metadata_header = (
        f"Title: {title}\n"
        f"Court: {court}\n"
        f"Court tier: {court_type}\n\n"
    )
    user_msg = metadata_header + "Judgment text (truncated):\n\n" + truncated

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": LIGHT_EXTRACTOR_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ============================================================
# Per-key budget tracking (separate from earlier 70B counter)
# Track tokens, not just request count — that's the binding limit
# ============================================================

PER_KEY_TPD_LIMIT_8B = 480_000  # 500K cap with 4% safety margin
_token_counts_8b = defaultdict(int)
_token_lock_8b = threading.Lock()


class AllKeysExhausted8B(Exception):
    pass


def pick_least_loaded_8b_key(estimated_tokens: int = 1500) -> int:
    """Atomically reserve token budget on the least-loaded 8B key."""
    with _token_lock_8b:
        available = [
            i for i in range(len(working_clients))
            if _token_counts_8b[i] + estimated_tokens < PER_KEY_TPD_LIMIT_8B
        ]
        if not available:
            raise AllKeysExhausted8B(
                f"All {len(working_clients)} keys near 8B TPD cap. "
                f"Counts: {dict(_token_counts_8b)}"
            )
        chosen = min(available, key=lambda i: (_token_counts_8b[i], i))
        _token_counts_8b[chosen] += estimated_tokens
        return chosen


def reset_8b_counts():
    with _token_lock_8b:
        _token_counts_8b.clear()


def get_8b_distribution() -> dict:
    with _token_lock_8b:
        return dict(_token_counts_8b)


print("✓ Lightweight extractor + 8B budget tracker ready")
print(f"  Truncation: first {INPUT_TRUNC_CHARS} chars")
print(f"  Per-key 8B TPD cap: {PER_KEY_TPD_LIMIT_8B:,} tokens")
print(f"  Total budget across 4 keys: {PER_KEY_TPD_LIMIT_8B * 4:,} tokens")
print(f"  Estimated per-case cost: ~1500 tokens")
print(f"  Estimated capacity: {(PER_KEY_TPD_LIMIT_8B * 4) // 1500} cases")

# COMMAND ----------

# Tighten truncation for headroom
INPUT_TRUNC_CHARS = 3000
print(f"Truncation tightened to {INPUT_TRUNC_CHARS} chars (~750 tokens input)")
print(f"New estimated per-case cost: ~1100 tokens")
print(f"New estimated capacity: {(PER_KEY_TPD_LIMIT_8B * 4) // 1100} cases (corpus: 1474)\n")

# Smoke test on 5 cases — mix of lengths to confirm truncation behaves
sample_smoke = pd.concat([
    df_cases_criminal[df_cases_criminal["court_type"] == "Supreme_Court"].head(2),
    df_cases_criminal[df_cases_criminal["court_type"] == "High_Court"].head(2),
    df_cases_criminal.nlargest(1, "cited_by_count"),  # most-cited landmark
]).drop_duplicates(subset=["case_id"]).head(5).reset_index(drop=True)

print(f"Smoke test on {len(sample_smoke)} cases:")
reset_8b_counts()

results_smoke = []
for _, row in tqdm(sample_smoke.iterrows(), total=len(sample_smoke), desc="8B smoke"):
    try:
        client_idx = pick_least_loaded_8b_key(estimated_tokens=1100)
        client = working_clients[client_idx]
        t0 = time.time()
        summary = extract_light(client, row["text"], row["title"], row["court"], row["court_type"])
        elapsed = time.time() - t0
        results_smoke.append({"case_id": row["case_id"], "summary": summary, "elapsed": elapsed})
        print(f"  ✓ {row['case_id']} ({elapsed:.1f}s, key {client_idx})")
        print(f"      summary:  {summary.get('case_summary', '?')[:150]}...")
        print(f"      sections: {summary.get('sections_invoked', [])[:5]}")
        print(f"      topics:   {summary.get('key_topics', [])[:5]}")
        print(f"      outcome:  {summary.get('outcome', '?')}")
    except Exception as e:
        print(f"  ✗ {row['case_id']}: {type(e).__name__}: {str(e)[:200]}")

print(f"\nDistribution: {get_8b_distribution()}")
print(f"Total tokens estimated: {sum(get_8b_distribution().values()):,}")

# COMMAND ----------

import os
import json
import time
import concurrent.futures

# ============================================================
# Full extraction run — 1474 criminal cases
# ============================================================

SUMMARIES_TABLE = f"{CATALOG}.{SCHEMA}.indian_criminal_case_summaries_light"

# Resume: skip cases already extracted
try:
    existing = spark.table(SUMMARIES_TABLE).select("case_id").toPandas()
    already_done = set(existing["case_id"].tolist())
    print(f"✓ Resuming: {len(already_done)} cases already in {SUMMARIES_TABLE}")
except Exception:
    already_done = set()
    print(f"✓ Fresh run: no existing summaries table")

# Reset budget tracker (in case smoke test left it at 4400)
reset_8b_counts()
# Pre-charge the budget tracker for cases already done so distribution stays roughly even on resume
# (skip — fresh accounting is fine since we only count what we send)

remaining = df_cases_criminal[~df_cases_criminal["case_id"].isin(already_done)].reset_index(drop=True)
print(f"✓ Cases to process this run: {len(remaining)}")
print(f"  Estimated tokens: {len(remaining) * 1100:,}")
print(f"  Total budget: {PER_KEY_TPD_LIMIT_8B * 4:,} tokens\n")

# ---------- The actual run ----------
CHECKPOINT_EVERY = 100
buffer = []
errors = []
MAX_WORKERS = 2  # light parallelism — budget tracker is the bottleneck, not LLM latency

def process_one(row_dict: dict) -> dict | None:
    """Process a single case. Returns the row to insert, or None on failure."""
    try:
        client_idx = pick_least_loaded_8b_key(estimated_tokens=1100)
        client = working_clients[client_idx]
        summary = extract_light(
            client,
            row_dict["text"],
            row_dict["title"],
            row_dict["court"],
            row_dict["court_type"],
        )
        return {
            "case_id": row_dict["case_id"],
            "title": row_dict["title"],
            "court": row_dict["court"],
            "court_type": row_dict["court_type"],
            "doc_url": row_dict["doc_url"],
            "cites_count": int(row_dict["cites_count"]),
            "cited_by_count": int(row_dict["cited_by_count"]),
            "summary_json": json.dumps(summary, ensure_ascii=False),
            "case_summary": summary.get("case_summary", ""),
            "sections_invoked_json": json.dumps(summary.get("sections_invoked", [])),
            "key_topics_json": json.dumps(summary.get("key_topics", [])),
            "outcome": str(summary.get("outcome", "") or ""),
            "extracted_via_key": client_idx,
        }
    except Exception as e:
        errors.append({"case_id": row_dict["case_id"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
        return None


def flush_to_delta(rows: list[dict], is_first_write: bool):
    """Write the buffer to Delta. Append unless it's the very first write of a fresh table."""
    if not rows:
        return
    df_chunk = pd.DataFrame(rows)
    mode = "overwrite" if is_first_write else "append"
    spark.createDataFrame(df_chunk.astype(str)).write \
        .format("delta") \
        .mode(mode) \
        .option("mergeSchema", "true") \
        .saveAsTable(SUMMARIES_TABLE)


# Drive the loop with tqdm so you see ETA
t_start = time.time()
processed = 0
is_first_write = (len(already_done) == 0)

rows_list = remaining.to_dict(orient="records")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    futures = {exe.submit(process_one, row): i for i, row in enumerate(rows_list)}
    
    pbar = tqdm(total=len(rows_list), desc="Extracting")
    for fut in concurrent.futures.as_completed(futures):
        result = fut.result()
        if result is not None:
            buffer.append(result)
        processed += 1
        pbar.update(1)
        
        # Periodic status
        if processed % 50 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            dist = get_8b_distribution()
            pbar.set_postfix({
                "rate": f"{rate:.1f}/s",
                "errors": len(errors),
                "key0": dist.get(0, 0),
                "key1": dist.get(1, 0),
                "key2": dist.get(2, 0),
                "key3": dist.get(3, 0),
            })
        
        # Checkpoint
        if len(buffer) >= CHECKPOINT_EVERY:
            flush_to_delta(buffer, is_first_write)
            is_first_write = False
            buffer = []
    
    pbar.close()

# Final flush
if buffer:
    flush_to_delta(buffer, is_first_write)

elapsed_total = time.time() - t_start
final_count = spark.table(SUMMARIES_TABLE).count()

print(f"\n{'=' * 60}")
print(f"DONE in {elapsed_total/60:.1f} minutes")
print(f"  Total cases in {SUMMARIES_TABLE}: {final_count:,}")
print(f"  Errors this run: {len(errors)}")
print(f"  Final per-key token distribution: {get_8b_distribution()}")
if errors:
    print(f"\n  First 5 errors:")
    for e in errors[:5]:
        print(f"    {e['case_id']}: {e['error']}")