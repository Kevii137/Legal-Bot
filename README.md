# ⚖️ Nyaya Sahayak — AI Legal Assistant for India

**An AI-powered multi-agent system that makes Indian criminal law accessible to everyone — by explaining IPC↔BNS section changes, checking government scheme eligibility, drafting FIRs, and responding in any of 12 Indian languages.**

---

## Architecture

<img width="1536" height="1024" alt="WhatsApp Image 2026-04-25 at 4 30 59 PM" src="https://github.com/user-attachments/assets/24a625db-ae1e-4807-b2c2-24fcadff6bb2" />

---

## What It Does

Nyaya Sahayak is a four-agent legal AI system:

**IPC/BNS Comparator** — Three query modes: (A) exact section lookup with LLM explanation, (B) semantic search by crime concept via FAISS, (C) full RAG scenario reasoning that extracts legal concepts from plain-English descriptions and retrieves grounding sections before generating a response.

**Scheme Eligibility Agent** — Three query modes mirroring the comparator: name lookup, category search (dual-stage with category routing), and eligibility check (profile extraction → multi-concept retrieval → LLM verdict). Backed by 3,400+ government schemes from MyScheme.gov.in.

**FIR Drafter** — A conversational bot that collects 11 required FIR fields one at a time, validates each answer with an LLM, and renders a court-ready First Information Report using the official BNSS Section 173 format. Exports to PDF via ReportLab.

**Language Layer** — Sarvam AI wrappers for Hindi/English text translation (Mayura), speech-to-English transcription (Saaras v3), and English-to-Indian-language voice synthesis (Bulbul v3) supporting 12 languages.

---

## Repository Structure

```
nyaya-sahayak/
├── app.py                    # FastAPI backend — all HTTP endpoints
├── main.py                   # Main orchestrator + agent router
├── requirements.txt
├── app.yaml                  # Cloud Run deployment config
├── frontend/
│   └── index.html            # Single-file chat UI (vanilla JS)
├── bns_ipc/
│   ├── ipc_bns_comparator.py # Comparator class — Modes A/B/C
│   ├── case_retriever.py     # HuggingFace ingest + case summarizer
│   └── language.py           # Sarvam AI wrapper (translate/STT/TTS)
├── FIR_drafter/
│   └── fir_drafter.py        # FIRBot class + PDF export
├── scheme_bot/
│   ├── scheme_ingester.py    # CSV → Delta → Databricks VS pipeline
│   ├── scheme_retriever.py   # ChromaDB-based retriever (early version)
│   ├── scheme_sahayak.py     # Databricks notebook entry point
│   ├── scheme_sahayak_v3.py  # SchemeSahayak class (production)
│   └── scheme_sahayak_explore_v3.py  # Exploration notebook
└── bns_bot/
    ├── Ingester.py           # BNS full-text ChromaDB ingestor
    └── Retriever.py          # 3-stage retrieval + chat pipeline
```

---

## How to Run

### Prerequisites

- Python 3.11+
- A Groq API key (free tier works): https://console.groq.com
- (Optional) Sarvam AI key for multilingual: https://sarvam.ai
- (Optional) Databricks workspace for full Delta/VS features

### 1. Clone and install

```bash
git clone https://github.com/kevii137/legal-bot
cd legal-bot
pip install -r requirements.txt
```

`requirements.txt` includes: `groq sentence-transformers faiss-cpu chromadb numpy databricks-vectorsearch`

Additional installs for full features:
```bash
pip install fastapi uvicorn reportlab tiktoken tqdm datasets huggingface_hub
```

### 2. Set environment variables

```bash
export GROQ_API_KEY="gsk_..."
export SARVAM_API_KEY="..."          # optional, for multilingual
export HF_TOKEN="hf_..."            # optional, for case dataset
```

### 3. Run the FastAPI server (local)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### 4. Run on Databricks

Upload the repo to your Databricks workspace, then in a notebook:

```python
%pip install groq nest_asyncio
import nest_asyncio
nest_asyncio.apply()

import uvicorn
from app import app

config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, loop="asyncio")
server = uvicorn.Server(config)

import asyncio
asyncio.get_event_loop().run_until_complete(server.serve())
```

Get your public URL:
```python
host = spark.conf.get("spark.databricks.workspaceUrl")
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(f"https://{host}/driver-proxy/o/0/{cluster_id}/8000/")
```

### 5. Build the IPC/BNS data layer (Databricks, run once)

```python
# In bns_ipc/ipc_bns_comparator_explore (notebook)
# Step 1: Downloads IPC-BNS mapping CSV from HuggingFace
# Step 2: Saves to workspace.default.ipc_bns_mapping
# Step 3: Builds FAISS index in-memory
# Step 4: Saves to workspace.default.ipc_bns_corpus
```

Load the comparator in any notebook:
```python
from bns_ipc.ipc_bns_comparator import Comparator

df_mapping = spark.table("workspace.default.ipc_bns_mapping").toPandas()
df_corpus  = spark.table("workspace.default.ipc_bns_corpus").toPandas()

comp = Comparator(df_mapping=df_mapping, df_corpus=df_corpus)

# Section lookup
result = comp.handle_query("IPC 302")

# Concept search
result = comp.handle_query("murder")

# Scenario reasoning
result = comp.handle_query("someone hacked my account and transferred money")
```

### 6. Build the Scheme data layer (Databricks, run once)

```python
from scheme_bot.scheme_sahayak_v3 import SchemeIngester

ingester = SchemeIngester(spark_session=spark)
ingester.run_pipeline()   # writes gov_schemes + gov_schemes_categories to Delta
```

Load the agent in any session:
```python
from scheme_bot.scheme_sahayak_v3 import SchemeSahayak

bot = SchemeSahayak.from_delta()   # reads from Delta, builds FAISS ~1 min
result = bot.handle_query("I am a poor farmer, what schemes exist for me?")
SchemeSahayak.pretty_print(result)
```

### 7. Run the FIR drafter standalone

```bash
python FIR_drafter/fir_drafter.py
```

---

## Demo Steps (What to Click / What to Type)

| Step | Action | What Happens |
|------|--------|-------------|
| 1 | Open http://localhost:8000 | Welcome screen with suggestion cards |
| 2 | Click **"What is IPC Section 302?"** | IPC § 302 → BNS § 103 card with severity tag |
| 3 | Type `my neighbor threatened me with a knife` | Mode C RAG — surfaces IPC 503/BNS 351 + FIR banner |
| 4 | Click **"Draft FIR"** | FIR drawer opens, progress bar starts |
| 5 | Answer 3–4 fields (police station, name, date) | Progress bar advances per field |
| 6 | (Pre-filled session) scroll to draft preview | Formatted FIR text appears |
| 7 | Click **"Download FIR as PDF"** | PDF downloads with proper letterhead |
| 8 | Click **"New Chat"**, ask about Ayushman Bharat | Scheme agent responds with eligibility + link |
| 9 | Toggle **हिं** button, type `मेरा मोबाइल चोरी हो गया` | Response in Hindi via Sarvam AI |

---

## Data Sources

| Source | License | Usage |
|--------|---------|-------|
| `nandhakumarg/IPC_and_BNS_transformation` (HuggingFace) | Apache 2.0 | IPC → BNS section mapping |
| `opennyaiorg/InJudgements_dataset` (HuggingFace) | CC BY 4.0 | Indian court judgments |
| `MyScheme.gov.in` / `gov_myscheme.csv` | Government Open Data | 3,400+ welfare schemes |
| Bharatiya Nyaya Sanhita 2023 (full text) | Public domain | BNS Bot ingestion |

---

## Key Design Decisions

**Why Delta Lake instead of ChromaDB for IPC/BNS?** The mapping data is tabular and needs SQL queries for exact lookups (Mode A). Delta gives us ACID transactions, time-travel, and direct Spark interoperability — FAISS sits on top for the semantic layer.

**Why FAISS in-memory instead of Databricks Vector Search for IPC/BNS?** The corpus is ~1,100 rows — small enough that FAISS rebuilds from Delta in under 60 seconds. DVS has startup latency and costs; FAISS is instant for this scale.

**Why the three-mode router?** Exact lookups (Mode A) should never go through embeddings — they're faster, cheaper, and deterministic. RAG (Mode C) is expensive; routing to it only for scenario queries keeps latency and cost down.

**Why Groq?** 800+ tokens/second on llama-3.3-70b. The FIR drafter needs low latency per field turn; the comparator needs multi-call pipelines to stay snappy.

---

## Team

Built for the Databricks Hackathon 2026.
