"""
ingestion_schemes.py
--------------------
Government Scheme Eligibility ingestion pipeline using Databricks Vector Search.

Source: Unity Catalog table
    catalog  : workspace
    schema   : default
    raw_files: gov_myscheme.csv

Expected CSV columns (gov_myscheme / MyScheme dataset):
    - Scheme Name       : Title of the government scheme
    - Description       : Brief overview of the scheme
    - Eligibility Criteria : Who can apply (primary embed target)
    - Benefits          : Incentives / support provided
    - Application Process : Steps to apply
    - Official Link     : URL to the scheme's official webpage

Workflow:
  1. Read the CSV from Databricks Unity Catalog via Spark → Pandas.
  2. Build a rich combined text per scheme for dense embedding.
  3. Embed every scheme and store in Delta tables with Vector Search indexes.
  4. Create category-level embeddings (mean-pooled) for dual-stage retrieval.

Run once (or re-run to rebuild the indexes from scratch).

Dependencies (install in your Databricks cluster / notebook):
    %pip install sentence-transformers databricks-vectorsearch
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "raw_files"
FILE_NAME = "gov_myscheme.csv"

# Databricks Unity Catalog Volume path format:
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{FILE_NAME}"

# Delta table names for Vector Search
SCHEMES_TABLE = f"{CATALOG}.{SCHEMA}.gov_schemes_embeddings"
CATEGORIES_TABLE = f"{CATALOG}.{SCHEMA}.gov_schemes_categories_embeddings"

# Vector Search endpoint (you'll need to create this once)
VS_ENDPOINT_NAME = "scheme_bot_endpoint"

# Vector Search index names
SCHEMES_INDEX = f"{CATALOG}.{SCHEMA}.gov_schemes_index"
CATEGORIES_INDEX = f"{CATALOG}.{SCHEMA}.gov_schemes_categories_index"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Column names — update if your CSV has slightly different headers
COL_NAME = "Scheme Name"
COL_DESC = "Description"
COL_ELIGIBILITY = "Eligibility Criteria"
COL_BENEFITS = "Benefits"
COL_APP_PROCESS = "Application Process"
COL_LINK = "Official Link"


# ---------------------------------------------------------------------------
# Helper: load CSV from Unity Catalog via Spark
# ---------------------------------------------------------------------------
def load_schemes_from_catalog() -> list[dict]:
    """
    Read gov_myscheme.csv directly from a Databricks Unity Catalog Volume
    and return a list of cleaned scheme dicts.
    """
    print(f"📦 Reading file from Volume path: {VOLUME_PATH}")

    # Spark can read directly from /Volumes/... paths
    spark_df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .option("multiLine", "true")       # handles newlines inside CSV fields
             .option("escape", '"')             # standard CSV quoting
             .csv(VOLUME_PATH)
    )

    pandas_df = spark_df.toPandas()

    print(f"📄 Loaded {len(pandas_df)} schemes.")
    print(f"   Columns detected: {list(pandas_df.columns)}")

    # Normalise column names (strip whitespace)
    pandas_df.columns = [c.strip() for c in pandas_df.columns]

    schemes = []
    for idx, row in pandas_df.iterrows():
        def safe(col):
            val = row.get(col, "")
            return str(val).strip() if val and str(val).lower() != "nan" else ""

        scheme = {
            "scheme_id":           idx,
            "scheme_name":         safe(COL_NAME),
            "description":         safe(COL_DESC),
            "eligibility":         safe(COL_ELIGIBILITY),
            "benefits":            safe(COL_BENEFITS),
            "application_process": safe(COL_APP_PROCESS),
            "official_link":       safe(COL_LINK),
        }

        scheme["category"] = infer_category(scheme["scheme_name"], scheme["description"])
        scheme["embed_text"] = build_embed_text(scheme)
        schemes.append(scheme)

    return schemes


# ---------------------------------------------------------------------------
# Helper: infer a broad category for chapter-style routing
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Agriculture & Farming":    ["agri", "farm", "crop", "kisan", "farmer", "irrigation", "soil", "horticulture"],
    "Education & Scholarships": ["scholar", "education", "school", "student", "college", "study", "fellowship"],
    "Health & Medical":         ["health", "medical", "hospital", "disease", "insurance", "ayushman", "sanitation"],
    "Women & Child Welfare":    ["women", "child", "maternity", "girl", "ladli", "beti", "widow", "mahila"],
    "Housing & Shelter":        ["housing", "house", "awas", "shelter", "pradhan mantri awas"],
    "Employment & Skill":       ["employment", "job", "skill", "mudra", "self-employ", "apprentice", "labour"],
    "Social Welfare & Pension": ["pension", "elderly", "disabled", "welfare", "senior citizen", "handicap"],
    "Financial Assistance":     ["loan", "credit", "subsidy", "grant", "financial", "bank", "interest"],
    "Rural Development":        ["rural", "village", "gram", "panchayat", "mnrega", "mgnrega"],
    "Minority & SC/ST Welfare": ["sc", "st", "obc", "minority", "tribal", "dalit", "schedule"],
}

def infer_category(name: str, description: str) -> str:
    """Assign a broad category based on keyword matching in name + description."""
    combined = (name + " " + description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return category
    return "General / Other"


# ---------------------------------------------------------------------------
# Helper: build a rich combined text for embedding
# ---------------------------------------------------------------------------
def build_embed_text(scheme: dict) -> str:
    """
    Combine fields into a single string optimised for semantic search.
    Eligibility is repeated / placed first because eligibility matching
    is the primary query intent for a rural scheme-checker agent.
    """
    parts = [
        f"Scheme: {scheme['scheme_name']}",
        f"Eligibility: {scheme['eligibility']}",
        f"Description: {scheme['description']}",
        f"Benefits: {scheme['benefits']}",
        f"Application Process: {scheme['application_process']}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[-1])  # skip empty fields


# ---------------------------------------------------------------------------
# STEP 1: Ingest schemes into Delta table and create Vector Search index
# ---------------------------------------------------------------------------
def ingest_schemes(
    schemes: list[dict],
    embedder: SentenceTransformer,
    vsc: VectorSearchClient,
) -> None:
    """
    Embed each scheme's combined text and store in a Delta table.
    Then create/sync a Vector Search index on that table.
    """
    print(f"🚀 Embedding and indexing {len(schemes)} schemes…")

    # Prepare data for Delta table
    rows = []
    for scheme in schemes:
        vector = embedder.encode(scheme["embed_text"]).tolist()
        
        rows.append({
            "id": f"SCHEME_{scheme['scheme_id']}",
            "text": scheme["embed_text"],
            "embedding": vector,
            "scheme_name": scheme["scheme_name"],
            "eligibility": scheme["eligibility"][:500],
            "benefits": scheme["benefits"][:300],
            "application_process": scheme["application_process"][:300],
            "official_link": scheme["official_link"],
            "category": scheme["category"],
        })

    # Create DataFrame with proper schema
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("text", StringType(), True),
        StructField("embedding", ArrayType(FloatType()), True),
        StructField("scheme_name", StringType(), True),
        StructField("eligibility", StringType(), True),
        StructField("benefits", StringType(), True),
        StructField("application_process", StringType(), True),
        StructField("official_link", StringType(), True),
        StructField("category", StringType(), True),
    ])

    df = spark.createDataFrame(rows, schema=schema)

    # Write to Delta table (overwrite mode to rebuild from scratch)
    print(f"💾 Writing to Delta table: {SCHEMES_TABLE}")
    df.write.format("delta").mode("overwrite").saveAsTable(SCHEMES_TABLE)

    print(f"✅ Wrote {len(rows)} schemes to Delta table.")

    # Create or sync Vector Search index
    try:
        print(f"🔍 Creating Vector Search index: {SCHEMES_INDEX}")
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=SCHEMES_INDEX,
            source_table_name=SCHEMES_TABLE,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_dimension=len(rows[0]["embedding"]),
            embedding_vector_column="embedding"
        )
        print("✅ Vector Search index created successfully.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("⚠️  Index already exists, syncing...")
            # Trigger sync for existing index
            index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=SCHEMES_INDEX)
            index.sync()
            print("✅ Index synced successfully.")
        else:
            raise


# ---------------------------------------------------------------------------
# STEP 2: Build category-level embeddings and Vector Search index
# ---------------------------------------------------------------------------
def build_category_collection(
    schemes: list[dict],
    embedder: SentenceTransformer,
    vsc: VectorSearchClient,
) -> None:
    """
    Group scheme texts by inferred category, compute mean embeddings,
    and store in a Delta table with Vector Search index for Stage-1 retrieval.
    """
    print(f"📂 Building category-level embeddings…")

    # Group embed_text per category
    category_groups: dict[str, list[str]] = {}
    for scheme in schemes:
        category_groups.setdefault(scheme["category"], []).append(scheme["embed_text"])

    # Prepare data for Delta table
    rows = []
    for cat_name, texts in category_groups.items():
        embeddings = embedder.encode(texts)  # (n_schemes, dim)
        mean_vector = np.mean(embeddings, axis=0).tolist()

        # Safe ID: replace special chars
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", cat_name)

        rows.append({
            "id": safe_id,
            "embedding": mean_vector,
            "category_name": cat_name,
            "scheme_count": len(texts),
        })
        print(f"  📂 Category: {cat_name} ({len(texts)} schemes)")

    # Create DataFrame with proper schema
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), True),
        StructField("category_name", StringType(), True),
        StructField("scheme_count", IntegerType(), True),
    ])

    df = spark.createDataFrame(rows, schema=schema)

    # Write to Delta table
    print(f"💾 Writing to Delta table: {CATEGORIES_TABLE}")
    df.write.format("delta").mode("overwrite").saveAsTable(CATEGORIES_TABLE)

    print(f"✅ Wrote {len(rows)} categories to Delta table.")

    # Create or sync Vector Search index
    try:
        print(f"🔍 Creating Vector Search index: {CATEGORIES_INDEX}")
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=CATEGORIES_INDEX,
            source_table_name=CATEGORIES_TABLE,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_dimension=len(rows[0]["embedding"]),
            embedding_vector_column="embedding"
        )
        print("✅ Vector Search index created successfully.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("⚠️  Index already exists, syncing...")
            index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=CATEGORIES_INDEX)
            index.sync()
            print("✅ Index synced successfully.")
        else:
            raise


# ---------------------------------------------------------------------------
# Main — run this script in your Databricks notebook
# ---------------------------------------------------------------------------
print("⏳ Initializing Spark session…")
spark = SparkSession.builder.getOrCreate()

print("⏳ Loading embedding model…")
embedder = SentenceTransformer(EMBED_MODEL)

print("⏳ Initializing Vector Search client…")
vsc = VectorSearchClient()

# Check if endpoint exists, create if not
try:
    vsc.get_endpoint(VS_ENDPOINT_NAME)
    print(f"✅ Vector Search endpoint '{VS_ENDPOINT_NAME}' exists.")
except Exception:
    print(f"⚠️  Vector Search endpoint '{VS_ENDPOINT_NAME}' does not exist.")
    print(f"🔨 Creating Vector Search endpoint (this may take a few minutes)...")
    try:
        vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
        print(f"✅ Vector Search endpoint '{VS_ENDPOINT_NAME}' created successfully!")
    except Exception as create_error:
        print(f"❌ Failed to create endpoint: {create_error}")
        print(f"📝 You may need to create it manually via the UI:")
        print(f"   https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint")
        raise

schemes = load_schemes_from_catalog()
ingest_schemes(schemes, embedder, vsc)
build_category_collection(schemes, embedder, vsc)

print("\n🎉 Ingestion complete! Vector Search indexes are ready.")
print(f"   Schemes index: {SCHEMES_INDEX}")
print(f"   Categories index: {CATEGORIES_INDEX}")
print("✅ Data is now persistent in Delta tables and queryable via Vector Search.")
