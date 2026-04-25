# Databricks notebook source
# DBTITLE 1,Cell 1
# Use spark.sql() instead of spark.table()
df_mapping = spark.sql("SELECT * FROM workspace.default.ipc_bns_mapping")
df_corpus  = spark.sql("SELECT * FROM workspace.default.ipc_bns_corpus")

# Print counts first
print("Mapping rows:", df_mapping.count())
print("Corpus rows:", df_corpus.count())


print("Done!")

# COMMAND ----------

pip install groq

# COMMAND ----------

import os
from groq import Groq

# Re-set Groq key (env wiped on the kernel restart earlier)
os.environ.setdefault("GROQ_API_KEY", "")  # set GROQ_API_KEY in your environment

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

df_summaries = spark.table("workspace.default.indian_criminal_case_summaries_light").toPandas()
print(f"Loaded {len(df_summaries)} summaries")
print(df_summaries.columns.tolist())
df_summaries.head(2)

# COMMAND ----------

pip install faiss-cpu sentence-transformers

# COMMAND ----------

from case_retriever import CaseRetriever

cr = CaseRetriever(df_summaries=df_summaries, groq_client=_groq_client)

# Test with a lawyer query (rerank=False to save tokens)
results = cr.find_similar_cases(
    "client accused of cheating in online transaction",
    k=5,
    rerank_with_llm=False,
)
for r in results:
    print(f"[{r['score']:.3f}] {r['title'][:70]}")
    print(f"        {r['case_summary'][:150]}...")
    print()

# COMMAND ----------

# Save directly as parquet files (no pandas conversion needed)
df_mapping.write.mode("overwrite").parquet("/dbfs/FileStore/ipc_bns_mapping.parquet")
df_corpus.write.mode("overwrite").parquet("/dbfs/FileStore/ipc_bns_corpus.parquet")

print("Done!")

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/workspace/default/legal_bot_data"))

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/workspace/default/legal_bot_data"))

# COMMAND ----------

# MAGIC %curl -X POST https://your-app-url/api/chat \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{"message": "test", "history": []}' \
# MAGIC   | python -m json.tool