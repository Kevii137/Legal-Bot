# Databricks notebook source
# MAGIC %pip install groq

# COMMAND ----------

# DBTITLE 1,Start FastAPI Server
import uvicorn
import nest_asyncio
import asyncio

# 1. Allow nested event loops
nest_asyncio.apply()

# 2. Define the configuration
config = uvicorn.Config(
    app=app, 
    host="0.0.0.0", 
    port=8000, 
    loop="asyncio" # Explicitly tell it to use asyncio
)

# 3. Create the server
server = uvicorn.Server(config)

# 4. Run it using the existing loop
print("Starting FastAPI server on http://0.0.0.0:8000")
print("API docs available at http://0.0.0.0:8000/docs")

# Instead of uvicorn.run(), we use the current loop to run the server's serve() method
loop = asyncio.get_event_loop()
loop.run_until_complete(server.serve())

# COMMAND ----------

host = spark.conf.get("spark.databricks.workspaceUrl")
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(f"https://{host}/driver-proxy/o/0/{cluster_id}/8000/")

# COMMAND ----------

pip install pyngrok

# COMMAND ----------

from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(public_url)

# COMMAND ----------

