# Databricks notebook source
# MAGIC %md
# MAGIC # Parsing and Chunking Summary of benefits

# COMMAND ----------

# MAGIC %md
# MAGIC ###Chunking Strategy
# MAGIC
# MAGIC In this notebook we will try to ingest the Coverage Summary documents in PDF format. This example makes the assumption that the coverage summary document is in the below format.
# MAGIC
# MAGIC First page is the summary of coverage as shown below
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/CareCost-Compass/refs/heads/main/resources/img_summary.png" alt="drawing" width="700"/>
# MAGIC
# MAGIC Remaining pages has the details of coverage as shown below
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/CareCost-Compass/refs/heads/main/resources/img_details.png" alt="drawing" width="700"/>
# MAGIC
# MAGIC Our aim is to extract this tabular data from PDF and create full text summary of each line item so that it captures the details appropriately. Below is an example
# MAGIC
# MAGIC For the line item
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/CareCost-Compass/refs/heads/main/resources/img_line.png" alt="drawing" width="700"/> we want to generate two paragraphs as below 
# MAGIC
# MAGIC **If you have a test, for Diagnostic test (x-ray, blood work) you will pay $10 copay/test In Network and 40% coinsurance Out of Network.**
# MAGIC
# MAGIC and 
# MAGIC
# MAGIC **If you have a test, for Imaging (CT/PET scans, MRIs) you will pay $50 copay/test In Network and 40% coinsurance Out of Network.**
# MAGIC
# MAGIC We have to create more pipelines and parsing logic for different kind of summary formats
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read PDF documents

# COMMAND ----------

# MAGIC %md
# MAGIC #####Import utility methods

# COMMAND ----------

# MAGIC %pip install docling mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

document_name = ["SBC_client1.pdf", "SBC_client2.pdf", "SBC_client3.pdf", "SBC_client4.pdf"]
x=0
all_chunks = []
while x < len(document_name):
  conv_res = DocumentConverter().convert(f"/Workspace/Users/austin.choi@databricks.com/dsa_ai_workshop/Setup/pdfs/{document_name[x]}")
  doc = conv_res.document
  chunker = HybridChunker()
  chunk_iter = chunker.chunk(dl_doc=doc)

  for i, chunk in enumerate(chunk_iter):
    enriched_text = chunker.serialize(chunk=chunk)

    all_chunks.append({
      "document_name": document_name[x],
      "chunk_id": f"{document_name[x]}_Chunk_{i}",
      "chunk_index": i,
      "content": enriched_text
    })
  x+=1
all_chunks

# COMMAND ----------

chunks_df = spark.createDataFrame(all_chunks)

# COMMAND ----------

display(chunks_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Save the SBC data to a Delta table in Unity Catalog

# COMMAND ----------

catalog = "genai_in_production_demo_catalog"
schema = "agents"
sbc_details_table_name = "sbc_details"
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{sbc_details_table_name}")
chunks_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{sbc_details_table_name}")

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.{sbc_details_table_name}"))

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `genai_in_production_demo_catalog`.`agents`.`sbc_details` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

