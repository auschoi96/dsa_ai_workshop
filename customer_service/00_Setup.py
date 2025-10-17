# Databricks notebook source
# MAGIC %md
# MAGIC # Data Setup

# COMMAND ----------

# DBTITLE 1,Create volume
# Change these to your catalog and schema names before running 00_Setup
catalog_name = "genai_in_production_demo_catalog"
schema_name = "customer_service"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.tech_support")

# COMMAND ----------

# DBTITLE 1,Copy files - runs on serverless
# MAGIC %sh cp -r /Workspace/Users/ben.dunmire@databricks.com/dsa_ai_workshop/customer_service/tech_support/knowledge_base /Volumes/genai_in_production_demo_catalog/customer_service/tech_support

# COMMAND ----------

# MAGIC %sh cp -r /Workspace/Users/ben.dunmire@databricks.com/dsa_ai_workshop/customer_service/tech_support/support_tickets /Volumes/genai_in_production_demo_catalog/customer_service/tech_support

# COMMAND ----------

# DBTITLE 1,Copy files - needs classic compute
import os

current_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
data_path = f"file:/Workspace{current_dir}/tech_support"

dbutils.fs.cp(f"{data_path}/", f"/Volumes/{catalog_name}/{schema_name}/tech_support/")
