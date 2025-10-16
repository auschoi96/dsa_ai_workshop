# Databricks notebook source
# MAGIC %pip install --upgrade dspy mlflow databricks-agents databricks-sdk databricks-mcp databricks-dspy uv unitycatalog-ai[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Making an Agent, or multi-agent? 
# MAGIC
# MAGIC Now that we've seen how to make a tool calling agent in dspy, we will use a combination of tools to create a multi-tool calling agent. 
# MAGIC
# MAGIC For this, we have prepared the following resources for you to use: 
# MAGIC 1. Genie Space called patient lookup with Fake Patient Data 
# MAGIC 2. Vector Search Index with sample insurance documents (genai_in_production_demo_catalog.agents.sbc_details_index). These sample insurance documents generally don't have the big company insurance documents in here so don't expect it to retrieve precise details on this
# MAGIC 3. UC Function that hits you.com 
# MAGIC
# MAGIC Combine these tools to do the following:
# MAGIC
# MAGIC 1. Look up the details of a patient and their associated insurance documents 
# MAGIC 2. Use the UC function you.com to look up more information about the patient 
# MAGIC 3. Compile all the information and write it back to a delta table

# COMMAND ----------

import dspy
import mlflow
import databricks_dspy

mlflow.dspy.autolog()

databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False)
dspy.configure(lm=databricksLM)

predict = dspy.Predict("question->answer")

print(predict(question="why did a chicken cross the kitchen?"))



# COMMAND ----------

# MAGIC %md
# MAGIC #Make a UC Function
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION genai_in_production_demo_catalog.agents.<TODO: update to your own name>(<TODO: update the input to what your function needs> COMMENT <TODO: add a description of what the input should be/look like>)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT <TODO: Add a description of what this function does to help the agent pick the function>
# MAGIC AS $$
# MAGIC <TODO: python code goes here. it's like defining a python function>
# MAGIC $$

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

def web_search(query):
    """Use to research around a specific date. The query should be designed to search for news articles based on key information
    like dates, events, names of people and so forth"""

    client = DatabricksFunctionClient(execution_mode="local")
    result = client.execute_function(
        "genai_in_production_demo_catalog.agents.<TODO: update to your own name>",
        # "genai_in_production_demo_catalog.agents.search_web", #default you.com search function you can use
        parameters={"query": query}
    )
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC #Make a Genie Tool
# MAGIC
# MAGIC The Genie space has already been created for you. Here are the details: 
# MAGIC
# MAGIC <insert genie id once made>

# COMMAND ----------

from databricks_ai_bridge.genie import Genie
from databricks.sdk import WorkspaceClient

patient_lookup = Genie(
    space_id="01f0aa277c491ec9bbed549be09984cd", #change to your Genie Space ID
    client=WorkspaceClient()  
)

response = patient_lookup.ask_question("what symptoms did Shelby Bender experience")
print(response.result)

# COMMAND ----------

#TODO: create a python function that returns the result of the Genie Space

def genie_tool():
    """"""
    return 

# COMMAND ----------

# MAGIC %md
# MAGIC #Make an Agent Bricks Tool

# COMMAND ----------

endpoint_name = "" #TODO: update this to the agent brick endpoint name you set

def sec_search(self, databricks_question):
        """This function needs the User's question. The question is used to pull documentation about Databricks. Use the information to answer the user's question"""
        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(
            endpoint=self.endpoint_name,
            inputs={"dataframe_split": {
                "columns": ["input"],
                "data": [[
                    [{"role": "user", "content": databricks_question}]
                ]]
            }}
        )
        return response['predictions']['output'][0]['content'][0]['text']

# COMMAND ----------

# MAGIC %md
# MAGIC #Put your tools together! 

# COMMAND ----------

class BasicModule(dspy.Module):
    """
    <TODO: Describe what this module does> 
    """
    
    def __init__(self):
        super().__init__()
        <TODO: add your 
        self.predictor = dspy.ChainOfThought(BasicSignature)
        
    def forward(self, input_field: str) -> dspy.Prediction:
        """
        Forward pass of the module.
        
        Args:
            input_field: Input string for processing
            
        Returns:
            dspy.Prediction object with output_field
        """
        prediction = self.predictor(input_field=input_field)
        return prediction