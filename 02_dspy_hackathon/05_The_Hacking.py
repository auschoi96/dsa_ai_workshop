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
# MAGIC 3. Compile all the information and write it back to a delta table using DBSQL MCP server

# COMMAND ----------

import dspy
import mlflow
import databricks_dspy

mlflow.dspy.autolog()

# databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False)
databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-claude-sonnet-4-5', cache=False)
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
# MAGIC #MCP Interaction
# MAGIC
# MAGIC The DBSQL MCP just came out! Here's how to use it

# COMMAND ----------

from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from databricks.sdk import WorkspaceClient
import dspy

# Initialize the Databricks workspace client
workspace_client = WorkspaceClient()

host = workspace_client.config.host
MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/functions/system/ai",
    f"{host}/api/2.0/mcp/sql" #this is the new MCP server 
]

databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-claude-sonnet-4-5', cache=False)
dspy.configure(lm=databricksLM)

async with streamablehttp_client(
    url=f"{host}/api/2.0/mcp/sql", #this is the new MCP server
    auth=DatabricksOAuthClientProvider(workspace_client),
) as (read_stream, write_stream, _):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = await session.list_tools()
        print(tools)
        # Convert MCP tools to DSPy tools
        dspy_tools = []
        for tool in tools.tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

        # Create the agent
        react = dspy.ReAct("question -> answer", tools=dspy_tools)

        result = await react.acall(question="find information of patients in genai_in_production_demo_catalog.agents.patient_vists")
        print(result)


# COMMAND ----------

import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    # Connect to HTTP MCP server
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List and convert tools
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Create and use ReAct agent
            class TaskSignature(dspy.Signature):
                task: str = dspy.InputField()
                result: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=TaskSignature,
                tools=dspy_tools,
                max_iters=5
            )

            result = await react_agent.acall(task="Check the weather in Tokyo")
            print(result.result)

asyncio.run(main())

# COMMAND ----------

# MAGIC %md
# MAGIC #Put your tools together! 
# MAGIC Combine these tools to do the following:
# MAGIC
# MAGIC 1. Look up the details of a patient and their associated insurance documents
# MAGIC 2. Use the UC function you.com to look up more information about the patient
# MAGIC 3. Compile all the information and write it back to a delta table using DBSQL MCP server

# COMMAND ----------

# MAGIC %md
# MAGIC #Easy way

# COMMAND ----------

# MAGIC %md
# MAGIC #Harder but more flexible way

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

# COMMAND ----------

# MAGIC %md
# MAGIC #testing

# COMMAND ----------

class patient_lookup_websearch(dspy.Signature):
  """This agent looks up some patients and then looks up each patient on the web using web_search"""
  query: str = dspy.InputField()
  patient_report: str = dspy.OutputField()

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_ai_bridge.genie import Genie
from databricks.sdk import WorkspaceClient

def web_search(people: str):
    """Use to research more information by doing a websearch"""

    client = DatabricksFunctionClient(execution_mode="local")
    result = client.execute_function(
        # "genai_in_production_demo_catalog.agents.<TODO: update to your own name>",
        "genai_in_production_demo_catalog.agents.search_web", #default you.com search function you can use
        parameters={"query": f"{people}"}
    )
    return result

def patient_lookup(query):
    """use natural language to lookup patient information"""
    patient_lookup = Genie(
        space_id="01f0aa277c491ec9bbed549be09984cd", #change to your Genie Space ID
        client=WorkspaceClient()  
    )

    response = patient_lookup.ask_question(f"{query}, limit to two results")
    return response.result

# COMMAND ----------

patient_report = dspy.ReAct(patient_lookup_websearch, tools=[web_search, patient_lookup])
result = patient_report(query="find information about patients with flu symptoms")
print(result)

# COMMAND ----------

from IPython.display import Markdown
Markdown(result.patient_report)

# COMMAND ----------

databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-20b', cache=False)
dspy.configure(lm=databricksLM)

# COMMAND ----------

patient_report = dspy.ReAct(patient_lookup_websearch, tools=[web_search, patient_lookup])
result = patient_report(query="find information about patients with flu symptoms")
print(result)

# COMMAND ----------

from IPython.display import Markdown
Markdown(result.patient_report)

# COMMAND ----------


