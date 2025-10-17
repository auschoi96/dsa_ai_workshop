# Databricks notebook source
# MAGIC %pip install --upgrade dspy mlflow databricks-agents databricks-sdk databricks-mcp databricks-dspy uv unitycatalog-ai[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Using MLflow make_judge to make a custom judge
# MAGIC
# MAGIC Mlflow make judge capability gives us flexibility in creating a judge that fits our use case 
# MAGIC
# MAGIC Let's take the agent we built in notebook 5 of the hackathon and evaluate it with our own judges
# MAGIC
# MAGIC Documentation: https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/make-judge/

# COMMAND ----------

import dspy
import mlflow
import databricks_dspy

mlflow.dspy.autolog()
# databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-claude-sonnet-4-5', cache=False)
databricksLM = databricks_dspy.DatabricksLM('databricks/databricks-gpt-oss-120b', cache=False)
dspy.configure(lm=databricksLM)

# COMMAND ----------

class patient_lookup_websearch(dspy.Signature):
  """This agent looks up some patients and then looks up each patient insurance details with health_insurance_look_up"""
  query: str = dspy.InputField()
  patient_report: str = dspy.OutputField()

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_ai_bridge.genie import Genie
from databricks.sdk import WorkspaceClient
from databricks_dspy import DatabricksRM

def health_insurance_look_up(question):
  """Used to query a vector search endpoint"""
  rm = DatabricksRM(
      databricks_index_name="genai_in_production_demo_catalog.agents.sbc_details_index", #change this to your index
      databricks_token="your_databricks_token", #change this to your token, Optional if working in a databricks notebook
      columns=["chunk_id", "content"], #change these to columns you would like to query and retrieve 
      text_column_name="content", #change this to the text column being retrieved 
      docs_id_column_name="chunk_id", #change to the ID 
      k=3 #change this to how many rows you would like to retrieve
  )

  result = rm(query=question) #adjust result to return what you're looking for 

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

# MAGIC %md
# MAGIC #Double check that it works

# COMMAND ----------

question = "Identify patients who have the most severe symptoms based on doctor notes"
patient_report = dspy.ReAct(patient_lookup_websearch, tools=[health_insurance_look_up, patient_lookup])
agent_output = patient_report(query=question)
print(agent_output)

# COMMAND ----------

from IPython.display import Markdown
Markdown(agent_output.patient_report)

# COMMAND ----------

context = json.dumps(agent_output.trajectory.as_dict() if hasattr(agent_output.trajectory, "as_dict") else str(agent_output.trajectory), indent=2)

# COMMAND ----------

# MAGIC %md
# MAGIC #Let's it out ourselves
# MAGIC
# MAGIC We can use these judges to power prompt optimizers like GEPA or set them in our Experiment to use for post-production evaluation and monitoring

# COMMAND ----------

from mlflow.genai.judges import is_context_relevant

feedback = is_context_relevant(
    request=question,
    context=agent_output.patient_report
)
print(feedback.value)  # "yes"
print(feedback.rationale)  # Explanation of groundedness

# COMMAND ----------

from mlflow.genai.judges import make_judge

# Create a scorer for customer support quality
tool_calling_eval = make_judge(
    name="tool_call_quality",
    instructions=(
        "Evaluate if the trajectory in {{ outputs }} shows accurately utilizes the provided tools to answer the question"
        "in {{ inputs }}. The web_search tool is a mandatory step to find information about a patient\n\n"
        "Check if the LLM sent the correct parameter in a good format and assess if a better parameter can be provided"
        "responds with understanding and care.\n"
        "Rate as: 'High', 'Medium' or 'Low'"
    ),
    model="databricks:/databricks-claude-sonnet-4-5"
)

# Test the scorer on a support interaction
result = tool_calling_eval(
    inputs={"question": question},
    outputs={"trajectory": context}
)

print(f"Rating: {result.value}")
print(f"Reasoning: {result.rationale}")

# COMMAND ----------

formatting_eval = make_judge(
name="formatting_eval",
instructions=(
    "Evaluate if the output in {{ outputs }} is structured with proper markdown and is human legible. "
    "based on question and context in {{ inputs }} . It should be a professional and detailed report yet concise \n\n"
    "Rate as: 'High', 'Medium' or 'Low'"
),
model="databricks:/databricks-claude-sonnet-4-5"
)

# Test the scorer on a support interaction
result = formatting_eval(
    inputs={"question": question, "context": context},
    outputs={"output": agent_output.patient_report}
)

print(f"Rating: {result.value}")
print(f"Reasoning: {result.rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Try it yourself! 
# MAGIC
# MAGIC Make your own judge for the Agent

# COMMAND ----------

TODO = make_judge(
name="TODO",
instructions=(
    TODO
),
model="databricks:/databricks-claude-sonnet-4-5"
)

# Test the scorer on a support interaction
result = formatting_eval(
    inputs=TODO,
    outputs=TODO
)

print(f"Rating: {result.value}")
print(f"Reasoning: {result.rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC #That's the end of the workshop!

# COMMAND ----------

# MAGIC %md
# MAGIC #Appendix: GEPA (it doesn't work rn)
# MAGIC
# MAGIC We won't cover how to run GEPA for this agent due to Rate Limits and other resource limits, especially from a bunch of people hitting the APIs at the same time. 
# MAGIC
# MAGIC Fundamentally, you use a combination of judges and datasets to ground and optimize your agents with your data. Try it out below!
# MAGIC

# COMMAND ----------

import time
import numpy as np
import json
from databricks.agents.evals import judges
from mlflow.genai.judges import is_context_relevant
from mlflow.genai.judges import make_judge

def tool_calling_eval_with_feedback(question, prediction, trace=None, pred_name=None, pred_trace=None) -> bool:
    """
    Uses Databricks AI judges to validate the prediction and return score and feedback
    """
    context = json.dumps(prediction.trajectory.as_dict() if hasattr(prediction.trajectory, "as_dict") else str(prediction.trajectory), indent=2)



    tool_calling_eval = make_judge(
    name="tool_call_quality",
    instructions=(
        "Evaluate if the trajectory in {{ outputs }} shows accurately utilizes the provided tools to answer the question"
        "in {{ inputs }}.\n\n"
        "Check if the LLM sent the correct parameter in a good format and assess if a better parameter can be provided"
        "responds with understanding and care.\n"
        "Rate as: 'High', 'Medium' or 'Low'"
    ),
    model="databricks:/databricks-claude-sonnet-4-5"
    )
    
    tool_calling_eval_result = tool_calling_eval(
        inputs={"question": question},
        outputs={"trajectory": context}
    )

    formatting_eval = make_judge(
    name="formatting_eval",
    instructions=(
        "Evaluate if the output in {{ outputs }} is structured with proper markdown and is human legible. "
        "based on question and context in {{ inputs }} . It should be a professional and detailed report yet concise \n\n"
        "Rate as: 'High', 'Medium' or 'Low'"
    ),
    model="databricks:/databricks-claude-sonnet-4-5"
    )

    # Test the scorer on a support interaction
    formatting_eval_result = formatting_eval(
        inputs={"question": question, "context": context},
        outputs={"output": prediction.patient_report}
    )
    

    # context_relevant_eval_result = is_context_relevant(
    #     request=question,
    #     context=context
    # )

    # print(f"Rating: {result.value}")
    # print(f"Reasoning: {result.rationale}")

    if tool_calling_eval_result.value == "High":
        tool_calling_metric = (100 * 0.65)
    if tool_calling_eval_result.value == "Medium":
        tool_calling_metric = (50 * 0.65)
    if tool_calling_eval_result.value == "Low":
        tool_calling_metric = (0 * 0.65)

    if formatting_eval_result.value == "High":
        formatting_metric = (100 * 0.35)
    if formatting_eval_result.value == "Medium":
        formatting_metric = (50 * 0.35)
    if formatting_eval_result.value == "Low":
        formatting_metric = (0 * 0.35)

    # if context_relevant_eval_result.value == "yes":
    #     context_relevancy_metric = (100 * 0.1)
    # if context_relevant_eval_result.value == "no":
    #     context_relevancy_metric = (0 * 0.1)

    # final_metric = tool_calling_metric + formatting_metric + context_relevancy_metric
    final_metric = (tool_calling_metric + formatting_metric)/100

    full_feedback = f"""Tool Calling feedback: {tool_calling_eval_result.rationale}
                    Formatting feedback:  {formatting_eval_result.rationale}
                    """ 

    return dspy.Prediction(score=final_metric, feedback=full_feedback)

def check_accuracy(dspy_program, test_data: list = test_dataset) -> float:
    """
    Checks the accuracy of the classifier on the test data.
    """
    scores = []
    for example in test_dataset:
        print(example)
        prediction = dspy_program(query=example)
        print(prediction)
        score = tool_calling_eval_with_feedback(question=example, prediction=prediction).score
        scores.append(score)
        
    return np.mean(scores)

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_ai_bridge.genie import Genie
from databricks.sdk import WorkspaceClient

def patient_web_search(people: str):
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

small_lm_name = "databricks-gpt-oss-20b"
databricksLM = databricks_dspy.DatabricksLM(f"databricks/{small_lm_name}", cache=False)
dspy.configure(lm=databricksLM)

# test_dataset = [
#   "Identify patients who have the most severe symptoms based on doctor notes",
#   "What patients are in los angeles?",
#   "which patients have united healthcare",
#   "who has visited the most doctors recently",
#   "Show me patients with diabetes who haven't had a checkup in 6 months",
#   "Which patients are overdue for their annual physical exam",
#   "Find all patients currently taking blood pressure medication",
#   "What patients have Medicare coverage and are over 65",
#   "List patients who visited the emergency room in the last 30 days",
#   "Which patients are scheduled for surgery next week",
#   "Show patients with chronic pain conditions in New York",
#   "Find patients who have missed more than 2 appointments this year",
#   "What patients need prescription refills within the next 7 days",
#   "List all pediatric patients under age 5",
#   "Which patients have had lab work done in the past month"
# ]

test_dataset = ["Identify patients who have the most severe symptoms based on doctor notes"]
dspy_patient_agent = dspy.ReAct(patient_lookup_websearch, tools=[health_insurance_look_up, patient_lookup], max_iters=3)

uncompiled_small_lm_accuracy = check_accuracy(dspy_program=dspy_patient_agent)

displayHTML(f"<h1>Uncompiled {small_lm_name} accuracy: {uncompiled_small_lm_accuracy}</h1>")

# COMMAND ----------

import uuid

# defining an UUID to identify the optimized module
id = str(uuid.uuid4())
print(f"id: {id}")

# COMMAND ----------

train_dataset = [
    "How many visits has Jessica Jones made, and what were her reasons for visit?",
    "Which insurance provider does Kyle Leonard use for his visits?",
    "What is the most recent doctor note for Matthew Walsh?",
    "Which city does Joseph Hernandez most frequently visit for medical care?",
    "What insurance type is associated with April Ramirez’s visits?",
    "Identify patients who have the most severe symptoms based on doctor notes",
    "What patients are in los angeles?",
    "which patients have united healthcare",
    "who has visited the most doctors recently",
    "Show me patients with diabetes who haven't had a checkup in 6 months",
    "Which patients are overdue for their annual physical exam",
    "Find all patients currently taking blood pressure medication",
    "What patients have Medicare coverage and are over 65",
    "List patients who visited the emergency room in the last 30 days",
    "Which patients are scheduled for surgery next week",
    "Show patients with chronic pain conditions in New York",
    "Find patients who have missed more than 2 appointments this year",
    "What patients need prescription refills within the next 7 days",
    "List all pediatric patients under age 5",
    "Which patients have had lab work done in the past month"
]
trainset = [
    dspy.Example(query=q).with_inputs("query")
    for q in train_dataset
]

# COMMAND ----------

# small_lm_name = "databricks-gpt-oss-20b"
small_lm_name = "k"
reflection_lm_name = "databricks-claude-sonnet-4-5"

databricksLM = databricks_dspy.DatabricksLM(f"databricks/{small_lm_name}", cache=False)
dspy.configure(lm=databricksLM)

gepa = dspy.GEPA(
    metric=tool_calling_eval_with_feedback,
    auto="light",
    reflection_minibatch_size=15,
    reflection_lm=dspy.LM(f"databricks/{reflection_lm_name}", max_tokens=25000),
    num_threads=16,
    seed=1
)

with mlflow.start_run(run_name=f"gepa_{id}"):
    compiled_gepa = gepa.compile(
        dspy_patient_agent,
        trainset=trainset, #reminder: Only passing 15 training sets!
    )

compiled_gepa.save(f"compiled_gepa_{id}.json")

# COMMAND ----------

small_lm_name = "gpt-oss-20b"
databricksLM = databricks_dspy.DatabricksLM(f"databricks/{small_lm_name}", cache=False)
dspy.configure(lm=databricksLM)

dspy_patient_agent_gepa = patient_lookup
dspy_patient_agent_gepa.load(f"compiled_gepa_{id}.json")

compiled_small_lm_accuracy = check_accuracy(dspy_program=dspy_patient_agent_gepa)
displayHTML(f"<h1>Compiled {small_lm_name} accuracy: {compiled_small_lm_accuracy}</h1>")
