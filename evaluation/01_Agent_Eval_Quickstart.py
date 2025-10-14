# Databricks notebook source
# MAGIC %md # Evaluate a GenAI App Quickstart
# MAGIC
# MAGIC This quickstart guides you through evaluating a GenAI application using MLflow. It uses a simple example: classifying a financial product into a set of defined classes using a basic prompt. The objective is to clarify the various judges, scorers, guidelines, and other tools to evaluate agents using MLflow. It will scale nicely to more complicated agentic tasks and solutions as long as the model/chain are managed through MLflow.
# MAGIC
# MAGIC
# MAGIC
# MAGIC If you'd prefer a more open-ended chat Q&A prompt, you can use the example here https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/eval which is based on the popular game [Mad Libs](https://en.wikipedia.org/wiki/Mad_Libs).
# MAGIC

# COMMAND ----------

# MAGIC %md ### Prequisites
# MAGIC - This evaluation can leverage Serverless compute or a cluster >= 16.4 DBR

# COMMAND ----------

# MAGIC %md ## Install required packages
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.1.0" openai "databricks-connect>=16.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 1. Create a classification "agent" using a chat completion function

# COMMAND ----------

# DBTITLE 1,Import libraries and set connection/model
import mlflow
from databricks.sdk import WorkspaceClient

# Enable automatic tracing
mlflow.openai.autolog()

# Create an OpenAI client that is connected to Databricks-hosted LLMs
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

# Select an LLM
model_name = "databricks-claude-3-7-sonnet"

# COMMAND ----------

# DBTITLE 1,Define System Prompt and Basic Agent Function
### import json

### # Define JSON schema for guaranteed structure
### response_format = {
###     "type": "json_schema",
###     "json_schema": {
###         "name": "classification_output",
###         "schema": {
###             "type": "object",
###             "properties": {
###                 "CLASSIFICATION": {"type": "string"},
###                 "RATIONALE": {"type": "string"},
###             },
###             "required": ["CLASSIFICATION", "RATIONALE"]
###         }
###     }
### }

# Basic system prompt to classify financial products - if you'd like to try with a defined response format, uncomment the json
SYSTEM_PROMPT = """Classify the following financial product description into one of the categories: Mortgage Loan, Credit Card, Auto Loan, Checking Account, Wealth Management Product, or None. Then provide a 1 sentence justification for why the classification was chosen. Do not include newline characters or breaks."""

@mlflow.trace
def generate_label(template: str):
    """Classify input text using an LLM."""

    response = client.chat.completions.create(
        model=model_name, 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template},
        ],
        ### response_format=response_format
    )
    
    return response.choices[0].message.content

# Test the app
sample_template = "30-year fixed mortgage with 4.2% APR"
result = generate_label(sample_template)
print(f"Input: {sample_template}")
print(f"Output: {result}")

# COMMAND ----------

# DBTITLE 1,Another Example
sample_template = "Marriott Bonvoy vacation club"
result = generate_label(sample_template)
print(f"Input: {sample_template}")
print(f"Output: {result}")

# COMMAND ----------

# DBTITLE 1,Another Example
sample_template = "Chase Sapphire preferred"
result = generate_label(sample_template)
print(f"Input: {sample_template}")
print(f"Output: {result}")

# COMMAND ----------

# MAGIC %md ## Step 2. Create evaluation data

# COMMAND ----------

# MAGIC %md
# MAGIC #### HINT: We need evaluation data to run evaluations!
# MAGIC Depending on a given use case and what information is available, there are a few autonomous and human-driven methodologies to create evaluation data (https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-datasets & https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/build-eval-dataset)
# MAGIC 1. Create evaluation data from existing traces - this can be helpful after initial rollout to a SME group of users or, over time, to monitor performance of an agents
# MAGIC 2. Create evaluation data from existing data (e.g. there may be a list of example customer service questions that can be cleaned and then used as a very good proxy for real questions that will be asked of an agent)
# MAGIC 3. Create synthetic data often using document info (pre-vectorized) as the source
# MAGIC
# MAGIC **If there is access to ground truths (the "right" answer), that is extremely useful and valuable for evaluation**

# COMMAND ----------

# DBTITLE 1,Create basic evaluation data

# This uses the most basic approach of generating a dictionary to use as our evaluation dataset
eval_data = [
    {
        "inputs": {
            "template": "30-year fixed-rate home loan for a primary residence"
        }
    },
    {
        "inputs": {
            "template": "Business travel card with airline miles and lounge access"
        }
    },
    {
        "inputs": {
            "template": "New car financing with a 4.5% APR for 72 months"
        }
    },
    {
        "inputs": {
            "template": "Free student checking account with no monthly fees"
        }
    },
    {
        "inputs": {
            "template": "Managed investment portfolio with a dedicated financial advisor"
        }
    },
    {
        "inputs": {
            "template": "Unsecured personal loan for debt consolidation"
        }
    },
    {
        "inputs": {
            "template": "0% APR balance transfer offer for 18 months"
        }
    },
    {
        "inputs": {
            "template": "Refinance your existing mortgage to a lower interest rate"
        }
    },
    {
        "inputs": {
            "template": "High-yield savings account with a 5.1% APY"
        }
    },
    {
        "inputs": {
            "template": "Used vehicle loan for a 2022 Toyota Camry"
        }
    },
    {
        "inputs": {
            "template": "Business checking account with unlimited transactions"
        }
    },
    {
        "inputs": {
            "template": "Rollover from a 401(k) to a traditional IRA"
        }
    },
    {
        "inputs": {
            "template": "Home equity line of credit with a variable rate"
        }
    },
    {
        "inputs": {
            "template": "Auto loan refinancing to lower your monthly payment"
        }
    },
    {
        "inputs": {
            "template": "12-month Certificate of Deposit (CD)"
        }
    },
    {
        "inputs": {
            "template": "Secured credit card to help build or rebuild credit"
        }
    },
    {
        "inputs": {
            "template": "529 college savings plan for a child's education"
        }
    },
    {
        "inputs": {
            "template": "High-yield interest checking with a $5,000 minimum balance"
        }
    },
    {
        "inputs": {
            "template": "Jumbo loan for a luxury property purchase over $1M"
        }
    },
    {
        "inputs": {
            "template": "Student cash back rewards credit card"
        }
    },
    {
        "inputs": {
            "template": "Term life insurance policy with a $500,000 benefit"
        }
    },
    {
        "inputs": {
            "template": "RV financing for a Class A motorhome"
        }
    },
    {
        "inputs": {
            "template": "Online-only checking account with mobile check deposit"
        }
    },
    {
        "inputs": {
            "template": "Robo-advisor service with diversified ETF portfolios"
        }
    },
    {
        "inputs": {
            "template": "Ice cream cone with sprinkles"
        }
    },
]

# COMMAND ----------

# MAGIC %md ## Step 3. Define evaluation criteria

# COMMAND ----------

# MAGIC %md
# MAGIC #### Judges, Scorers, Ground Truths, Oh My!
# MAGIC - **Judges** are granular evaluation functions, powered by an LLM, to determine if an output meets a specific criteria. They must be wrapped in a scorer to be used by MLflow's evaluation harness, but can used directly via SDK/API.
# MAGIC   - **Predefined Judges** are research-backed, created by the MLflow team for common checks (https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/pre-built-judges-scorers). There is the judge (e.g. `is_safe`) and the predefined scorer (e.g. `Safety`) which uses that judge API all inbuilt.
# MAGIC   - **Custom Judges** come in two flavors and allow you to create custom checks that are best performed by an LLM. Things like semantic correctness, style/tone, safety/compliance, and relative quality. 
# MAGIC     - **Guidelines-Based** are pass/fail criteria where the LLM evaluates output and assigns pass/fail (https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/guidelines). The direct `judges.meets_guidelines` SDK is wrapped in prebuilt `Guidelines()` scorer.
# MAGIC     - **Prompt-Based** are fully customized checks for more complex evaluations where you want an LLM-based assessment that provides a different response than pass/fail (https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/prompt-based-judge)
# MAGIC
# MAGIC - **Scorers** are
# MAGIC
# MAGIC - **Ground Truths** are expected/correct outputs (e.g. labels, fact, gold-standard answers) used as the basis for quality assessment; some scorers/judges require ground truths to function

# COMMAND ----------

# MAGIC %md
# MAGIC 1 - Predefined inbuilt scorers using prebuilt LLM judges

# COMMAND ----------

# DBTITLE 1,First, decide which predefined Scorers to use
import mlflow.genai
from mlflow.genai.scorers import Safety, RelevanceToQuery

# Set predefined evaluation scorers
predefined_scorers = [
    Safety(),  # Built-in safety scorer using `is_safe` judge
    RelevanceToQuery(), # Built-in relevance scorer using `is_context_relevant` judge
    
    ## TO-DO: Would you add any others? Why or why not? 
    ## For reference: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/pre-built-judges-scorers
]

# COMMAND ----------

# MAGIC %md
# MAGIC 2 - Guidelines (pass/fail custom LLM judges)

# COMMAND ----------

# DBTITLE 1,Second, add guidelines for straightforward pass/fail LLM assessments
from mlflow.genai.scorers import Guidelines

# Define guidelines, wrapped as scorers, which are assessed by an LLM and return a pass/fail designation
guideline_scorers = [
    Guidelines(
        guidelines="Response must include label in the predefined set: Mortgage Loan, Credit Card, Auto Loan, Checking Account, Wealth Management Product, or None.",
        name="exact_match",
    ),
    Guidelines(
        guidelines="Response must include a one sentence justification for why the classification was chosen. This should not be too verbose and should provide sensible evidence of the classification decision.",
        name="rationale",
        model="databricks:/databricks-claude-sonnet-4-5",  # Optional custom judge model
    ),

    ## TO-DO: Add a guideline that checks if the response is in English and has no grammatical mistakes.
    
    ## TO-DO: Add a guideline for out-of-scope detection to test if the agent is properly assigning "None" and not hallucinating.
]

# COMMAND ----------

# MAGIC %md
# MAGIC 3 - Custom prompt-based LLM judges providing evaluation results beyond pass/fail

# COMMAND ----------

# DBTITLE 1,Third, add custom prompt-based LLM judges unique to the use case
from mlflow.genai.judges import custom_prompt_judge
import mlflow

# Define prompt-based judge to provide confidence in the classification
confidence_judge = custom_prompt_judge(
  name="classification_confidence",
  prompt_template="""You are an expert financial analyst. Your task is to judge the correctness and the quality of reasoning for the classification below.

**Financial Product Description:**
{{input_description}}

**Proposed Classification & Reasoning:**
{{classification_output}}

Evaluate the proposed classification. Consider two things:
1.  Is the classification label factually correct?
2.  Does the provided reasoning logically support the label?

Select one of the following ratings:

[[correct_and_well_reasoned]]: The classification is correct and the reasoning is sound, relevant, and directly supports the conclusion.
[[correct_but_poorly_reasoned]]: The classification is correct, but the reasoning is weak, incomplete, or contains errors.
[[incorrect]]: The classification is factually wrong, regardless of the reasoning.
[[unclear]]: The classification or reasoning is too ambiguous or vague to judge.""",
# Map confidence levels to numeric values
    numeric_values = {
        "correct_and_well_reasoned": 1.0,
        "correct_but_poorly_reasoned": 0.6,
        "incorrect": 0.3,
        "unclear": 0
    }
)

# Direct usage of SDK with a hard-coded example
feedback = confidence_judge(
  input_description = 'Chase Sapphire preferred',
  classification_output = 'Credit Card. The Chase Sapphire Preferred is a well-known credit card product offered by Chase Bank that provides rewards, travel benefits, and purchase protections to cardholders.'
)

print(feedback.value)
print(feedback.metadata)
print(feedback.rationale)

# COMMAND ----------

# DBTITLE 1,Now, let's wrap in a scorer so we can use with MLflow
from mlflow.genai.scorers import scorer
import mlflow

@scorer
def prompt_confidence_scorer(inputs, outputs, trace):
  """Custom prompt-driven LLM scorer to determine confidence in classifier"""

  return confidence_judge(
    input_description = inputs.get("template"),
    classification_output = outputs
  )

# Simple isolated test to determine if the wrapped scorer matches the direct judge call
eval_data_prompt_scorer = [
    {
        "inputs": {
            "template": "Chase Sapphire preferred"
        },
        "outputs": "Credit Card. The Chase Sapphire Preferred is a well-known credit card product offered by Chase Bank that provides rewards, travel benefits, and purchase protections to cardholders."
    }
]

results_isolated_prompt_scorer = mlflow.genai.evaluate(
    data=eval_data_prompt_scorer,
    scorers=[prompt_confidence_scorer]
)

# COMMAND ----------

# DBTITLE 1,Create your own custom prompt for this use case
#### BONUS QUESTION ####
# Create a custom prompt-based scorer for this use case (which generally is not super applicable for classification agent tasks)
# Some ideas: 
#   synonym normalization (is there consistency across languages/regions/different models of nomenclature, e.g. "home loan" <> "mortgage")
#   identify specific financial nuance (APR vs APY, fixed vs variable) and if any of those are misunderstood leading to inaccuracy - i.e. create a domain-specific custom prompt
#   custom correctness judge

# COMMAND ----------

# MAGIC %md
# MAGIC 4 - Potentially add a custom code-based scorer

# COMMAND ----------

## placeholder for custom python scorer

# COMMAND ----------

# MAGIC %md ## Step 4. Run evaluation

# COMMAND ----------

# DBTITLE 1,Combine scorers
scorers = predefined_scorers + guideline_scorers + [prompt_confidence_scorer]

for scorer in scorers:
    print(f"- {scorer}")

# COMMAND ----------

# DBTITLE 1,Run evaluation
# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=generate_label,
    scorers=scorers
)

# COMMAND ----------

# MAGIC %md ## Step 5. Query Traces to Use & Analyze
# MAGIC Doesn't seem like the sync to Delta captures feedback... so check API.

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

mlflow.set_registry_uri("databricks")

experiment_id_example = MlflowClient().get_experiment_by_name("/Users/ben.dunmire@databricks.com/Demo/classifier-eval-quickstart").experiment_id
print(experiment_id_example)

df = spark.read.format("mlflow-experiment").load(experiment_id_example)
display(df)

# COMMAND ----------

import mlflow
import pandas as pd

experiment_id = experiment_id_example

# Example 1: Get ALL traces (no filter) to see what we have
all_traces_df = mlflow.search_traces(
    experiment_ids=[experiment_id],
    max_results=50,
    return_type="pandas"
)
print(f"All traces DataFrame shape: {all_traces_df.shape}")
print(f"Available columns: {list(all_traces_df.columns)}")
if len(all_traces_df) > 0:
    print(all_traces_df.head())
else:
    print("No traces found in this experiment")


# COMMAND ----------

import mlflow
import pandas as pd
from pyspark.sql import SparkSession
import json

# Get all traces from the experiment
all_traces = mlflow.search_traces(
    experiment_ids=[experiment_id],
    return_type="pandas"
)

# Filter to only traces that have the "classification_sme_assessment" assessment
traces_with_assessment = all_traces[
    all_traces['assessments'].apply(
        lambda assessments: has_assessment(assessments, 'classification_sme_assessment')
    )
]

# Convert complex columns to JSON strings to avoid serialization issues
traces_cleaned = traces_with_assessment.copy()

# Convert complex nested data to JSON strings
complex_columns = ['assessments', 'spans', 'trace_metadata', 'tags', 'request', 'response']
for col in complex_columns:
    if col in traces_cleaned.columns:
        traces_cleaned[col] = traces_cleaned[col].apply(
            lambda x: json.dumps(x) if x is not None else None
        )

# Convert to Spark DataFrame with all columns
spark = SparkSession.builder.appName("TraceAnalysis").getOrCreate()
spark_df = spark.createDataFrame(traces_cleaned)

print(f"Found {len(traces_with_assessment)} traces with classification_sme_assessment")
print(f"Spark DataFrame created with {spark_df.count()} rows")

# Show all columns
spark_df.printSchema()
spark_df.display(5, truncate=False)