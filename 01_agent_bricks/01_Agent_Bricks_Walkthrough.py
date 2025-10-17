# Databricks notebook source
# MAGIC %md
# MAGIC #DEMO: Create a Knowledge Assistant (KA)
# MAGIC
# MAGIC We'll show a Knowledge Assistant that will help provide technical support and customer service for a customer service company using 1) knowledge base articles and 2) example support tickets.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuring Your Tech Support Knowledge Assistant
# MAGIC In this step, you'll configure your agent by providing its name and description, and you'll add two key knowledge sources.
# MAGIC
# MAGIC 1.1. Name and Describe Your Agent
# MAGIC - Agent Name: **tech_support_knowledge_assistant+something_unique**
# MAGIC - Agent Description: Tech support agent to handle technical support queries. It can answer questions about device problems, technical issues, and company policies.
# MAGIC
# MAGIC <img src=https://raw.githubusercontent.com/chen-data-ai/Agent-Bricks-Workshop/1b6163bb59305a54fbaf945abb64131deee5af19/resources/screenshots/screenshot_knowledge_assistant_config_1.png width="60%">
# MAGIC
# MAGIC 1.2. Add Knowledge Sources
# MAGIC We will add two main knowledge sources. When adding each knowledge source, you must specify the path to the underlying files stored in your Unity Catalog volume.
# MAGIC
# MAGIC - **Knowledge Base:** This contains support articles from our knowledge base. It has FAQs, debugging tips, and more. It also contains company policies around things like data overages and early termination fees.
# MAGIC   - /Volumes/genai_in_production_demo_catalog/customer_service/tech_support/knowledge_base
# MAGIC - **Support Tickets:** This contains support tickets from 2024 and 2025 and the resolution to issues. Use this to find error codes, how to fix technical issues.
# MAGIC   - /Volumes/genai_in_production_demo_catalog/customer_service/tech_support/support_tickets
# MAGIC
# MAGIC <img src=https://raw.githubusercontent.com/chen-data-ai/Agent-Bricks-Workshop/refs/heads/main/resources/screenshots/screenshot_knowledge_assistant_config_2.png width="60%">

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create and Validate
# MAGIC Once you’ve finished configuring your agent and adding the knowledge sources, the creation process typically takes about 15 minutes. After your agent is ready, you will see a chat interface on the right side, which you can use to start interacting with the agent. Alternatively, you can simply click on it to open and interact with it in the AI Playground. This allows you to immediately begin asking questions and testing the agent’s capabilities based on your uploaded documentation and support tickets.
# MAGIC
# MAGIC <img src=https://raw.githubusercontent.com/chen-data-ai/Agent-Bricks-Workshop/refs/heads/main/resources/screenshots/Screenshot_playground.png width="60%">
# MAGIC
# MAGIC Example Questions You Can Ask Your Agent
# MAGIC - What information is needed to add a line to my account?
# MAGIC - How long does it take to activate a new line?
# MAGIC - How can I prevent incorrect roaming charges from happening again?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Improve Quality
# MAGIC To ensure your Knowledge Assistant delivers accurate and relevant answers, you can continuously improve its performance by leveraging labeled data. With MLflow 3’s integrated labeling and feedback features (currently in Beta), you can submit questions to domain experts for review, creating a high-quality labeled dataset that guides further tuning and evaluation of your agent. By using expert-reviewed feedback, your assistant becomes more reliable, delivering better results for a wide range of customer inquiries.
# MAGIC
# MAGIC In your agent’s Bricks configuration screen, there is an “Improve Quality” button at the center of the top toolbar.
# MAGIC
# MAGIC <img src=https://raw.githubusercontent.com/chen-data-ai/Agent-Bricks-Workshop/refs/heads/main/resources/screenshots/Screenshot_improve_quality_0.png width="60%">
# MAGIC
# MAGIC You will see the following interface:
# MAGIC
# MAGIC <img src=https://raw.githubusercontent.com/chen-data-ai/Agent-Bricks-Workshop/refs/heads/main/resources/screenshots/Screenshot_improve_quality_1.png width="60%">
# MAGIC
# MAGIC Next, click the “Add” button to add a new question that you would like to label:
# MAGIC
# MAGIC <img src=https://raw.githubusercontent.com/chen-data-ai/Agent-Bricks-Workshop/refs/heads/main/resources/screenshots/Screenshot_improve_quality_2.png width="60%">
# MAGIC
# MAGIC After adding the question, you can begin the labeling session, where you may define guidelines or expectations and provide feedback as needed:
# MAGIC
# MAGIC <img src=https://raw.githubusercontent.com/chen-data-ai/Agent-Bricks-Workshop/refs/heads/main/resources/screenshots/Screenshot_improve_quality_4.png width="60%">

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # HANDS-ON: Create a Multi-Agent Supervisor (MAS) 
# MAGIC
# MAGIC Later, we're going to create a multi-agent supervisor using DSPy and then come back to Agent Bricks. Use this time to explore the UI, explore the existing functions/agents/tools in the workspace, and build an initial supervisor.
# MAGIC
# MAGIC We highly recommend reviewing and applying the Anthropic guide for tool calling: https://www.anthropic.com/engineering/writing-tools-for-agents.
# MAGIC
# MAGIC And while not critical to today's exercises, the prompt engineering guide is equally helpful: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview.

# COMMAND ----------

# MAGIC %md
# MAGIC Start by creating a new Multi-Agent Supervisor. You'll see a few pieces of information that you can configure in a name (make it something unique since we'll be creating a lot), description of the agent, up to 20 different agent/tools you can provide to the supervisor, and then optional instructions. 
# MAGIC
# MAGIC ![MAS Overview.png](./Screenshots/MAS Overview.png "MAS Overview.png")
# MAGIC
# MAGIC There is not a precise MAS you need to create now, use the time to experiment with different tool calling techniques and different assortments of tools. Below will list out the tools that exist in the workspace (or you can create your own!). The goal is not to achieve a specific MAS but to understand what setup looks like and what customers go through in identifying a use case and then building a tool to solve it.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overview of existing agents/tools in the workspace
# MAGIC At present, MAS can call Genie Spaces, Agent Endpoints, and UC Functions (with MCP very soon). You have access to the following tools in the workspace to experiment with now and into the hackathon:
# MAGIC - ####Genie Spaces
# MAGIC   - **Patient Lookup** which has fake patient information about visits to medical clinics
# MAGIC   - **Bakehouse Sales Start Space** which is the default Genie containing information about bakery sales and inventory
# MAGIC - ####Agent Endpoints
# MAGIC   - **ka-f67b8b54-endpoint** which is a technical customer support agent for a telecommunications company
# MAGIC     - Source files in /Volumes/genai_in_production_demo_catalog/customer_service/tech_support
# MAGIC   - **ka-2b918ccd-endpoint or ka-b77ac499-endpoint** which contains 10K financial documents for a number of large, enterprise companies
# MAGIC     - Source files in /Volumes/genai_in_production_demo_catalog/agents/dsa_volume
# MAGIC - ####UC Functions
# MAGIC   - **system.ai.python_exec** an inbuilt function to execute stateless Python code
# MAGIC   - **genai_in_production_demo_catalog.agents.search_web** which can search the web
# MAGIC
# MAGIC Of course, feel free to add in your own agent tools if you'd like to experiment, but the objective is to learn about MAS and how to set it up before we get deeper into evaluation.
# MAGIC
# MAGIC If helpful, use Austin's MAS **multi-agent-2025-10-15-14-40-47** for inspiration. If you'd like, you can also create your own KA using existing 10K financial PDFs or telecommunications customer service docs.
