# Databricks notebook source
# MAGIC %md
# MAGIC #Reminder
# MAGIC
# MAGIC You do need to manually create the Genie Space after creating these tables in the UI

# COMMAND ----------

pip install faker

# COMMAND ----------

import pandas as pd
import random
from faker import Faker

CATALOG = "genai_in_production_demo_catalog"
SCHEMA = "agents"

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.patient_visits")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.practice_locations")

# COMMAND ----------

fake = Faker()

# Define fixed options
patients = [fake.unique.first_name() + " " + fake.unique.last_name() for _ in range(15)]
insurance_providers = ["BlueCross", "Aetna", "UnitedHealth"]
insurance_types = ["HMO", "PPO", "EPO"]
reasons_for_visit = [
    "Routine Checkup", "Flu Symptoms", "Injury", "Chronic Condition", 
    "Follow-up", "Prescription Refill", "Surgery", "Physical Therapy"
]
cities = ["LA", "Chicago", "NY"]

# Assign a random but fixed insurance provider and type to each patient
patient_insurance = {patient: random.choice(insurance_providers) for patient in patients}
patient_insurance_type = {patient: random.choice(insurance_types) for patient in patients}

# Generate patient visit data
data = []
for _ in range(300):
    patient = random.choice(patients)
    first_name, last_name = patient.split(" ")
    insurance_provider = patient_insurance[patient]
    insurance_type = patient_insurance_type[patient]
    policy_number = fake.uuid4() if random.random() > 0.2 else None  # 80% chance to have a policy number
    email = fake.email()
    city = random.choice(cities)
    practice_id = fake.random_int(min=1000, max=9999)
    doctor_notes = fake.sentence()
    reason_for_visit = random.choice(reasons_for_visit)

    data.append([
        first_name, last_name, insurance_provider, insurance_type, policy_number,
        email, city, practice_id, doctor_notes, reason_for_visit
    ])

# Create DataFrame
columns = [
    "first_name", "last_name", "insurance_provider_name", "insurance_type",
    "insurance_policy_number", "email", "city", "practice_visited_practice_id",
    "doctor_notes", "reason_for_visit"
]
patients_visits_df = pd.DataFrame(data, columns=columns)

# Convert to Spark DataFrame
patients_visits_spark_df = spark.createDataFrame(patients_visits_df)

# Save DataFrame to specified catalog and schema
patients_visits_spark_df.write.saveAsTable(f"{CATALOG}.{SCHEMA}.patient_visits")

# COMMAND ----------

# Practice Location Table 

import random

# Define possible values
cities = ["LA", "Chicago", "NY"]
insurance_providers = ["BlueCross", "Aetna", "UnitedHealth"]
insurance_plan_types = ["HMO", "PPO", "EPO"]
network_statuses = ["In Network", "Out of Network"]

# Generate sample data
num_entries = 50
data = {
    "practice_name": [f"Medical Center {i}" for i in range(1, num_entries + 1)],
    "city": [random.choice(cities) for _ in range(num_entries)],
    "contact": [f"(555) 555-12{str(i).zfill(2)}" for i in range(num_entries)],
    "insurance_id": [f"INS-{random.randint(1000, 9999)}" for _ in range(num_entries)],
    "insurance_company": [random.choice(insurance_providers) for _ in range(num_entries)],
    "insurance_plan_type": [random.choice(insurance_plan_types) for _ in range(num_entries)],
    "network_status": [random.choice(network_statuses) for _ in range(num_entries)],
}

# Create DataFrame
practice_locations_df = spark.createDataFrame(pd.DataFrame(data))

# Save DataFrame to specified catalog and schema
practice_locations_df.write.saveAsTable(f"{CATALOG}.{SCHEMA}.practice_locations")

# COMMAND ----------

