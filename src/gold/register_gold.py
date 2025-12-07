from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

# ----------------------------
# WORKSPACE CONFIG
# ----------------------------
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106136"
WORKSPACE_NAME = "mltumor60106136"

# ----------------------------
# Connect to MLClient
# ----------------------------
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# ----------------------------
# Define the environment
# ----------------------------
env = Environment(
    name="gold_reg",
    version="10",
    description="Environment for Gold Layer pipeline (feature selection + GA + training)",
    conda_file="tumor-env.yml",   # must be present in same folder
    image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py39-cpu-inference:latest",
)

# ----------------------------
# Register environment
# ----------------------------
registered_env = ml_client.environments.create_or_update(env)

print("✔ Environment registered successfully")
print(f"Name: {registered_env.name}")
print(f"Version: {registered_env.version}")
print(f"ID: {registered_env.id}")
