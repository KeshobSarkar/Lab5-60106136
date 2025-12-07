from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component

# ---------------------------------------------------
# Azure ML Workspace details
# ---------------------------------------------------
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106136"
WORKSPACE_NAME = "mltumor60106136"

# ---------------------------------------------------
# Connect to Azure ML
# ---------------------------------------------------
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# ---------------------------------------------------
# Load component YAML (relative to lab5 folder)
# ---------------------------------------------------
component_path = "src/silver/extract_features/component.yaml"
print("Loading component from:", component_path)

extract_component = load_component(component_path)

# Force creation of a new version every run
extract_component.version = None

# ---------------------------------------------------
# Register (upload) the component to Azure ML
# ---------------------------------------------------
registered = ml_client.components.create_or_update(extract_component)

print("\n----------------------------------------")
print("Component registered successfully!")
print("Name:", registered.name)
print("Version:", registered.version)
print("----------------------------------------\n")
