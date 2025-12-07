from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
import datetime

SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106136"
WORKSPACE_NAME = "mltumor60106136"

ENDPOINT_NAME = "tumor-detector-endpoint"

client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# ----------------------------
# Get latest registered model
# ----------------------------
models = client.models.list(name="train_eval_component", order_by="createdtime desc")
latest_model = next(models)
print("Using model:", latest_model.id)

# ----------------------------
# Create or update endpoint
# ----------------------------
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    auth_mode="key",
    description="Tumor detection endpoint",
)

client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"✔ Endpoint ready: {ENDPOINT_NAME}")

# ----------------------------
# Create BLUE deployment
# ----------------------------
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=ENDPOINT_NAME,
    model=latest_model.id,
    environment="azureml:gold_reg:10",
    code_path="./",
    scoring_script="score.py",
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

client.online_deployments.begin_create_or_update(deployment).result()

# Set BLUE as default
endpoint.defaults = {"deployment_name": "blue"}
client.online_endpoints.begin_create_or_update(endpoint)

print("✔ Deployment completed successfully")
print("✔ Endpoint is LIVE")
