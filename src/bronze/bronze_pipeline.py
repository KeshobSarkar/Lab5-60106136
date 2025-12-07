from azure.ai.ml import MLClient, Data
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    resource_group_name="rg-60106136",
    workspace_name="ml_60106136_tumor"
)

my_bronze = Data(
    name="tumor_60106136_data",
    version="1",
    type="uri_folder",
    path="azureml://datastores/workspaceblobstore/paths/raw/tumor_images/"
)

ml_client.data.create_or_update(my_bronze)

print("Bronze dataset registered!")
