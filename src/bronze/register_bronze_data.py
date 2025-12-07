from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

ml = MLClient(
    DefaultAzureCredential(),
    "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    "rg-60106136",
    "<workspace>"
)

data = Data(
    name="tumor_images_raw",
    version="1",
    type="uri_folder",
    path="azureml://datastores/<your-store>/paths/raw/tumor_images/"
)

ml.data.create_or_update(data)
