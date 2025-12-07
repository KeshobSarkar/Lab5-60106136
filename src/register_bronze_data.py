from azure.ai.ml import MLClient, Data
from azure.identity import DefaultAzureCredential

ml = MLClient(
    DefaultAzureCredential(),
    "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    "rg-60106136",
    "mltumor60106136"
)

data = Data(
    name="bronze_data",
    version="1",
    type="uri_folder",
    path="https://tumor60106136.blob.core.windows.net/raw/tumor_images"
)

ml.data.create_or_update(data)
print("bronze_data version 1 registered!")
