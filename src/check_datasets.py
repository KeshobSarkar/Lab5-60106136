from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml = MLClient(
    DefaultAzureCredential(),
    "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    "rg-60106136",
    "mltumor60106136"
)

print("DATA ASSETS:")
for d in ml.data.list():
    print(d.name, d.version)
