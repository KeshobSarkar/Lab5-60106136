from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml = MLClient(
    DefaultAzureCredential(),
    "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    "rg-60106136",
    "mltumor60106136"
)

comp = ml.components.get("extract_features_component", "2025-12-05-11-36-19-1840663")

print("Inputs:", comp.inputs)
print("Outputs:", comp.outputs)
print("COMMAND:", comp.command)
