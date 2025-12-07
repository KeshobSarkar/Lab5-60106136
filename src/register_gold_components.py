from azure.ai.ml import MLClient, load_component
from azure.identity import DefaultAzureCredential

SUB = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RG = "rg-60106136"
WS = "ml_60106136_tumor"

ml = MLClient(DefaultAzureCredential(), SUB, RG, WS)

paths = [
    "./gold/feature_retrieval/component.yaml",
    "./gold/feature_selection/component.yaml",
    "./gold/train_eval/component.yaml",
]

for p in paths:
    comp = load_component(p)
    comp.version = None   # auto-versioning
    reg = ml.components.create_or_update(comp)
    print("Registered:", reg.name, "version:", reg.version)
