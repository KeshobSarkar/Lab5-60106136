from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.dsl import pipeline

SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106136"
WORKSPACE_NAME = "mltumor60106136"

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

component_version = "2025-12-06-19-13-59-7184351"

extract_features = ml_client.components.get(
    name="extract_features_component",
    version=component_version
)

@pipeline()
def silver_pipeline(input_data):
    step = extract_features(input_data=input_data)
    return {"output_features": step.outputs.output_features}

job = silver_pipeline(
    input_data = Input(
        type="uri_folder",
        path="azureml:bronze_data:1"
    )

)

job.settings.default_compute = "computelayer"

submitted = ml_client.jobs.create_or_update(job)
print("PIPELINE SUBMITTED:", submitted.name)
