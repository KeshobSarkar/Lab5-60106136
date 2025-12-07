from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# ----------------------------
# CONFIG – THINGS THAT ARE YOURS
# ----------------------------
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106136"
WORKSPACE_NAME = "mltumor60106136"

# Your Azure ML compute cluster name
COMPUTE_TARGET = "computelayer"

# Silver features data asset (acting as your feature store output)
SILVER_DATA_ASSET_NAME = "silver_data"
SILVER_DATA_VERSION = "1"

# ----------------------------
# MLClient
# ----------------------------
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# ----------------------------
# Load components (YAMLs under src/gold)
# ----------------------------
feature_retrieval = load_component("./feature_retrieval/component.yaml")
feature_selection = load_component("./feature_selection/component.yaml")
train_eval = load_component("./train_eval/component.yaml")

# ----------------------------
# Pipeline definition
# ----------------------------
@pipeline()
def gold_pipeline(silver_features: Input(type="uri_file")):
    # Step 1: Feature retrieval (read Silver parquet -> train/test parquet)
    step1 = feature_retrieval(silver_features=silver_features)
    step1.compute = COMPUTE_TARGET

    # Step 2: Baseline + GA feature selection
    step2 = feature_selection(train_input=step1.outputs.train_output)
    step2.compute = COMPUTE_TARGET

    # Step 3: Train + evaluate with GA-selected features
    step3 = train_eval(
        train_input=step1.outputs.train_output,
        test_input=step1.outputs.test_output,
        selected_features=step2.outputs.selected_features,
    )
    step3.compute = COMPUTE_TARGET

    # ✅ RETURN NODE OUTPUTS
    return {
        "train_output": step1.outputs.train_output,
        "test_output": step1.outputs.test_output,
        "selected_features": step2.outputs.selected_features,
        "baseline_metrics": step2.outputs.baseline_metrics,
        "ga_metrics": step2.outputs.ga_metrics,
        "metrics_output": step3.outputs.metrics_output,
        "model_output": step3.outputs.model_output,   # <-- registered model
    }

# ----------------------------
# Build + submit job
# ----------------------------
job = gold_pipeline(
    silver_features=Input(
        type="uri_file",
        path=f"azureml:{SILVER_DATA_ASSET_NAME}:{SILVER_DATA_VERSION}",
    )
)

returned = ml_client.jobs.create_or_update(job, experiment_name="gold_pipeline")
print("✅ Pipeline submitted:", returned.name)
