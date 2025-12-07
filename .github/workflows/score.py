import json
import pandas as pd
import mlflow.pyfunc

def init():
    global model
    # Azure ML mounts model under /var/azureml-app/azureml-models
    model = mlflow.pyfunc.load_model("./")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame([data])
        pred = model.predict(df)
        return {"prediction": str(pred[0])}
    except Exception as e:
        return {"error": str(e)}
