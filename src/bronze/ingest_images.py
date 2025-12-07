import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# -------------------------------
# CONFIG
# -------------------------------
ACCOUNT_URL = "https://tumor60106136.blob.core.windows.net"
CONTAINER = "raw"
KEY = "Pw89Tp8aJhmM61d0oCFI8B66gdHIut8436q7xqku1ed8YP72cL4MAGvlqNn16wV1rBXGw1MApk5B+AStojpVKw=="

LOCAL_DATASET = r"C:/Users/kesho/Documents/cloudpor/lab5/Data and Reference Code/assignment/data/brain_tumor_dataset"

blob_service = BlobServiceClient(account_url=ACCOUNT_URL, credential=KEY)
container_client = blob_service.get_container_client(CONTAINER)

def upload_if_missing(blob_path, local_fp):
    """Upload blob only if not already present (idempotent)."""
    blob_client = container_client.get_blob_client(blob_path)

    if blob_client.exists():
        print(f"[SKIP] Already exists: {blob_path}")
        return

    with open(local_fp, "rb") as f:
        blob_client.upload_blob(f)
        print(f"[UPLOADED] {blob_path}")


def ingest():
    for label in ["yes", "no"]:
        folder = os.path.join(LOCAL_DATASET, label)
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            local_fp = os.path.join(folder, fname)

            # Simulate directories using blob prefixes
            blob_path = f"tumor_images/{label}/{fname}"

            upload_if_missing(blob_path, local_fp)

    print("Ingestion completed.")


if __name__ == "__main__":
    ingest()
