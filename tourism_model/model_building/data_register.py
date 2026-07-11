from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

print("Starting the Data Register")

repo_id = "Chandrashekhara/tourism-project"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOURISM"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# data_register.py is inside tourism_model/model_building/
# so parent.parent is tourism_model/
folder_path = Path(__file__).resolve().parent.parent / "data"

print(f"Uploading from: {folder_path}")

if not folder_path.is_dir():
    raise FileNotFoundError(f"Data directory not found: {folder_path}")

api.upload_folder(
    folder_path="tourism_model/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("End of Data Register")
