from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

print("End of Data Register")

repo_id = "Chandrashekhara/tourism-project"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOURISM"))
print(os.getenv("HF_TOURISM"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project_local/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("End of Data Register")
