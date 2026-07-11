from huggingface_hub import HfApi
import os
repo_id = "Chandrashekhara/tourism-project"

token = os.getenv("HF_TOURISM")
api = HfApi(token=token)

api.upload_folder(
    folder_path="tourism_model/deployment",     # the local folder containing your files
    repo_id=repo_id,                            # the target repo
    repo_type="space",                          # dataset, model, or space
    path_in_repo="",                            # optional: subfolder path inside the repo
    token=token
)
