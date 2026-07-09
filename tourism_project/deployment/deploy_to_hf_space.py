from huggingface_hub import HfApi, create_repo
import os

FRONTEND_REPO_ID = "MadhaviSura/tourism-prediction-app" # Replace with your desired Space ID
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the Hugging Face Space if it doesn't exist
try:
    api.repo_info(repo_id=FRONTEND_REPO_ID, repo_type=repo_type)
    print(f"Space '{FRONTEND_REPO_ID}' already exists. Using it.")
except Exception:
    print(f"Space '{FRONTEND_REPO_ID}' not found. Creating new space...")
    create_repo(repo_id=FRONTEND_REPO_ID, repo_type=repo_type, private=False, space_sdk="docker") # Changed space_sdk to 'docker'
    print(f"Space '{FRONTEND_REPO_ID}' created.")

# Upload all files from the 'deployment' directory to the Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=FRONTEND_REPO_ID,
    repo_type=repo_type,
)
print(f"Deployment files uploaded to Hugging Face Space '{FRONTEND_REPO_ID}'.")
