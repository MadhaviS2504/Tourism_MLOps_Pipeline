from huggingface_hub import HfApi
import os
import shutil

api = HfApi(token=os.getenv("HF_TOKEN"))

source_deployment_dir = "tourism_project/deployment"
temp_upload_dir = "tourism_project/deployment_hf_space"

# Create a temporary directory for files to upload
os.makedirs(temp_upload_dir, exist_ok=True)

# Copy ONLY app.py, requirements.txt, and README.md (NO Dockerfile)
shutil.copy(os.path.join(source_deployment_dir, "app.py"), temp_upload_dir)
shutil.copy(os.path.join(source_deployment_dir, "requirements.txt"), temp_upload_dir)
shutil.copy(os.path.join(source_deployment_dir, "README.md"), temp_upload_dir)

# Upload files to Hugging Face Space
api.upload_folder(
    folder_path=temp_upload_dir,
    repo_id="MadhaviSura/tourism-app",
    repo_type="space",
    path_in_repo="",
)

# Clean up
shutil.rmtree(temp_upload_dir)

print("✅ Successfully uploaded app.py, requirements.txt, and README.md to Hugging Face Space.")
print("📋 Dockerfile is excluded to avoid GPU detection conflicts.")
print("🚀 Hugging Face will now auto-rebuild the Space with CPU-only setup.")
