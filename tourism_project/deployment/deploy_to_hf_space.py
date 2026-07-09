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

# Create a README.md file for the Hugging Face Space to explicitly define SDK and disable GPU
readme_content = f"""
---
title: Tourism Prediction App
emoji: 🚀
colorFrom: yellow
colorTo: green
sdk: docker
app_file: app.py
gpu: false
---
# Tourism Prediction App

This is a Streamlit application for predicting whether a customer will purchase a tourism package.
"""

readme_path = os.path.join("tourism_project/deployment", "README.md")
with open(readme_path, "w") as f:
    f.write(readme_content)
print(f"Generated README.md at {readme_path}")

# Upload all files from the 'deployment' directory to the Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=FRONTEND_REPO_ID,
    repo_type=repo_type,
)
print(f"Deployment files uploaded to Hugging Face Space '{FRONTEND_REPO_ID}'.")
