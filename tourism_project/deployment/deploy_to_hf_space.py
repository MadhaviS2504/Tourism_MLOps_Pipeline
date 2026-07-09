from huggingface_hub import HfApi, create_repo
import os

FRONTEND_REPO_ID = "MadhaviSura/tourism-prediction-app" # Replace with your desired Space ID
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the Hugging Face Space if it doesn't exist
try:
    api.repo_info(repo_id=FRONTEND_REPO_ID, repo_type=repo_type)
    print(f"Space '{FRONTEND_REPO_ID}' already exists. Using it.")
except Exception as e:
    print(f"Space '{FRONTEND_REPO_ID}' not found or inaccessible: {e}")
    print("\nHugging Face Spaces now require a Pro subscription to create new Docker SDK spaces via the API.")
    print("To proceed with a free CPU-basic Streamlit Space, please create it manually on the Hugging Face website:")
    print(f"1. Go to https://huggingface.co/new-space")
    print(f"2. Set the Space name to '{FRONTEND_REPO_ID.split('/')[-1]}'")
    print(f"3. Select the owner ('{FRONTEND_REPO_ID.split('/')[0]}').")
    print(f"4. Choose 'Gradio' as the Space SDK and 'CPU Basic' as the Hardware (this should be free).")
    print(f"5. After creating the space, re-run this script to upload the files.")
    # Exit the script to prevent further errors when the space isn't created
    import sys
    sys.exit(1)

# Create a README.md file for the Hugging Face Space to explicitly define SDK and disable GPU
readme_content = f"""
---
title: Tourism Prediction App
emoji: 🚀
colorFrom: yellow
colorTo: green
sdk: gradio
app_file: app.py
gpu: false
---\n# Tourism Prediction App\n\nThis is a Gradio application for predicting whether a customer will purchase a tourism package.
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
