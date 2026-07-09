import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, hf_hub_download, create_repo

# Define constants
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "MadhaviSura/tourism-prediction-pipeline"
MODEL_REPO_ID = REPO_ID

# Initialize HfApi
api = HfApi(token=HF_TOKEN)

# --- Load Data ---
print("Loading train and test datasets from Hugging Face Hub...")
Xtrain = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename="Xtrain.csv", repo_type="dataset", token=HF_TOKEN))
Xtest = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename="Xtest.csv", repo_type="dataset", token=HF_TOKEN))
ytrain = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename="ytrain.csv", repo_type="dataset", token=HF_TOKEN))
ytest = pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename="ytest.csv", repo_type="dataset", token=HF_TOKEN))
print("Datasets loaded successfully.")

# Prepare target for evaluation
ytest_raveled = ytest.values.ravel()

# --- Define Models and Parameters ---
models_and_params = {
    'Decision Tree': (
        DecisionTreeClassifier(random_state=42),
        {
            'max_depth': [3, 5, 7, 10],
            'min_samples_leaf': [1, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10]
        }
    ),
    'XGBoost': (
        XGBClassifier(random_state=42, eval_metric='logloss'),
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    )
}

# --- MLflow Tracking and Model Tuning ---
# Set the MLflow tracking URI to point to the local MLflow UI server
mlflow.set_tracking_uri("http://localhost:5000")

best_overall_model = None
best_overall_score = -1
best_overall_model_name = ""

for model_name, (model, param_grid) in models_and_params.items():
    print(f"\n--- Tuning {model_name} ---")
    mlflow.set_experiment(f"Tourism_Prediction_{model_name.replace(' ', '_')}")
    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        mlflow.set_tag("model_type", model_name)

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(Xtrain, ytrain.values.ravel())

        current_best_model = grid_search.best_estimator_
        current_best_score = grid_search.best_score_

        print(f"Best parameters: {grid_search.best_params_}")
        mlflow.log_params(grid_search.best_params_)

        # Log the model with an input example to infer the signature
        mlflow.sklearn.log_model(current_best_model, "model", input_example=Xtrain.head(1))

        # Evaluation
        y_pred_proba = current_best_model.predict_proba(Xtest)[:, 1]
        roc_auc = roc_auc_score(ytest_raveled, y_pred_proba)
        mlflow.log_metrics({"test_roc_auc": roc_auc})
        print(f"Test ROC AUC: {roc_auc:.4f}")

        if roc_auc > best_overall_score:
            best_overall_score = roc_auc
            best_overall_model = current_best_model
            best_overall_model_name = model_name

print(f"\n--- Winner: {best_overall_model_name} with ROC AUC: {best_overall_score:.4f} ---")

# --- Register Best Model to Hugging Face ---
model_output_dir = "model_artifacts"
os.makedirs(model_output_dir, exist_ok=True)
model_path = os.path.join(model_output_dir, "best_tourism_predictor.joblib")
joblib.dump(best_overall_model, model_path)

try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
except Exception:
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_tourism_predictor.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)

print(f"Successfully uploaded {best_overall_model_name} to {MODEL_REPO_ID}")
