# for data manipulation
import pandas as pd
import numpy as np
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for encoding categorical features
from sklearn.preprocessing import OneHotEncoder
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, hf_hub_download

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
REPO_ID = "MadhaviSura/tourism-prediction-pipeline"

# Create a temporary directory to store downloaded raw data
local_raw_data_dir = "./hf_downloaded_raw_data"
os.makedirs(local_raw_data_dir, exist_ok=True)

# Download and load the main dataset
tourism_csv_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="tourism.csv",
    repo_type="dataset",
    local_dir=local_raw_data_dir,
    token=api.token
)
df = pd.read_csv(tourism_csv_path)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
    print("Dropped 'Unnamed: 0' column.")

target_col = 'ProdTaken'

# --- Initial Data Cleaning (before split for consistent categories) ---

# Standardize 'MaritalStatus' column
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')
print("Standardized 'MaritalStatus' values.")

# Standardize 'Gender' column
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
print("Standardized 'Gender' values.")

# --- Initial Split to prevent data leakage from imputation and encoding ---

X_full = df.drop(columns=[target_col])
y_full = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full # Stratify if target is imbalanced
)

# Reset indices to avoid potential issues after splitting
Xtrain = Xtrain.reset_index(drop=True)
Xtest = Xtest.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
ytest = ytest.reset_index(drop=True)

print("Initial train-test split complete.")

# --- Data Cleaning (Imputation) - using ONLY training data statistics ---

# Impute numerical columns with median
numerical_cols_for_imputation = Xtrain.select_dtypes(include=np.number).columns.tolist()

# we can exclude binary columns(1,0) and already clean columns from median imputation if they are not truly continuous
exclude_from_median_imputation = ['Passport', 'OwnCar', 'CityTier', 'PitchSatisfactionScore', 'PreferredPropertyStar']
for col in [c for c in numerical_cols_for_imputation if c not in exclude_from_median_imputation]:
    if Xtrain[col].isnull().any():
        median_val = Xtrain[col].median()
        Xtrain[col].fillna(median_val, inplace=True)
        Xtest[col].fillna(median_val, inplace=True) # Apply same median to test
        print(f"Filled missing values in '{col}' with training median: {median_val}")

# Impute categorical columns with mode
categorical_cols_for_imputation = Xtrain.select_dtypes(include='object').columns.tolist()
for col in categorical_cols_for_imputation:
    if Xtrain[col].isnull().any():
        mode_val = Xtrain[col].mode()[0]
        Xtrain[col].fillna(mode_val, inplace=True)
        Xtest[col].fillna(mode_val, inplace=True) # Apply same mode to test
        print(f"Filled missing values in '{col}' with training mode: {mode_val}")

print("Missing values handled without data leakage.")

# --- Feature Engineering - apply consistently to both train and test ---

# Create 'IncomePerPerson' feature
# Handle potential division by zero by replacing 0 with 1 for NumberOfPersonVisiting
Xtrain['IncomePerPerson'] = Xtrain['MonthlyIncome'] / Xtrain['NumberOfPersonVisiting'].replace(0, 1)
Xtest['IncomePerPerson'] = Xtest['MonthlyIncome'] / Xtest['NumberOfPersonVisiting'].replace(0, 1)
print("Created 'IncomePerPerson' feature.")

# Create 'HasChildren' feature
Xtrain['HasChildren'] = (Xtrain['NumberOfChildrenVisiting'] > 0).astype(int)
Xtest['HasChildren'] = (Xtest['NumberOfChildrenVisiting'] > 0).astype(int)
print("Created 'HasChildren' feature.")

# Create 'AgeGroup' feature
age_bins = [0, 18, 30, 45, 60, 100]
age_labels = ['<18', '18-30', '31-45', '46-60', '60+']
Xtrain['AgeGroup'] = pd.cut(Xtrain['Age'], bins=age_bins, labels=age_labels, right=False)
Xtest['AgeGroup'] = pd.cut(Xtest['Age'], bins=age_bins, labels=age_labels, right=False)
print("Created 'AgeGroup' feature.")

# New Feature: IsMajorCity (assuming CityTier 1 is major)
Xtrain['IsMajorCity'] = (Xtrain['CityTier'] == 1).astype(int)
Xtest['IsMajorCity'] = (Xtest['CityTier'] == 1).astype(int)
print("Created 'IsMajorCity' feature.")

# New Feature: PitchEffectiveness (Ratio of PitchSatisfactionScore to NumberOfFollowups)
# Handle potential division by zero for NumberOfFollowups
Xtrain['PitchEffectiveness'] = Xtrain['PitchSatisfactionScore'] / Xtrain['NumberOfFollowups'].replace(0, 1)
Xtest['PitchEffectiveness'] = Xtest['PitchSatisfactionScore'] / Xtest['NumberOfFollowups'].replace(0, 1)
print("Created 'PitchEffectiveness' feature.")

# --- Drop original 'Age' column if AgeGroup is preferred as categorical ---
Xtrain.drop(columns=['Age'], inplace=True)
Xtest.drop(columns=['Age'], inplace=True)
print("Dropped original 'Age' column, 'AgeGroup' will be used as a categorical feature.")

# --- One-Hot Encoding Categorical Features ---

# Identify numerical and categorical columns AFTER feature engineering and dropping original 'Age'
numerical_features = Xtrain.select_dtypes(include=np.number).columns.tolist()
categorical_features = Xtrain.select_dtypes(include='object').columns.tolist()

# Initialize OneHotEncoder
# handle_unknown='ignore' will set columns for unknown categories to all zeros.
# sparse_output=False ensures dense array output for concatenation.
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit on training data and transform both train and test
onehot_encoder.fit(Xtrain[categorical_features])

Xtrain_encoded = onehot_encoder.transform(Xtrain[categorical_features])
Xtest_encoded = onehot_encoder.transform(Xtest[categorical_features])

# Get feature names for encoded columns
encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)

# Convert to DataFrames
Xtrain_encoded_df = pd.DataFrame(Xtrain_encoded, columns=encoded_feature_names, index=Xtrain.index)
Xtest_encoded_df = pd.DataFrame(Xtest_encoded, columns=encoded_feature_names, index=Xtest.index)

# Drop original categorical columns and concatenate with encoded ones
Xtrain = pd.concat([Xtrain[numerical_features], Xtrain_encoded_df], axis=1)
Xtest = pd.concat([Xtest[numerical_features], Xtest_encoded_df], axis=1)

print("Categorical features one-hot encoded.")
print(f"Final Xtrain shape: {Xtrain.shape}")
print(f"Final Xtest shape: {Xtest.shape}")


# Create a directory to save the processed data locally before uploading
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

Xtrain.to_csv(os.path.join(output_dir, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(output_dir, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(output_dir, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(output_dir, "ytest.csv"), index=False)

print("Processed data saved locally.")

# Upload processed data to Hugging Face Hub
files_to_upload = [
    os.path.join(output_dir, "Xtrain.csv"),
    os.path.join(output_dir, "Xtest.csv"),
    os.path.join(output_dir, "ytrain.csv"),
    os.path.join(output_dir, "ytest.csv")
]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),  # just the filename
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded {os.path.basename(file_path)} to Hugging Face Hub.")
