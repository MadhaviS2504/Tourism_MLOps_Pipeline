import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import HfApi, hf_hub_download

# Define the model repository ID
MODEL_REPO_ID = "MadhaviSura/tourism-prediction-decision-tree"
MODEL_FILENAME = "best_tourism_predictor.joblib"

# Function to download the model from Hugging Face Hub
@st.cache_resource
def load_model_from_hf():
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        model = joblib.load(model_path)
        st.success(f"Model '{MODEL_FILENAME}' loaded successfully from Hugging Face Hub.")
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        st.stop()

# Load the model
model = load_model_from_hf()

# Title of the Streamlit app
st.title("Tourism Package Purchase Predictor")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Define the input fields based on the Xtrain columns
# Assuming these are the columns after preprocessing (Label Encoding applied)
# The order of these features must match the order the model was trained on.
feature_columns = [
    'Age', 'CityTier', 'DurationOfPitch', 'MonthlyIncome',
    'NumberOfFollowups', 'NumberOfPersonVisiting', 'NumberOfTrips',
    'PitchSatisfactionScore', 'PreferredPropertyStar', 'Passport', 'OwnCar',
    'NumberOfChildrenVisiting', 'Designation_Executive', 'Designation_Manager',
    'Designation_Senior Manager', 'Designation_VP', 'Gender_Male',
    'MaritalStatus_Married', 'MaritalStatus_Single', 'Occupation_Freelancer',
    'Occupation_Salaried', 'Occupation_Small Business', 'ProductPitched_Deluxe',
    'ProductPitched_Standard', 'ProductPitched_Super Deluxe', 'ProductPitched_Basic',
    'ProductPitched_Platinum', 'TypeofContact_Company Invited',
    'TypeofContact_Self Inquiry'
]

# Collect user input
with st.form("prediction_form"):
    st.subheader("Customer Information")
    age = st.slider("Age", 18, 80, 30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)
    number_of_followups = st.slider("Number of Follow-ups", 0, 10, 3)
    number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 10, 2)
    number_of_trips = st.slider("Number of Trips Annually", 0, 20, 5)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    preferred_property_star = st.slider("Preferred Property Star", 1, 5, 3)
    passport = st.checkbox("Has Passport")
    own_car = st.checkbox("Owns Car")
    number_of_children_visiting = st.slider("Number of Children Visiting", 0, 5, 0)

    st.subheader("Categorical Features")
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'VP', 'Analyst', 'Engineer'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Unemployed'])
    product_pitched = st.selectbox("Product Pitched", ['Deluxe', 'Standard', 'Super Deluxe', 'Basic', 'Platinum'])
    type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Create a dictionary for the input features, initializing all one-hot encoded columns to 0
        input_data = {
            col: 0 for col in feature_columns if any(cat_val in col for cat_val in ['Designation_', 'Gender_', 'MaritalStatus_', 'Occupation_', 'ProductPitched_', 'TypeofContact_'])
        }

        # Fill in numerical features
        input_data['Age'] = age
        input_data['CityTier'] = city_tier
        input_data['DurationOfPitch'] = duration_of_pitch
        input_data['MonthlyIncome'] = monthly_income
        input_data['NumberOfFollowups'] = number_of_followups
        input_data['NumberOfPersonVisiting'] = number_of_person_visiting
        input_data['NumberOfTrips'] = number_of_trips
        input_data['PitchSatisfactionScore'] = pitch_satisfaction_score
        input_data['PreferredPropertyStar'] = preferred_property_star
        input_data['Passport'] = 1 if passport else 0
        input_data['OwnCar'] = 1 if own_car else 0
        input_data['NumberOfChildrenVisiting'] = number_of_children_visiting

        # Fill in one-hot encoded categorical features
        input_data[f'Designation_{designation}'] = 1 if f'Designation_{designation}' in input_data else 0
        input_data[f'Gender_{gender}'] = 1 if f'Gender_{gender}' in input_data else 0
        input_data[f'MaritalStatus_{marital_status}'] = 1 if f'MaritalStatus_{marital_status}' in input_data else 0
        input_data[f'Occupation_{occupation}'] = 1 if f'Occupation_{occupation}' in input_data else 0
        input_data[f'ProductPitched_{product_pitched}'] = 1 if f'ProductPitched_{product_pitched}' in input_data else 0
        input_data[f'TypeofContact_{type_of_contact}'] = 1 if f'TypeofContact_{type_of_contact}' in input_data else 0

        # Create a DataFrame from the input data, ensuring correct column order
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[:, 1][0]

        st.subheader("Prediction Results")
        if prediction == 1:
            st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba:.2f})")
        else:
            st.warning(f"The customer is unlikely to purchase the package. (Probability: {prediction_proba:.2f})")

        st.write("--- Input Data ---")
        st.dataframe(input_df)
