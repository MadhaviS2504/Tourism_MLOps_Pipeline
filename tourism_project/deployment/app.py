import gradio as gr
import pandas as pd
import joblib
import os
import logging
from huggingface_hub import hf_hub_download

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model repository ID
MODEL_REPO_ID = "MadhaviSura/tourism-prediction-pipeline"
MODEL_FILENAME = "best_tourism_predictor.joblib"

# Function to download and load the model from Hugging Face Hub
def load_model_from_hf():
    try:
        logger.info(f"Loading model from {MODEL_REPO_ID}/{MODEL_FILENAME}")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
            token=os.getenv("HF_TOKEN")
        )
        model = joblib.load(model_path)
        logger.info(f"Model '{MODEL_FILENAME}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None  # Return None instead of raising exception

# Load the model once when the app starts
model = load_model_from_hf()

# Feature columns after preprocessing and one-hot encoding
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

def predict_purchase(
    age, city_tier, duration_of_pitch, monthly_income,
    number_of_followups, number_of_person_visiting,
    number_of_trips, pitch_satisfaction_score,
    preferred_property_star, passport, own_car,
    number_of_children_visiting, designation, gender,
    marital_status, occupation, product_pitched, type_of_contact
):
    
    if model is None:
        return "⚠️ Error: Model is not available. Please try again later.", ""

    try:
        # Initialize all feature columns to 0.0
        input_data = {col: 0.0 for col in feature_columns}

        # Fill in numerical features
        input_data['Age'] = float(age)
        input_data['CityTier'] = float(city_tier)
        input_data['DurationOfPitch'] = float(duration_of_pitch)
        input_data['MonthlyIncome'] = float(monthly_income)
        input_data['NumberOfFollowups'] = float(number_of_followups)
        input_data['NumberOfPersonVisiting'] = float(number_of_person_visiting)
        input_data['NumberOfTrips'] = float(number_of_trips)
        input_data['PitchSatisfactionScore'] = float(pitch_satisfaction_score)
        input_data['PreferredPropertyStar'] = float(preferred_property_star)
        input_data['Passport'] = 1.0 if passport else 0.0
        input_data['OwnCar'] = 1.0 if own_car else 0.0
        input_data['NumberOfChildrenVisiting'] = float(number_of_children_visiting)

        # Fill in one-hot encoded categorical features
        if f'Designation_{designation}' in input_data:
            input_data[f'Designation_{designation}'] = 1.0
        if 'Gender_Male' in input_data:
            input_data['Gender_Male'] = 1.0 if gender == 'Male' else 0.0
        if f'MaritalStatus_{marital_status}' in input_data:
            input_data[f'MaritalStatus_{marital_status}'] = 1.0
        if f'Occupation_{occupation}' in input_data:
            input_data[f'Occupation_{occupation}'] = 1.0
        if f'ProductPitched_{product_pitched}' in input_data:
            input_data[f'ProductPitched_{product_pitched}'] = 1.0
        if f'TypeofContact_{type_of_contact}' in input_data:
            input_data[f'TypeofContact_{type_of_contact}'] = 1.0

        # Create DataFrame and make prediction
        input_df = pd.DataFrame([input_data], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[:, 1][0]

        result_message = ""
        if prediction == 1:
            result_message = f"✅ **Likely to Purchase** (Confidence: {prediction_proba:.2%})"
        else:
            result_message = f"❌ **Unlikely to Purchase** (Confidence: {prediction_proba:.2%})"

        return result_message, input_df.to_html(index=False)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"⚠️ Error during prediction: {str(e)}", ""

# Define Gradio Interface inputs
gr_inputs = [
    gr.Slider(minimum=18, maximum=80, value=30, step=1, label="Age"),
    gr.Dropdown(choices=[1, 2, 3], value=1, label="City Tier"),
    gr.Slider(minimum=5, maximum=60, value=15, step=1, label="Duration of Pitch (minutes)"),
    gr.Number(value=50000.0, label="Monthly Income"),
    gr.Slider(minimum=0, maximum=10, value=3, step=1, label="Number of Follow-ups"),
    gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Number of Persons Visiting"),
    gr.Slider(minimum=0, maximum=20, value=5, step=1, label="Number of Trips Annually"),
    gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Pitch Satisfaction Score"),
    gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Preferred Property Star"),
    gr.Checkbox(label="Has Passport", value=False),
    gr.Checkbox(label="Owns Car", value=False),
    gr.Slider(minimum=0, maximum=5, value=0, step=1, label="Number of Children Visiting"),
    gr.Dropdown(choices=['Executive', 'Manager', 'Senior Manager', 'VP', 'Analyst', 'Engineer'], value='Executive', label="Designation"),
    gr.Dropdown(choices=['Male', 'Female'], value='Male', label="Gender"),
    gr.Dropdown(choices=['Single', 'Married', 'Divorced'], value='Single', label="Marital Status"),
    gr.Dropdown(choices=['Salaried', 'Freelancer', 'Small Business', 'Unemployed'], value='Salaried', label="Occupation"),
    gr.Dropdown(choices=['Deluxe', 'Standard', 'Super Deluxe', 'Basic', 'Platinum'], value='Standard', label="Product Pitched"),
    gr.Dropdown(choices=['Company Invited', 'Self Inquiry'], value='Company Invited', label="Type of Contact")
]

# Define Gradio Interface outputs
gr_outputs = [
    gr.Markdown(label="Prediction Result"),
    gr.HTML(label="Input Data for Prediction")
]

# Create and launch the Gradio Interface
iface = gr.Interface(
    fn=predict_purchase,
    inputs=gr_inputs,
    outputs=gr_outputs,
    title="Tourism Package Purchase Predictor",
    description="Enter customer details to predict if they will purchase the Wellness Tourism Package."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
