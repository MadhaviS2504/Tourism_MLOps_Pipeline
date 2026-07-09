import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_USERNAME = "MadhaviSura"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-prediction-pipeline"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set as a Space secret

FEATURE_ORDER = [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
    "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]
CAT_COLS = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

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



def main():
    st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
    st.title("🧳 Wellness Tourism Package — Purchase Predictor")
    st.write(
        "Predict whether a customer is likely to purchase the newly launched "
        "**Wellness Tourism Package**, based on their profile and past interactions."
    )

    model = load_model_from_hf()

    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        type_of_contact = st.selectbox("Type of Contact", encoders["TypeofContact"].classes_)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", encoders["Occupation"].classes_)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)
        num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
        product_pitched = st.selectbox("Product Pitched", encoders["ProductPitched"].classes_)
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

    with col2:
        marital_status = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
        num_trips = st.number_input("Avg Number of Trips per Year", min_value=0, max_value=20, value=3)
        passport = st.selectbox("Has Passport?", ["No", "Yes"])
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
        num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", encoders["Designation"].classes_)
        monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)

    if st.button("Predict"):
        # Get inputs and save them into a dataframe (rubric requirement)
        input_df = pd.DataFrame([{
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_person_visiting,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": pitch_satisfaction,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": num_children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])[FEATURE_ORDER]

        # Encode categoricals with the same encoders used at training time
        encoded_df = input_df.copy()
        for col in CAT_COLS:
            le = encoders[col]
            encoded_df[col] = encoded_df[col].apply(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

        prediction = model.predict(encoded_df)[0]
        proba = (
            model.predict_proba(encoded_df)[0][1]
            if hasattr(model, "predict_proba") else None
        )

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"✅ Likely to PURCHASE the Wellness Tourism Package"
                       + (f" (probability: {proba:.2%})" if proba is not None else ""))
        else:
            st.warning(f"❌ Unlikely to purchase the Wellness Tourism Package"
                       + (f" (probability: {proba:.2%})" if proba is not None else ""))

        with st.expander("View input data sent to the model"):
            st.dataframe(input_df)


if __name__ == "__main__":
    main()

