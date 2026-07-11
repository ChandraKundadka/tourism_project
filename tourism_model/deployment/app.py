import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub

repo_id = "Chandrashekhara/tourism-project-model"
model_path = hf_hub_download(repo_id=repo_id, filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("Model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.")
st.write("Kindly enter the customer details to check whether they are likely to purchase Wellness Tourism Package.")

# Collect user input
# -------------------------------
# Numerical Inputs
# -------------------------------
Age                     = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier                = st.selectbox("City Tier", options=[1, 2, 3], help="Tier 1 > Tier 2 > Tier 3")
DurationOfPitch         = st.number_input("Duration Of Pitch (minutes)", min_value=0, value=300)
NumberOfPersonVisiting  = st.number_input("Number Of Persons Visiting", min_value=1, value=100)
NumberOfFollowups       = st.number_input("Number Of Followups", min_value=0, value=10)
PreferredPropertyStar   = st.selectbox("Preferred Property Star",options=[1, 2, 3, 4, 5])
NumberOfTrips           = st.number_input("Number Of Trips per year", min_value=0, value=100)
Passport                = st.radio("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PitchSatisfactionScore  = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar                  = st.radio("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting (<5 yrs)", min_value=0, value=10)
MonthlyIncome           = st.number_input("Monthly Income", min_value=0, value=300000)

# -------------------------------
# Categorical Inputs
# -------------------------------
TypeofContact   = st.selectbox("Type of Contact",options=["Company Invited", "Self Inquiry"])
Occupation      = st.selectbox("Occupation",options=["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender          = st.selectbox("Gender",options=["Male", "Female"])
ProductPitched  = st.selectbox("Product Pitched",options=["Basic", "Deluxe", "King","Standard", "Super Deluxe"])
MaritalStatus   = st.selectbox("Marital Status",options=["Single", "Married", "Divorced"])
Designation     = st.selectbox("Designation",options=["AVP", "Executive", "Manager", "Senior Manager","VP"])

# -------------------------------
# Collect Input into Dictionary
# -------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome

}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "likely to purchase Wellness Tourism Package " if prediction == 1 else "Not likely to purchase Wellness Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
