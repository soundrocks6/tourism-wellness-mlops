# -----------------------------------------------------------------------------
# app.py
# Streamlit frontend for the Wellness Tourism Package purchase predictor.
# -----------------------------------------------------------------------------

import os

import joblib
import pandas as pd
import streamlit as st

# ---- Page config MUST be the first Streamlit command ----
st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🌴")

# ---- Load the trained model committed to the repository by the pipeline ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---- Page header ----
st.title("🌴 Visit with Us — Wellness Tourism Package Predictor")
st.write(
    "Enter the customer's details below. The model predicts whether the "
    "customer is likely to purchase the newly introduced **Wellness Tourism Package**, "
    "helping the marketing team contact the right customers first."
)

# ---- Customer Details ----
st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    maritalstatus = st.selectbox(
        "Marital Status", ["Single", "Unmarried", "Married", "Divorced"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthlyincome = st.number_input(
        "Monthly Income", min_value=1000.0, max_value=100000.0, value=22000.0, step=500.0
    )

with col2:
    numberofpersonvisiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=3
    )
    numberofchildrenvisiting = st.number_input(
        "Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=1
    )
    preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    numberoftrips = st.number_input(
        "Average Number of Trips per Year", min_value=0, max_value=25, value=3
    )
    passport = st.selectbox("Holds a Valid Passport?", ["Yes", "No"])
    owncar = st.selectbox("Owns a Car?", ["Yes", "No"])

# ---- Customer Interaction Data ----
st.header("Customer Interaction Data")
col3, col4 = st.columns(2)

with col3:
    productpitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

with col4:
    numberoffollowups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=4
    )
    durationofpitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=60, value=15
    )

# ---- Collect the inputs into a dataframe (same schema used in training) ----
input_df = pd.DataFrame(
    [
        {
            "Age": float(age),
            "TypeofContact": typeofcontact,
            "CityTier": int(citytier),
            "DurationOfPitch": float(durationofpitch),
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": int(numberofpersonvisiting),
            "NumberOfFollowups": float(numberoffollowups),
            "ProductPitched": productpitched,
            "PreferredPropertyStar": float(preferredpropertystar),
            "MaritalStatus": maritalstatus,
            "NumberOfTrips": float(numberoftrips),
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": int(pitchsatisfactionscore),
            "OwnCar": 1 if owncar == "Yes" else 0,
            "NumberOfChildrenVisiting": float(numberofchildrenvisiting),
            "Designation": designation,
            "MonthlyIncome": float(monthlyincome),
        }
    ]
)

st.subheader("Input Summary")
st.dataframe(input_df)

# ---- Predict and display the result ----
if st.button("Predict Purchase"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(
            f"✅ This customer is **likely to purchase** the Wellness Tourism Package "
            f"(purchase probability: {probability:.2%}). Prioritise them for outreach!"
        )
    else:
        st.warning(
            f"❌ This customer is **unlikely to purchase** the Wellness Tourism Package "
            f"(purchase probability: {probability:.2%})."
        )
