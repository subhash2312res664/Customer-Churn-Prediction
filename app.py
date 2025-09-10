import streamlit as st
import pandas as pd
import joblib
import os

# Define the list of features the model expects, in the correct order
# This should match the features used to train the model after preprocessing
EXPECTED_FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges"
]

# Load the trained model (the full pipeline)
model_filepath = "models/churn_model.pkl"
try:
    if not os.path.exists(model_filepath):
        st.error(f"Error: The model file '{model_filepath}' was not found. Please run train_model.py first to create the model.")
        st.stop()
    model = joblib.load(model_filepath)
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.write("Fill the details below to predict whether a customer will churn.")

# --- User Inputs ---
with st.form("churn_prediction_form"):
    st.subheader("Customer Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

    st.subheader("Service Details")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.subheader("Billing Information")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)

    submitted = st.form_submit_button("🔮 Predict Churn")

if submitted:
    # Convert 'Senior Citizen' to the numerical format the model expects
    senior_citizen_val = 1 if senior == 'Yes' else 0

    # Create a DataFrame from user inputs, ensuring all features are present
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior_citizen_val,  # Use the converted value
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }], columns=EXPECTED_FEATURES) # Use the predefined column order

    # Make prediction using the loaded pipeline
    prediction_label = model.predict(input_data)[0]
    prediction_probability = model.predict_proba(input_data)[0][1]

    if prediction_label == 'Yes':
        st.error(f"⚠️ Customer is likely to CHURN with probability {prediction_probability:.2f}")

    else:
        st.success(f"✅ Customer is likely to STAY with probability {1 - prediction_probability:.2f}")
        st.balloons()