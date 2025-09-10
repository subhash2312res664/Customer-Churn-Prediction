import joblib
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Define the path to the model file
model_filepath = "models/churn_model.pkl"

# Check if the model file exists before trying to load it
if not os.path.exists(model_filepath):
    print(f"Error: The model file '{model_filepath}' was not found.")
    print("Please run train_model.py first to create the model.")
    # Exit or handle the error gracefully
    model = None
else:
    # Load the trained model (which is a full pipeline)
    try:
        model = joblib.load(model_filepath)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        model = None


def predict_single(customer_data: dict):
    """
    Predict churn for a single customer using the trained pipeline.

    The pipeline handles all necessary data transformations automatically.

    Args:
        customer_data (dict): A dictionary with customer features.

    Returns:
        tuple: A tuple containing the prediction label ('Yes' or 'No')
               and the churn probability (float).
    """
    if model is None:
        return "Model not loaded", 0.0

    # Define all expected columns from the training dataset.
    # This list now contains only the features used for training.
    expected_columns = [
        'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
        'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies',
        'Contract', 'Paperless Billing', 'Payment Method',
        'Monthly Charges', 'Total Charges'
    ]

    # Create a DataFrame from the input data, ensuring the columns are in the correct order.
    df = pd.DataFrame([customer_data])
    df = df.reindex(columns=expected_columns, fill_value=None)

    # Use the pipeline to predict. It will preprocess the data automatically.
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # Map the prediction to a human-readable label
    pred_label = 'Yes' if prediction == 'Yes' else 'No'

    return pred_label, probability


if __name__ == "__main__":
    # Example input for testing
    sample_input = {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 12,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 70.35,
        "Total Charges": 845.5
    }

    pred_label, prob = predict_single(sample_input)
    print(f"Prediction: {'Churn' if pred_label == 'Yes' else 'Not Churn'}")
    print(f"Churn Probability: {prob:.2f}")
