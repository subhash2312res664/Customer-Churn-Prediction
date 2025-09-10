import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os
from data_preprocessing import load_data

def train_and_save_model(data_path="Telco_customer_churn.xlsx", model_path="models/churn_model.pkl"):
    """
    Trains a customer churn prediction model and saves the trained pipeline.

    The function performs the following steps:
    1. Loads and preprocesses the data using the load_data function.
    2. Separates features and the target variable.
    3. Defines preprocessor pipelines for numerical and categorical features.
    4. Combines the preprocessors into a single ColumnTransformer.
    5. Creates a full machine learning pipeline with the preprocessor and a classifier.
    6. Trains the pipeline on the full dataset.
    7. Saves the trained pipeline to a file.

    Args:
        data_path (str): Path to the raw data file.
        model_path (str): Path to save the trained model.
    """
    df = load_data(data_path)
    if df is None:
        return

    # Define features (X) and target (y)
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline with the classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(random_state=42))])

    print("Training the model...")
    pipeline.fit(X, y)
    print("Model training complete.")

    # Save the trained pipeline
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to '{model_path}'")

if __name__ == "__main__":
    train_and_save_model()
