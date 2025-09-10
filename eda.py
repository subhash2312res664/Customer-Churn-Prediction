import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def perform_eda(file_path="Telco_customer_churn.xlsx"):
    """
    Performs Exploratory Data Analysis (EDA) on the Telco customer churn dataset.

    This function loads the data, displays summary statistics, checks for missing
    values and duplicates, and generates visualizations to explore the data,
    saving each plot as a PNG file.

    Args:
        file_path (str): The path to the Excel file containing the dataset.
    """
    print("--- Loading Data ---")
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # Drop unnecessary columns that are not useful for EDA or modeling
    columns_to_drop = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Latitude", "Longitude", "Lat Long"
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    print("\n--- Data Overview ---")
    print(df.info())

    print("\n--- Summary Statistics for Numerical Features ---")
    print(df.describe())

    print("\n--- Value Counts for Categorical Features ---")
    for column in df.select_dtypes(include='object').columns:
        print(f"\nValue counts for '{column}':")
        print(df[column].value_counts())

    print("\n--- Handling 'Total Charges' Column ---")
    # Convert "Total Charges" to numeric, coercing errors to NaN
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    print("\n--- Checking for Missing Values ---")
    print(df.isnull().sum())

    # Check for duplicates
    print("\n--- Checking for Duplicate Rows ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Renaming target column for clarity
    if "Churn Label" in df.columns:
        df.rename(columns={"Churn Label": "Churn"}, inplace=True)

    print("\n--- Generating and Saving Visualizations ---")

    # Create a directory to save the plots
    output_dir = "eda_plots"
    Path(output_dir).mkdir(exist_ok=True)

    # Set a color palette for consistency
    sns.set_palette("viridis")

    # 1. Churn Distribution
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.xlabel('Churn Status')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'churn_distribution.png')}")

    # 2. Churn rate by Internet Service
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Internet Service', hue='Churn', data=df)
    plt.title('Churn by Internet Service Type')
    plt.xlabel('Internet Service')
    plt.ylabel('Count')
    plt.legend(title='Churn')
    plt.savefig(os.path.join(output_dir, 'churn_by_internet_service.png'))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'churn_by_internet_service.png')}")

    # 3. Distribution of Tenure
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Tenure Months'], bins=30, kde=True)
    plt.title('Distribution of Customer Tenure')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'tenure_distribution.png'))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'tenure_distribution.png')}")

    # 4. Monthly Charges vs Total Charges
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Monthly Charges', y='Total Charges', hue='Churn', data=df)
    plt.title('Monthly Charges vs. Total Charges by Churn Status')
    plt.xlabel('Monthly Charges ($)')
    plt.ylabel('Total Charges ($)')
    plt.legend(title='Churn')
    plt.savefig(os.path.join(output_dir, 'monthly_vs_total_charges.png'))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'monthly_vs_total_charges.png')}")


if __name__ == "__main__":
    # Ensure matplotlib and seaborn are installed: pip install matplotlib seaborn
    perform_eda()
