import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def load_data(path="Telco_customer_churn.xlsx"):
    """
    Load and preprocess the Telco Churn dataset.

    This function performs the following steps:
    1. Loads the data from an Excel file.
    2. Drops the 'CustomerID' and other unnecessary columns like location data and scores.
    3. Strips any leading/trailing whitespace from string columns.
    4. Converts the 'Total Charges' column to a numeric type, handling errors.
    5. Fills missing values for object columns with the mode and numeric columns with the median.

    Args:
        path (str): The file path to the Telco customer churn dataset.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    try:
        data = pd.read_excel(path)
        df = data.copy()
    except FileNotFoundError:
        print(f"Error: The file at '{path}' was not found. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

    # Define a list of columns to drop
    columns_to_drop = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Latitude", "Longitude", "Lat Long", "Churn Label",
        "Churn Reason", "Churn Score", "CLTV"
    ]

    # Drop the unnecessary columns if they exist
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Strip whitespace from string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert "Total Charges" to numeric, coercing errors
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            # For categorical data, fill with the most frequent value
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                # Fallback in case mode is empty (rare)
                df[col].fillna("Unknown", inplace=True)
        else:
            # For numerical data, fill with the median
            median_val = df[col].median()
            if not pd.isna(median_val):
                df[col].fillna(median_val, inplace=True)

    return df

# print(load_data())
