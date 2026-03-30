# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation will be computed for the predict function only.
#
# Good luck!
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib

# ============================================================
# Constants — must match train_model.py exactly
# ============================================================
DROP_COLS = ["User_ID", "Employer_ID"]

CAT_COLS = [
    "Region_Code",
    "Broker_Agency_Type",
    "Deductible_Tier",
    "Acquisition_Channel",
    "Payment_Schedule",
    "Employment_Status",
    "Policy_Start_Month",
]

FILL_WITH_MINUS1 = ["Broker_ID"]
FILL_WITH_ZERO = ["Child_Dependents"]
FILL_WITH_UNKNOWN = ["Region_Code", "Deductible_Tier", "Acquisition_Channel"]

TARGET = "Purchased_Coverage_Bundle"


def preprocess(df):
    # Implement any preprocessing steps required for your model here.
    # Return a Pandas DataFrame of the data
    #
    # Note: Don't drop the 'User_ID' column here.
    # It will be used in the predict function to return the final predictions.

    df = df.copy()

    # Drop Employer_ID (94% missing, not useful)
    if "Employer_ID" in df.columns:
        df = df.drop(columns=["Employer_ID"])

    # Fill missing values
    for col in FILL_WITH_MINUS1:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    for col in FILL_WITH_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in FILL_WITH_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Feature engineering
    df["Total_Dependents"] = (
        df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]
    )
    df["Has_Children"] = ((df["Child_Dependents"] + df["Infant_Dependents"]) > 0).astype(int)
    df["Income_Per_Dependent"] = df["Estimated_Annual_Income"] / (df["Total_Dependents"] + 1)
    df["Claim_Ratio"] = df["Previous_Claims_Filed"] / (
        df["Previous_Policy_Duration_Months"] + 1
    )
    df["Is_New_Customer"] = (1 - df["Existing_Policyholder"]).astype(int)
    df["Quote_to_UW_Ratio"] = df["Days_Since_Quote"] / (
        df["Underwriting_Processing_Days"] + 1
    )
    df["Total_Vehicles_Riders"] = df["Vehicles_on_Policy"] + df["Custom_Riders_Requested"]

    return df


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------

    artifact = joblib.load("model.pkl")
    model = artifact  # Contains both 'model' and 'encoders'

    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    predictions = None
    # ------------------ PREDICTION LOGIC ------------------

    # Extract components
    lgb_model = model["model"]

    # Save User_ID before dropping
    user_ids = df["User_ID"].copy()

    # Drop User_ID for prediction
    X = df.drop(columns=["User_ID"], errors="ignore")

    # Drop target if present (shouldn't be in test, but just in case)
    if TARGET in X.columns:
        X = X.drop(columns=[TARGET])

    # Label encode categoricals (must match training) — optimized for speed using pre-calculated maps
    label_maps = model["label_maps"]
    for col in CAT_COLS:
        if col in X.columns:
            l_map = label_maps[col]
            unknown_idx = l_map.get("Unknown", 0)
            X[col] = X[col].astype(str).map(l_map).fillna(unknown_idx).astype(int)

    # Predict
    preds = lgb_model.predict(X)

    predictions = pd.DataFrame({
        "User_ID": user_ids,
        "Purchased_Coverage_Bundle": preds.astype(int),
    })

    # ------------------ END PREDICTION LOGIC ------------------
    return predictions


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------


def run(df) -> tuple[float, float, float]:
    from time import time

    # Load the processed data:
    df_processed = preprocess(df)

    # Load the model:
    model = load_model()
    size = get_model_size(model)

    # Get the predictions and time taken:
    start = time.perf_counter()
    predictions = predict(
        df_processed, model
    )  # NOTE: Don't call the `preprocess` function here.

    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
