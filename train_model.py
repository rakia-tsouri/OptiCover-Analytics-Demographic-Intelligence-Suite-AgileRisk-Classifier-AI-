"""
DataQuest Hackathon — Training Script
Multi-class classification: Predict Purchased_Coverage_Bundle (0-9)
Model: LightGBM with class imbalance handling
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG
# ============================================================
SEED = 42
N_FOLDS = 5
MODEL_PATH = "model.pkl"

# Columns to drop
DROP_COLS = ["User_ID", "Employer_ID"]

# Categorical columns to label-encode
CAT_COLS = [
    "Region_Code",
    "Broker_Agency_Type",
    "Deductible_Tier",
    "Acquisition_Channel",
    "Payment_Schedule",
    "Employment_Status",
    "Policy_Start_Month",
]

# Columns with missing values to fill
FILL_WITH_MINUS1 = ["Broker_ID"]
FILL_WITH_ZERO = ["Child_Dependents"]
FILL_WITH_UNKNOWN = ["Region_Code", "Deductible_Tier", "Acquisition_Channel"]

TARGET = "Purchased_Coverage_Bundle"


# ============================================================
# PREPROCESSING (must match solution.py exactly)
# ============================================================
def build_label_encoders(df):
    """Fit label encoders on training data."""
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def preprocess_data(df, encoders, is_train=True):
    """Apply all preprocessing steps."""
    df = df.copy()

    # Drop columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if is_train:
        # Keep target separate
        target = df[TARGET].copy() if TARGET in df.columns else None
    else:
        target = None

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

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

    # Label encode categoricals
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = encoders[col]
            if "Unknown" not in set(le.classes_):
                le.classes_ = np.append(le.classes_, "Unknown")
            label_map = {cls: idx for idx, cls in enumerate(le.classes_)}
            unknown_idx = label_map["Unknown"]
            df[col] = df[col].map(label_map).fillna(unknown_idx).astype(int)

    # Drop target column from features if present
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    return df, target


# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================
def compute_sample_weights(y):
    """Compute per-sample weights inversely proportional to class frequency."""
    class_counts = np.bincount(y, minlength=10)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    total = len(y)
    weights = total / (10 * class_counts)
    # Boost ultra-rare classes (8, 9) even more
    for cls in [8, 9]:
        weights[cls] *= 3.0
    for cls in [5, 6]:
        weights[cls] *= 1.5
    sample_weights = weights[y]
    return sample_weights


# ============================================================
# TRAINING
# ============================================================
def main():
    print("Loading training data...")
    df = pd.read_csv("train.csv")
    print(f"Train shape: {df.shape}")
    print(f"Class distribution:\n{df[TARGET].value_counts().sort_index()}")

    # Build label encoders
    print("\nBuilding label encoders...")
    encoders = build_label_encoders(df)

    # Preprocess
    print("Preprocessing...")
    X, y = preprocess_data(df, encoders, is_train=True)
    y = y.values

    print(f"Features: {X.columns.tolist()}")
    print(f"Feature matrix shape: {X.shape}")

    # LightGBM parameters optimized for multi-class + imbalance
    params = {
        "objective": "multiclass",
        "num_class": 10,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 5,  # Low — to learn rare classes
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1000,
        "is_unbalance": True,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    # Cross-validation
    print(f"\nRunning {N_FOLDS}-fold Stratified CV...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    best_model = None
    best_score = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Compute sample weights
        sw_train = compute_sample_weights(y_train)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="macro")
        fold_scores.append(score)

        print(f"  Fold {fold+1}: macro_f1 = {score:.5f} (best_iter={model.best_iteration_})")

        # Per-class breakdown
        per_class_f1 = f1_score(y_val, preds, average=None, labels=list(range(10)))
        for cls_id, cls_f1 in enumerate(per_class_f1):
            count = (y_val == cls_id).sum()
            print(f"    Class {cls_id}: F1={cls_f1:.4f} (n={count})")

        if score > best_score:
            best_score = score
            best_model = model

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print(f"\n{'='*50}")
    print(f"CV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")
    print(f"Best fold: {max(fold_scores):.5f}")
    print(f"{'='*50}")

    # Train final model on ALL data
    print("\nTraining final model on all data...")
    sw_all = compute_sample_weights(y)
    final_model = lgb.LGBMClassifier(**{**params, "n_estimators": best_model.best_iteration_})
    final_model.fit(X, y, sample_weight=sw_all)

    # Pre-calculate label mappings for faster inference
    label_maps = {}
    for col, le in encoders.items():
        if "Unknown" not in set(le.classes_):
            le.classes_ = np.append(le.classes_, "Unknown")
        label_maps[col] = {cls: idx for idx, cls in enumerate(le.classes_)}

    # Save model + maps together
    artifact = {
        "model": final_model,
        "label_maps": label_maps,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)

    import os
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    size_penalty = max(0.5, 1 - size_mb / 200)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Size penalty: {size_penalty:.4f}")
    print(f"Effective score (approx): {mean_f1 * size_penalty:.5f}")

    # Feature importance
    print("\nTop 15 features:")
    fi = pd.Series(
        final_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    for feat, imp in fi.head(15).items():
        print(f"  {feat}: {imp}")

    print("\nDone!")


if __name__ == "__main__":
    main()
