
import pandas as pd
from time import perf_counter
import os
from solution import preprocess, load_model, predict

def test_pipeline():
    print("Testing pipeline...")
    
    # 1. Load Model
    print("Loading model...")
    try:
        model = load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # 2. Load Test Data
    print("Loading test data...")
    try:
        df = pd.read_csv("test.csv")
        print(f"Test data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"FAILED to load test data: {e}")
        return

    # 3. Preprocess
    print("Preprocessing...")
    try:
        df_processed = preprocess(df)
        print("Preprocessing completed.")
    except Exception as e:
        print(f"FAILED to preprocess: {e}")
        return

    # 4. Predict
    print("Predicting...")
    start_time = perf_counter()
    try:
        predictions = predict(df_processed, model)
        duration = perf_counter() - start_time
        print(f"Predictions completed in {duration:.3f}s")
    except Exception as e:
        print(f"FAILED to predict: {e}")
        return

    # 5. Validate Output
    print("\nValidating results...")
    if not isinstance(predictions, pd.DataFrame):
        print("ERROR: Predictions is not a DataFrame")
        return
    
    if list(predictions.columns) != ["User_ID", "Purchased_Coverage_Bundle"]:
        print(f"ERROR: Incorrect columns: {predictions.columns.tolist()}")
        return

    if len(predictions) != len(df):
        print(f"ERROR: Count mismatch. Expected {len(df)}, got {len(predictions)}")
        return

    print("Success! Output format is correct.")
    print(f"Class distribution in predictions:\n{predictions['Purchased_Coverage_Bundle'].value_counts().sort_index()}")
    print("\nSample predictions:")
    print(predictions.head())

    # 6. Check constraints
    model_size = os.path.getsize("model.pkl") / (1024 * 1024)
    print(f"\nModel Size: {model_size:.2f} MB")
    
    latency_penalty = max(0.5, 1 - duration / 10)
    size_penalty = max(0.5, 1 - model_size / 200)
    
    print(f"Latency Penalty Score: {latency_penalty:.4f}")
    print(f"Size Penalty Score: {size_penalty:.4f}")

if __name__ == "__main__":
    test_pipeline()
