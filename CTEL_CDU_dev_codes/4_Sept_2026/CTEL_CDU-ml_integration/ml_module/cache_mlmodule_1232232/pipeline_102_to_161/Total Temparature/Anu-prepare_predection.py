import os
import joblib
import pandas as pd

MODEL_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_102_to_161\Total Temparature\total_temperature_rf_model.joblib"

INPUT_DIR = r"D:\Anurag BPCL WORK\Abhay Code\copy_cdu\experiments\102to161\102to161"

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]
target_column = artifacts["target"]

processed = 0

for file in os.listdir(INPUT_DIR):

    if not file.endswith(".xlsx"):
        continue

    path = os.path.join(INPUT_DIR, file)

    df = pd.read_excel(path)

    missing = [c for c in features if c not in df.columns]

    if missing:
        print(f"Skipping {file}")
        print("Missing columns:", missing)
        continue

    X = df[features].values

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)

    # Overwrite target column
    df[target_column] = pred

    # Save back to the same Excel file
    df.to_excel(
        path,
        index=False
    )

    processed += 1

    print(f"Updated -> {path}")

print(f"\nPrediction completed. Files updated: {processed}")