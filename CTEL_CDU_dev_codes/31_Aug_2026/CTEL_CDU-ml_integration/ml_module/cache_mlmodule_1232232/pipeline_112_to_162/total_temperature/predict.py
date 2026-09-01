"""
Total Temperature [K] Batch Prediction Script (section 112to162)
==============================================================
- Loads trained model/scaler/features from total_temperature_rf_model.joblib
- Reads all .xlsx files in data/calculated/112to162
- Predicts Total Temperature [ K ] for each row
- Saves new CSVs with predicted_total_temperature column in prediction_results/
"""

import os
import numpy as np
import pandas as pd
import joblib

# --- Config ---
data_dir = r"C:\Users\intel1\Desktop\copy_cdu\data\calculated\112to162"
model_path = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_112_to_162\total_temperature\total_temperature_rf_model.joblib"
output_dir = r"C:\Users\intel1\Desktop\copy_cdu\experiments\112to162\total_temperature\prediction_results"
os.makedirs(output_dir, exist_ok=True)

# --- Load model artifacts ---
artifacts = joblib.load(model_path)
model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['features']

# --- Predict for each file ---
files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
for f in files:
    df = pd.read_excel(os.path.join(data_dir, f), engine='openpyxl')
    if not all(col in df.columns for col in features):
        print(f"Skipping {f}: missing required columns")
        continue
    X = df[features].values
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    df['predicted_total_temperature'] = pred
    out_path = os.path.join(output_dir, f.replace('.xlsx', '_predicted.csv'))
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

print("\nAll predictions complete.")
