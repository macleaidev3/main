"""
Total Temperature [K] Model Training Script (section 112to162, LOSO)
==================================================================
- Input features: X [ m ], Y [ m ], Z [ m ], DENSITY, API, Sulphur, VR%, Cp, Viscosity, Molecular Weight, Thermal Conductivity
- Target: Total Temperature [ K ]
- Reads all .xlsx files in data/calculated/112to162
- LOSO (Leave-One-Simulation-Out) cross-validation
- Saves per-simulation metrics, predictions, and final model
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Config ---
data_dir = r"C:\Users\intel1\Desktop\copy_cdu\data\calculated\102to161"
features = [
    'X [ m ]', ' Y [ m ]', ' Z [ m ]', 'DENSITY', 'API', 'Sulphur', 'VR%',
    'Cp', 'Viscosity', 'Molecular Weight', 'Thermal Conductivity'
]
target = ' Total Temperature [ K ]'

# --- Load all data ---
files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
dfs = []
for f in files:
    df = pd.read_excel(os.path.join(data_dir, f), engine='openpyxl')
    df['__sim__'] = f
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=features + [target])

# --- LOSO Cross-validation ---
results = []
predictions = []
for sim in data['__sim__'].unique():
    train_df = data[data['__sim__'] != sim]
    test_df = data[data['__sim__'] == sim]
    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'simulation': sim, 'rmse': rmse, 'mae': mae, 'r2': r2, 'n_samples': len(y_test)})
    pred_df = test_df.copy()
    pred_df['predicted_total_temperature'] = y_pred
    pred_df['error'] = y_pred - y_test
    pred_df['error_pct'] = np.abs(pred_df['error'] / (np.abs(y_test) + 1e-6)) * 100
    predictions.append(pred_df)
    print(f"LOSO: {sim} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f} | n={len(y_test)}")

# --- Save LOSO results ---
results_df = pd.DataFrame(results)
results_df.to_csv('loso_total_temperature_results.csv', index=False)

# Save simulation-wise prediction reports
os.makedirs('loso_reports', exist_ok=True)
for pred_df in predictions:
    sim = pred_df['__sim__'].iloc[0].replace('.xlsx', '')
    out_path = os.path.join('loso_reports', f'report_{sim}.csv')
    pred_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

print("\nLOSO results saved to loso_total_temperature_results.csv and loso_reports/")

# --- Train final model on all data ---
X = data[features].values
y = data[target].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
final_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
final_model.fit(X_scaled, y)
artifacts = {
    'model': final_model,
    'scaler': scaler,
    'features': features,
    'target': target
}
joblib.dump(artifacts, 'total_temperature_rf_model.joblib')
print("Final model, scaler, and features saved to total_temperature_rf_model.joblib")
