import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import logging
import warnings  # <--- Added to handle clutter
from src.utils.core_utility_functions import resource_path
import os

# ==========================================
# CLEAN LOGGING: Suppress harmless version warnings
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
# ==========================================

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

# ----------------------------
# MODEL ARCHITECTURE
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 16, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ----------------------------
# ENCAPSULATED PREDICTION CLASS
# ----------------------------
class ID00001Model:
    def __init__(
        self, 
        artifact_dir=resource_path("ml_module/cache_mlmodule_1232232/ID_00001"),
        historical_path=resource_path("ml_module/cache_mlmodule_1232232/ID_00001/test_original.csv")
    ):
        logger.info("Initializing ML Model ID_00001.")
        self.artifact_dir = artifact_dir
        self.historical_path = historical_path
        self.seq_len = 10
        
        self.features = [
            "Density", "API", "Sulphur", "thickness_diff", 
            "rolling_mean", "time_diff_hours", "corrosion_rate"
        ]

        # 1. Setup Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Model device set to: %s", self.device)

        # 2. Load Preprocessors
        try:
            #self.pt_X = joblib.load(f"{self.artifact_dir}/pt_X.pkl") ##Sornarjit

            print("==============================================")
            print("ARTIFACT DIRECTORY:")
            print(self.artifact_dir)

            print("PT_X PATH:")
            print(os.path.abspath(f"{self.artifact_dir}/pt_X.pkl"))

            print("PT_X EXISTS:")
            print(os.path.exists(f"{self.artifact_dir}/pt_X.pkl"))

            print("==============================================")

            self.pt_X = joblib.load(f"{self.artifact_dir}/pt_X.pkl")

            self.pt_y = joblib.load(f"{self.artifact_dir}/pt_y.pkl") ##Sornarjit
            self.feature_scaler = joblib.load(f"{self.artifact_dir}/feature_scaler.pkl")
            self.target_scaler = joblib.load(f"{self.artifact_dir}/target_scaler.pkl")
            
            # 3. Initialize and Load Model
            self.model = LSTMModel(input_size=len(self.features)).to(self.device)
            self.model.load_state_dict(torch.load(f"{self.artifact_dir}/lstm_model.pth", map_location=self.device))
            self.model.eval()
            logger.debug("Model components loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load model artifacts: %s", str(e))
            raise

        # 4. Pre-load and process historical data ONCE into state memory
        self._initialize_historical_data()

    def _initialize_historical_data(self):
        logger.debug("Initializing historical data baseline.")
        hist = pd.read_csv(self.historical_path)

        hist.columns = hist.columns.str.strip()
        # Suppressing date warnings by providing an explicit format if known, 
        # but filterwarnings already handles the messy logs
        hist['Sent Time'] = pd.to_datetime(hist['Sent Time'], dayfirst=True)
        hist = hist.sort_values("Sent Time").reset_index(drop=True)

        hist["thickness"] = hist["UT measurement (mm)"]
        hist["time_diff_hours"] = hist["Sent Time"].diff().dt.total_seconds() / 3600
        hist["time_diff_hours"] = hist["time_diff_hours"].fillna(12)

        hist["thickness_smooth"] = hist["thickness"].rolling(3, min_periods=1).mean()
        hist["thickness_diff"] = hist["thickness_smooth"].diff().fillna(0)
        hist["corrosion_rate"] = hist["thickness_diff"] / hist["time_diff_hours"]
        hist["rolling_mean"] = hist["thickness_smooth"].rolling(5, min_periods=1).mean()

        hist = hist.dropna(subset=["Density", "API", "Sulphur"]).reset_index(drop=True)
        self.current_data = hist.copy()
        logger.debug("Historical baseline initialized with %d rows.", len(self.current_data))


    def predict_single_instance(self, input_tuple):
        try:
            sent_time, density, api, sulphur = input_tuple
            
            new_row = pd.DataFrame([{
                'Sent Time': pd.to_datetime(sent_time, dayfirst=True),
                'Density': density,
                'API': api,
                'Sulphur': sulphur,
                'thickness': np.nan 
            }])

            temp_data = pd.concat([self.current_data, new_row], ignore_index=True)
            
            # Recompute rolling features
            temp_data["time_diff_hours"] = temp_data["Sent Time"].diff().dt.total_seconds() / 3600
            temp_data["time_diff_hours"] = temp_data["time_diff_hours"].fillna(12)
            temp_data["thickness"] = temp_data["thickness"].ffill()
            temp_data["thickness_smooth"] = temp_data["thickness"].rolling(3, min_periods=1).mean()
            temp_data["thickness_diff"] = temp_data["thickness_smooth"].diff().fillna(0)
            temp_data["corrosion_rate"] = temp_data["thickness_diff"] / temp_data["time_diff_hours"]
            temp_data["rolling_mean"] = temp_data["thickness_smooth"].rolling(5, min_periods=1).mean()

            for col in self.features:
                temp_data[col] = temp_data[col].ffill().fillna(0)

            # Sequence Length
            if len(temp_data) < self.seq_len:
                X_input = temp_data[self.features].iloc[-1:].values
                X_input = np.repeat(X_input, self.seq_len, axis=0)
            else:
                X_input = temp_data[self.features].iloc[-self.seq_len:].values

            X_input = self.pt_X.transform(pd.DataFrame(X_input, columns=self.features))
            X_input = self.feature_scaler.transform(X_input)
            X_tensor = torch.tensor(X_input.reshape(1, self.seq_len, -1), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                delta = self.model(X_tensor).cpu().numpy()

            delta = self.pt_y.inverse_transform(self.target_scaler.inverse_transform(delta))[0][0]
            new_thickness = float(temp_data["thickness"].iloc[-2] + delta)

            temp_data.loc[len(temp_data) - 1, "thickness"] = new_thickness
            self.current_data = temp_data

            logger.debug("Inference successful. Predicted thickness: %.4f", new_thickness)
            return new_thickness

        except Exception as e:
            logger.error("Inference failed for input %s: %s", str(input_tuple), str(e))
            return None