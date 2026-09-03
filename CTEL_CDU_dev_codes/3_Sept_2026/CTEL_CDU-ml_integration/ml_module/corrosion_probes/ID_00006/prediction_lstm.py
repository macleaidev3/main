import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import logging
import warnings

from src.utils.core_utility_functions import resource_path

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
class ID00006Model:
    def __init__(
    self,
    artifact_dir=resource_path("ml_module/cache_mlmodule_1232232/ID_00006"),
    historical_path=resource_path("ml_module/cache_mlmodule_1232232/ID_00006/test_original.csv")
):
        """
        Initializes the model, loads scalers, and pre-loads the historical data into memory.
        """

        logger.info("Initializing ID00006 corrosion probe prediction model.")

        self.artifact_dir = artifact_dir
        self.historical_path = historical_path
        self.seq_len = 10

        self.features = [
            "Density",
            "API",
            "Sulphur",
            "thickness_diff",
            "rolling_mean",
            "time_diff_hours",
            "corrosion_rate",
        ]

        try:
            # ---------------------------------------------------------
            # Setup Device
            # ---------------------------------------------------------
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            logger.info(f"Inference device selected: {self.device}")

            # ---------------------------------------------------------
            # Load preprocessing artifacts
            # ---------------------------------------------------------
            logger.debug("Loading preprocessing artifacts.")

            self.pt_X = joblib.load(f"{self.artifact_dir}/pt_X.pkl")
            self.pt_y = joblib.load(f"{self.artifact_dir}/pt_y.pkl")
            self.feature_scaler = joblib.load(f"{self.artifact_dir}/feature_scaler.pkl")
            self.target_scaler = joblib.load(f"{self.artifact_dir}/target_scaler.pkl")

            logger.info("Preprocessing artifacts loaded successfully.")

            # ---------------------------------------------------------
            # Load trained model
            # ---------------------------------------------------------
            logger.debug("Loading trained LSTM model.")

            self.model = LSTMModel(input_size=len(self.features)).to(self.device)

            self.model.load_state_dict(
                torch.load(
                    f"{self.artifact_dir}/lstm_model.pth",
                    map_location=self.device,
                )
            )

            self.model.eval()

            logger.info("LSTM model loaded successfully.")

            # ---------------------------------------------------------
            # Load historical baseline
            # ---------------------------------------------------------
            self._initialize_historical_data()

            logger.info(
                "ID00006 corrosion probe prediction model initialized successfully."
            )

        except Exception:
            logger.exception(
                "Failed to initialize ID00006 corrosion probe prediction model."
            )
            raise
    def _initialize_historical_data(self):
        """Loads and prepares the initial historical baseline."""

        logger.debug(
            f"Loading historical baseline from '{self.historical_path}'."
        )

        hist = pd.read_csv(self.historical_path)

        hist.columns = hist.columns.str.strip()
        hist["Sent Time"] = pd.to_datetime(hist["Sent Time"], dayfirst=True)
        hist = hist.sort_values("Sent Time").reset_index(drop=True)

        hist["thickness"] = hist["UT measurement (mm)"]
        hist["time_diff_hours"] = hist["Sent Time"].diff().dt.total_seconds() / 3600
        hist["time_diff_hours"] = hist["time_diff_hours"].fillna(12)

        hist["thickness_smooth"] = (
            hist["thickness"]
            .rolling(3, min_periods=1)
            .mean()
        )

        hist["thickness_diff"] = (
            hist["thickness_smooth"]
            .diff()
            .fillna(0)
        )

        hist["corrosion_rate"] = (
            hist["thickness_diff"]
            / hist["time_diff_hours"]
        )

        hist["rolling_mean"] = (
            hist["thickness_smooth"]
            .rolling(5, min_periods=1)
            .mean()
        )

        hist = hist.dropna(
            subset=["Density", "API", "Sulphur"]
        ).reset_index(drop=True)

        self.current_data = hist.copy()

        logger.info(
            f"Historical baseline initialized successfully with "
            f"{len(self.current_data)} records."
        )


    def predict_single_instance(self, input_tuple):
        """
        Predicts the thickness for a single new data point.
        Returns the float prediction if successful, or None if an error occurs.
        """

        logger.debug(
            "Prediction request received | "
            f"Sent Time: {input_tuple[0]}, "
            f"Density: {input_tuple[1]}, "
            f"API: {input_tuple[2]}, "
            f"Sulphur: {input_tuple[3]}"
        )

        try:
            # 1. Unpack the tuple and format it
            sent_time, density, api, sulphur = input_tuple

            new_row = pd.DataFrame([{
                "Sent Time": pd.to_datetime(sent_time, dayfirst=True),
                "Density": density,
                "API": api,
                "Sulphur": sulphur,
                "thickness": np.nan
            }])

            # 2. Append to a TEMPORARY dataframe
            temp_data = pd.concat(
                [self.current_data, new_row],
                ignore_index=True
            )

            # 3. Recompute rolling features
            temp_data["time_diff_hours"] = (
                temp_data["Sent Time"].diff().dt.total_seconds() / 3600
            )
            temp_data["time_diff_hours"] = temp_data["time_diff_hours"].fillna(12)

            temp_data["thickness"] = temp_data["thickness"].ffill()

            temp_data["thickness_smooth"] = (
                temp_data["thickness"]
                .rolling(3, min_periods=1)
                .mean()
            )

            temp_data["thickness_diff"] = (
                temp_data["thickness_smooth"]
                .diff()
                .fillna(0)
            )

            temp_data["corrosion_rate"] = (
                temp_data["thickness_diff"]
                / temp_data["time_diff_hours"]
            )

            temp_data["rolling_mean"] = (
                temp_data["thickness_smooth"]
                .rolling(5, min_periods=1)
                .mean()
            )

            # Fill NaNs
            for col in self.features:
                temp_data[col] = (
                    temp_data[col]
                    .ffill()
                    .fillna(0)
                )

            # 4. Ensure Sequence Length
            if len(temp_data) < self.seq_len:

                logger.debug(
                    "Historical sequence shorter than required "
                    f"({len(temp_data)} < {self.seq_len}). "
                    "Repeating latest sample."
                )

                X_input = temp_data[self.features].iloc[-1:].values
                X_input = np.repeat(X_input, self.seq_len, axis=0)

            else:
                X_input = temp_data[self.features].iloc[-self.seq_len:].values

            # 5. Transform and Predict
            X_input = self.pt_X.transform(
                pd.DataFrame(X_input, columns=self.features)
            )

            X_input = self.feature_scaler.transform(X_input)

            X_tensor = torch.tensor(
                X_input.reshape(1, self.seq_len, -1),
                dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                delta = self.model(X_tensor).cpu().numpy()

            delta = self.pt_y.inverse_transform(
                self.target_scaler.inverse_transform(delta)
            )[0][0]

            # Calculate new thickness
            last_thickness = temp_data["thickness"].iloc[-2]
            new_thickness = float(last_thickness + delta)

            # Update state
            temp_data.loc[len(temp_data) - 1, "thickness"] = new_thickness
            self.current_data = temp_data

            logger.info(
                "Prediction completed successfully | "
                f"Sent Time: {sent_time} | "
                f"Predicted Thickness: {new_thickness:.6f} mm"
            )

            return new_thickness

        except Exception:
            logger.exception(
                "Prediction failed | "
                f"Sent Time: {input_tuple[0]}, "
                f"Density: {input_tuple[1]}, "
                f"API: {input_tuple[2]}, "
                f"Sulphur: {input_tuple[3]}"
            )
            return None
        
        