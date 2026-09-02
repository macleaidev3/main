from src.server_manager.operation_manager import DatabaseManager
import numpy as np

from ml_module.pipeline_112_to_162.predict_temp import TotalTemperaturePredictor
from ml_module.pipeline_112_to_162.predict_pressure import TotalPressurePredictor
from ml_module.pipeline_112_to_162.predict_wall_shear import WallShearPredictor
from ml_module.pipeline_112_to_162.predict_cr import CorrosionRatePredictor


from src.ml_pipelines.ICEV112_ICE162.input_df_builder import PredictionInputBuilder
from src.utils.core_utility_functions import sigfig, format_date_long
class CRPredictionICV112TO162():
    
    def __init__(self, month: str, year: str, yesterday_date: str):
        
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        self.year = year
        self.month = month
        self.table_name = f"{year}_112_to_162_cr"

        self.yesterday_date = yesterday_date
        self.is_required_calculation = True
        self.data_to_be_updated = {}

        self.set_up()

    def set_up(self) -> None:
    
            builder = PredictionInputBuilder(
                db_manager=self.db_manager,
                db_name=self.db_name,
                year=self.year,
                month=self.month,
                equipment="112_to_162",
                prediction_date=self.yesterday_date,
            )
    
            input_df = builder.build()
    
            _serial_number = input_df["S no"].tolist()
            column_name = format_date_long(self.yesterday_date)
            updated_corrosion_values = []
    
            # If any required input is missing, mark all predictions as Pending.
            if input_df.isna().any().any():
    
                updated_corrosion_values = ["Pending"] * len(_serial_number)
    
            else:
                total_temperature_predictor = TotalTemperaturePredictor()
                target_total_temperature_column, pred_total_temperature = total_temperature_predictor.predict(input_df)
                input_df[target_total_temperature_column] = pred_total_temperature
               
               
    
                total_pressure_predictor = TotalPressurePredictor()
                target_total_pressure_column, pred_total_pressure = total_pressure_predictor.predict(input_df)
                input_df[target_total_pressure_column] = pred_total_pressure
    
                
                wall_shear_predictor = WallShearPredictor()
                target_wall_shear_column, pred_wall_shear = wall_shear_predictor.predict(input_df)
                input_df[target_wall_shear_column] = pred_wall_shear
                 
                corrosion_rate_predictor = CorrosionRatePredictor()
                target_corrosion_rate_column, pred_corrosion_rate = corrosion_rate_predictor.predict(input_df)
                input_df[target_corrosion_rate_column] = pred_corrosion_rate
                corrosion_values = input_df[target_corrosion_rate_column].tolist()
    
                for corrosion_value in corrosion_values:
                    try:
                        updated_corrosion_values.append(sigfig(corrosion_value))
                    except Exception:
                        updated_corrosion_values.append("Pending")
    
            for serial_no, cr_value in zip(_serial_number, updated_corrosion_values):
    
                data_to_be_updated = {
                    column_name: cr_value
                }
    
                self.db_manager.update_a_row(
                    self.db_name,
                    self.table_name,
                    "S no",
                    serial_no,
                    data_to_be_updated,
                )
    
            check_data = self.db_manager.read_columns(
                self.db_name,
                self.table_name,
                ["S no", column_name],
            )
    
            print(f"Data for {self.yesterday_date}: {len(check_data)}")
            for d in check_data[:50]:
                print(f"S no: {d[0]}, {column_name}: {d[1]}")
    