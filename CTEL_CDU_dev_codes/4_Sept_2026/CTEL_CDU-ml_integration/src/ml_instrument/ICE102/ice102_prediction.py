from src.server_manager.operation_manager import DatabaseManager




from src.ml_instrument.ICV113.input_df_builder import PredictionInputBuilder
from src.utils.core_utility_functions import sigfig, format_date_long
class CRPredictionICE102():
    
    def __init__(self, month: str, year: str, yesterday_date: str):
        
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        self.year = year
        self.month = month
        self.table_name = f"{year}_102_cr"

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
            equipment="113",
            prediction_date=self.yesterday_date,
        )

        input_df = builder.build()

        predictors = [
            CrudePHPredictor(),
            DensityPredictor(),
            H2SPredictor(),
            TotalPressurePredictor(),
            TotalTemperaturePredictor(),
            WallShearPredictor(),
            TurbulentKineticPredictor(),
        ]

        for predictor in predictors:
            target, prediction = predictor.predict(input_df)
            input_df[target] = prediction

        corrosion_predictor = CorrosionRatePredictor()
        cr_target, cr_prediction = corrosion_predictor.predict(input_df)
        input_df[cr_target] = cr_prediction

        corrosion_values = input_df[cr_target].tolist()

            
        updated_corrosion_values = []
        column_name = format_date_long(self.yesterday_date)
        for corrosion_value in corrosion_values:
            try:
                updated_corrosion_values.append(sigfig(corrosion_value))
            except:
                updated_corrosion_values.append("Pending")

        _serial_number = input_df["S no"].tolist()

        

        for i in range(len(_serial_number)):
            cr_value = updated_corrosion_values[i]

            data_to_be_updated = {
                column_name: cr_value
            }

            self.db_manager.update_a_row(self.db_name, self.table_name, "S no", i + 1, data_to_be_updated)


        check_data = self.db_manager.read_columns(self.db_name, self.table_name, ["S no", column_name])

        print(f"Data for {self.yesterday_date}: {len(check_data)}")
        for d in check_data[:50]:
            print(f"S no: {d[0]}, {column_name}: {d[1]}")