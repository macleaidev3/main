import logging
from datetime import datetime
from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import sigfig

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class UTThicknessPrediction():
    
    def __init__(self, month: str, year: str, yesterday_date: str, probe_id: str, parent = None):
        logger.debug("Initializing UTThicknessPrediction for Probe ID: %s | Date: %s", probe_id, yesterday_date)
        
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        self.year = year
        self.month = month
        self.probe_id = probe_id
        self.model = None

        self.yesterday_date = yesterday_date
        self.is_calculation_done = False
        self.data_to_be_updated = {}

    def set_up(self) -> None:
        logger.info("Starting UTThicknessPrediction setup for Probe ID: %s", self.probe_id)

        # get the required parameters value for the prediction
        property_required = [ "Density(g/ml)", "API", "Sulphur%"]
        self.required_properties_dict = {}

        # get (yesterday) crude blend properties
        logger.debug("Fetching input properties for prediction from contributor table.")
        for prop in property_required:
            try:
                value = self.db_manager.get_cell_value(self.db_name,f"ut_{self.probe_id}_{self.year}_{self.month}_contributor", 
                                                    prop, "Date", self.yesterday_date)
                if value == None:
                    # if value is None, set it to None
                    self.required_properties_dict[prop] = None
                else:
                    self.required_properties_dict[prop] = float(value)
            except Exception as e:
                logger.error("Error fetching property '%s' from DB: %s", prop, str(e))
                self.required_properties_dict[prop] = None

        logger.debug("Input properties retrieved: %s", self.required_properties_dict)
        self.predicted_thickness = None
        
        try:
            density = self.required_properties_dict["Density(g/ml)"]
            api = self.required_properties_dict["API"]
            sulphur = self.required_properties_dict["Sulphur%"]

            # Checking for None values before casting
            if any(v is None for v in [density, api, sulphur]):
                logger.warning("Missing input parameters for prediction. Skipping calculation.")
            else:
                density = float(density)
                api = float(api)
                sulphur = float(sulphur)

                input_instance = (
                    f"{self.yesterday_date} 00:00:00",
                    density,
                    api,
                    sulphur
                )
                
                logger.debug("Running ML prediction model with inputs: %s", input_instance)
                self.predicted_thickness = self.model.predict_single_instance(input_instance)
                
                if self.predicted_thickness is not None:
                    self.is_calculation_done = True
                    logger.info("ML prediction successful: Thickness = %s", self.predicted_thickness)
        
        except Exception:
            logger.exception("An error occurred during the ML prediction calculation.")

        # record the calculation status
        if self.is_calculation_done:
            self.data_to_be_updated["Status"] = "Completed"
        else:
            self.data_to_be_updated["Status"] = "Pending"

        # record the time at which calculation was done
        current_date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.data_to_be_updated["Check on"] = current_date_time

        # record the predicted thickness
        if self.predicted_thickness is None:
            self.data_to_be_updated["Thickness(mm)"] = None
        else:
            self.data_to_be_updated["Thickness(mm)"] = sigfig(float(self.predicted_thickness))

        table_name = f"ut_{self.probe_id}_{self.year}_{self.month}_thickness"
        logger.debug("Updating database table '%s' with results: %s", table_name, self.data_to_be_updated)
        
        try:
            self.db_manager.update_a_row(self.db_name, table_name, "Date", self.yesterday_date, self.data_to_be_updated)
            logger.info("Successfully updated database prediction for %s", self.yesterday_date)
        except Exception:
            logger.exception("Failed to update UT thickness prediction in database.")