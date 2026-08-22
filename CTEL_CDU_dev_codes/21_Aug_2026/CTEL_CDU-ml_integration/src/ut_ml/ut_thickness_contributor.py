import logging
from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import check_daily_lab_report_data_update, check_ip21_update

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class UTThicknessContributor():
    
    def __init__(self, month: str, year: str, yesterday_date: str, probe_id: str,  parent = None):
        logger.debug("Initializing UTThicknessContributor for Probe ID: %s | Date: %s", probe_id, yesterday_date)
        
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        self.year = year
        self.month = month
        self.probe_id = probe_id

        self.yesterday_date = yesterday_date
        self.data_to_be_updated = {}

    def set_up(self) -> None:
        logger.info("Starting UTThicknessContributor setup sequence for Probe ID: %s", self.probe_id)

        # =========== CHECK FOR DAILY LAB REPORT UPDATE ============== 
        # check if daily lab report data is updated for yesterday date
        logger.debug("Checking daily lab report update status for %s", self.yesterday_date)
        is_updated, table = check_daily_lab_report_data_update(self.yesterday_date)
        
        if not is_updated:
            logger.warning("UTThicknessContributor Aborted: Daily lab report data for '%s' is not updated for %s.", table, self.yesterday_date)
            return
        
        logger.debug("Lab report update verified. Proceeding with data fetch.")

        logger.debug("Checking IP21 update status for %s", self.yesterday_date)
        is_updated, table = check_ip21_update(self.yesterday_date)
        if not is_updated:
            logger.warning("UTThicknessContributor Aborted: IP21 data for '%s' is not updated for %s.", table, self.yesterday_date)
            return

        logger.debug("IP21 data verified. Proceeding with data fetch.")

        # get (yesterday) crude property from crude blend table
        blend_properties_to_get = ["DENSITY(g/mL)", "API", "SULPHUR%"]
        self.value_blend_properties ={}
        
        # get (yesterday) crude blend properties
        for prop in blend_properties_to_get:
            try:
                value = self.db_manager.get_cell_value(self.db_name, f"blend_properties_{self.year}_{self.month}",
                                                    prop, "Date", self.yesterday_date)
                if value == None:
                    # if value is None, set it to None
                    logger.debug("Fetched None for property '%s' on %s.", prop, self.yesterday_date)
                    self.value_blend_properties[prop] = None
                else:
                    self.value_blend_properties[prop] = float(value)
            except Exception as e:
                # Log the exception along with the traceback instead of just printing it
                logger.exception("Database query failed while fetching property '%s'. Setting value to None.", prop)
                self.value_blend_properties[prop] = None


        # 1. Define how input keys map to output keys
        prop_mapping = {
            "DENSITY(g/mL)": "Density(g/mL)",
            "API": "API",
            "SULPHUR%": "Sulphur%"
        }

        logger.debug("Mapping blend properties to update dictionary.")
        # 2. Iterate and apply the logic
        for prop in blend_properties_to_get:
            target_key = prop_mapping.get(prop)
            
            # Only process if the property is in our mapping
            if target_key: 
                value = self.value_blend_properties.get(prop)
                
                # 3. Use an inline ternary operator for the float conversion
                self.data_to_be_updated[target_key] = float(value) if value is not None else None
        

        table_name = f"ut_{self.probe_id}_{self.year}_{self.month}_contributor"
        logger.info("Pushing mapped data to database table: %s", table_name)
        
        self.db_manager.update_a_row(self.db_name, table_name, "Date", self.yesterday_date, self.data_to_be_updated)
        
        logger.debug("UTThicknessContributor setup sequence completed successfully.")