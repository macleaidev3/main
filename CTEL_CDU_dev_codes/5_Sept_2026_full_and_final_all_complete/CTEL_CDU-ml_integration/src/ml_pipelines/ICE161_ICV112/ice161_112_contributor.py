
import logging
from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import  check_ip21_update, check_daily_lab_report_data_update, sigfig
from src.utils.unit_conversion import celsius_to_kelvin
import random

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class CRContributorICE161toICV112():
    
    def __init__(self, month: str, year: str, yesterday_date: str):
        """
        Initializes the CRContributorICE161toICV112 class.

        Args:
            yesterday_date (str): The date for which data needs to be fetched. dd/mm/yyyy
        """
        logger.debug("Initializing CRContributorICE161toICV1122 for Date: %s", yesterday_date)
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        
        self.yesterday_date = yesterday_date
        
        self.month = month
        self.year = int(year)
        self.data_to_be_updated = {}

        self.set_up()

    def set_up(self) -> None:
        # =========== CHECK FOR DAILY LAB REPORT UPDATE ============== 
        # check if daily lab report data is updated for yesterday date
        logger.debug("Checking daily lab report update status for %s", self.yesterday_date)
        is_updated, table = check_daily_lab_report_data_update(self.yesterday_date)
        
        if not is_updated:
            logger.warning("CRContributorICE161toICV112 Aborted: Daily lab report data for '%s' is not updated for %s.", table, self.yesterday_date)
            return
        
        logger.debug("Lab report update verified. Proceeding with data fetch.")

        logger.debug("Checking IP21 update status for %s", self.yesterday_date)
        is_updated, table = check_ip21_update(self.yesterday_date)
        if not is_updated:
            logger.warning("CRContributorICE161toICV112 Aborted: IP21 data for '%s' is not updated for %s.", table, self.yesterday_date)
            return

        logger.debug("IP21 data verified. Proceeding with data fetch.")
        
        

        #================================ GET THE REQUIRED CRUDE BLEND DATA =============================
        blend_property_mapping = {
            # name in crude blend table -> name in contributor table
            "Molecular weight(g/mol)": "Molecular Weight",
            "Thermal conductivity(W/m·K)": "Thermal Conductivity",
            "DENSITY(g/mL)": "DENSITY",
            "Specific heat(J/kg·K)": "Cp",
            "Viscosity(Pa·s)": "Viscosity",
           "API": "API",
           "SULPHUR%": "Sulphur",
           "VR%": "VR%"
        }

        for source_column, destination_column in blend_property_mapping.items():
            try:
                value = self.db_manager.get_cell_value(
                    self.db_name,
                    f"blend_properties_{self.year}_{self.month}",
                    source_column,
                    "Date",
                    self.yesterday_date,
                )

                if value is None:
                    logger.debug(
                        "Fetched None for property '%s' on %s.",
                        source_column,
                        self.yesterday_date,
                    )
                    self.data_to_be_updated[destination_column] = None
                else:
                    self.data_to_be_updated[destination_column] = float(value)

            except Exception:
                logger.exception(
                    "Database query failed while fetching property '%s'. "
                    "Setting '%s' to None.",
                    source_column,
                    destination_column,
                )
                self.data_to_be_updated[destination_column] = None
    
        
    

        print(f"Data to be updated from ice161_icv112 contributor thread: {self.data_to_be_updated}")
        table_name = f"{self.year}_{self.month}_161_to_112_contributor"
        self.db_manager.update_a_row(self.db_name, table_name, "Date", self.yesterday_date, self.data_to_be_updated)

        data = self.db_manager.read_table(self.db_name, table_name)
        for row in data:
            print(row)

        logger.info("CRContributorICE161toICV112 completed for Date: %s", self.yesterday_date)


