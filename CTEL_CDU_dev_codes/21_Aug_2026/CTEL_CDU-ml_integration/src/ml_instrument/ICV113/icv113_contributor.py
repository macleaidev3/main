
import logging
from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import  check_ip21_update, check_daily_lab_report_data_update, sigfig
from src.utils.unit_conversion import celsius_to_kelvin


# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class CRContributorICV113():
    
    def __init__(self, month: str, year: str, yesterday_date: str):
        """
        Initializes the CRContributorICV113 class.

        Args:
            yesterday_date (str): The date for which data needs to be fetched. dd/mm/yyyy
        """
        logger.debug("Initializing CRContributorICV113 for Date: %s", yesterday_date)
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
            logger.warning("CRContributorICV113 Aborted: Daily lab report data for '%s' is not updated for %s.", table, self.yesterday_date)
            return
        
        logger.debug("Lab report update verified. Proceeding with data fetch.")

        logger.debug("Checking IP21 update status for %s", self.yesterday_date)
        is_updated, table = check_ip21_update(self.yesterday_date)
        if not is_updated:
            logger.warning("CRContributorICV113 Aborted: IP21 data for '%s' is not updated for %s.", table, self.yesterday_date)
            return

        logger.debug("IP21 data verified. Proceeding with data fetch.")
        
        #=============================== GET THE REQUIRED IP21 DATA ===============================
        # required_ip21_data = ["crude temperature(k)", "temperature overhead drum (k)"]
        # field_name_in_ip_table = ["Temp of Naphtha from IC-E-126 °C", "CDU col Top temp °C"]
        ip_data = self.db_manager.special_method_ip_21_get_rows_for_day(self.yesterday_date)
        temp_naptha = ip_data["Temp of Naphtha from IC-E-126 °C"]
        temp_cdu = ip_data["CDU col Top temp °C"]

        # print(f"Temp of Naphtha from IC-E-126 °C data from ip21 for {self.yesterday_date}: {temp_naptha}")
        # print(f"CDU col Top temp °C data from ip21 for {self.yesterday_date}: {temp_cdu}")

        average_naptha_temp = 0
        try:
            average_naptha_temp = sigfig(celsius_to_kelvin(sum(temp_naptha)/len(temp_naptha)))
        except:
            average_naptha_temp = None

        average_cdu_temp = 0
        try:
            average_cdu_temp = sigfig(celsius_to_kelvin(sum(temp_cdu)/len(temp_cdu)))
        except:
            average_cdu_temp = None

        self.data_to_be_updated["crude temperature(k)"] = average_naptha_temp
        self.data_to_be_updated["temperature overhead drum (k)"] = average_cdu_temp

        #================================ GET THE REQUIRED CRUDE BLEND DATA =============================
        blend_property_mapping = {
            "Molecular weight(g/mol)": "mw(g/gmol)",
            "Thermal conductivity(W/m·K)": "k (w/m-k)",
            "DENSITY(g/mL)": "density(kg/m3)",
            "Specific heat(J/kg·K)": "cp (j/kg-k)",
            "Viscosity(Pa·s)": "viscosity (pa-s)",
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
    
        
        # ================================ INTERNAL CALCULATIONS =================================
        #=========================================================================================

        #============================== GET FLOW RATE DATA ===============================
        # Flow rate max = 352 TPH * 1.2                                                   
        # Flow rate min = 352 TPH / 8 
        # And then unit conversion into kg/s for easier input in simulations.
        import random
        from src.utils.unit_conversion import tph_to_kg_per_s

        total_flowrate = random.uniform(352 / 8, 352 * 1.2)
        # convert to kg/s
        total_flowrate = sigfig(tph_to_kg_per_s(total_flowrate))

        # Flow rate crude inlet
        #Phase split ratio is not calculated is varying between 0.6 - 0.95
        phase_split_ratio = random.uniform(0.6, 0.95)
        crude_inlet_flowrate = sigfig(phase_split_ratio  * total_flowrate)

        # Flow rate ww inlet
        ww_inlet_flowrate = sigfig(total_flowrate - crude_inlet_flowrate)

        # Flow rate outlet 1
        flow_rate_out1_base = total_flowrate * 0.8
        flow_rate_out2_base = total_flowrate * 0.2
        #Split outlet base = Flow rate out1 base kg/s / (Flow rate out1 base kg/s + Flow rate out2 base kg/s) ;
        split_outlet_base = flow_rate_out1_base / (flow_rate_out1_base + flow_rate_out2_base)
        split_out_min = 0.5 * split_outlet_base 
        split_out_max = 1.2 * split_outlet_base
        outlet_split_ratio = random.uniform(split_out_min, split_out_max)
        flow_rate_out1 = sigfig(total_flowrate * outlet_split_ratio)

        # Flow rate outlet 2
        flow_rate_out2 = sigfig(total_flowrate - flow_rate_out1)

        self.data_to_be_updated["flow rate crude inlet(kg/s)"] = crude_inlet_flowrate
        self.data_to_be_updated["flow rate ww inlet(kg/s)"] = ww_inlet_flowrate
        self.data_to_be_updated["flow rate outlet 1(kg/s)"] = flow_rate_out1
        self.data_to_be_updated["flow rate outlet 2(kg/s)"] = flow_rate_out2

        #================================ GET MOLE FRACTIONS =============================
        # Sulphur mole fraction
        #  Mf sulfur min = 0.001 - mf sulfur max = 0.05
        mf_sulfur = sigfig(random.uniform(0.001, 0.05))

        # H mole fraction
        # Mf H+ min = 0.0005 -  H+ max = 0.05
        mf_h = sigfig(random.uniform(0.0005, 0.05))

        # Crude inlet mf
        # Calculation details : 1- (mf sulfur  + mf H+) 
        mf_crude_inlet = sigfig(1 - (mf_sulfur + mf_h))

        self.data_to_be_updated["sulfur mf"] = mf_sulfur
        self.data_to_be_updated["h+ mf"] = mf_h
        self.data_to_be_updated["crude inlet mf"] = mf_crude_inlet

        print(f"Data to be updated from icv113 contributor thread: {self.data_to_be_updated}")
        table_name = f"{self.year}_{self.month}_113_contributor"
        self.db_manager.update_a_row(self.db_name, table_name, "Date", self.yesterday_date, self.data_to_be_updated)

        # data = self.db_manager.read_table(self.db_name, table_name)
        # for row in data:
        #     print(row)

        logger.info("CRContributorICV113 completed for Date: %s", self.yesterday_date)


