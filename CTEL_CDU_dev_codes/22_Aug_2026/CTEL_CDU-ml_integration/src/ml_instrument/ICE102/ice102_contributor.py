import random
from src.utils.unit_conversion import tph_to_kg_per_s
import logging
from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import  check_ip21_update, check_daily_lab_report_data_update, sigfig
from src.utils.unit_conversion import celsius_to_kelvin


# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class CRContributorICE102():
    
    def __init__(self, month: str, year: str, yesterday_date: str):
        """
        Initializes the CRContributorICV102 class.

        Args:
            yesterday_date (str): The date for which data needs to be fetched. dd/mm/yyyy
        """
        logger.debug("Initializing CRContributorICV102 for Date: %s", yesterday_date)
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
            logger.warning("CRContributorICV102 Aborted: Daily lab report data for '%s' is not updated for %s.", table, self.yesterday_date)
            return
        
        logger.debug("Lab report update verified. Proceeding with data fetch.")

        logger.debug("Checking IP21 update status for %s", self.yesterday_date)
        is_updated, table = check_ip21_update(self.yesterday_date)
        if not is_updated:
            logger.warning("CRContributorICV102 Aborted: IP21 data for '%s' is not updated for %s.", table, self.yesterday_date)
            return

        logger.debug("IP21 data verified. Proceeding with data fetch.")
        
        #=============================== GET THE REQUIRED IP21 DATA ===============================
        ip_data = self.db_manager.special_method_ip_21_get_rows_for_day(self.yesterday_date)
        crude_temp = ip_data["Temp of Crude to IC-E-102 temp °C"]
        
        average_crude_temp = 0
        try:
            average_crude_temp = sigfig(celsius_to_kelvin(sum(crude_temp)/len(crude_temp)))
        except:
            average_crude_temp = None


        self.data_to_be_updated["Temp Crude (K)"] = average_crude_temp

        #================================ GET THE REQUIRED CRUDE BLEND DATA =============================
        blend_property_mapping = {
            "Molecular weight(g/mol)": "MW(gm/gmol)",
            "Thermal conductivity(W/m·K)": "k(W/m-K)",
            "DENSITY(g/mL)": "Density (kg/m3)",
            "Specific heat(J/kg·K)": "Cp(J/kg-k)",
            "Viscosity(Pa·s)": "viscosity(Pa-s)",
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

        
        # =============================== TOTAL CRUDE FLOW RATE ==============================
        flow_rate_max = 88 * 1.2                                                   
        flow_rate_min = 88 / 8 
        total_crude_flowrate = random.uniform(flow_rate_min, flow_rate_max)
        total_crude_flowrate = sigfig(tph_to_kg_per_s(total_crude_flowrate))
        self.data_to_be_updated["Total crude flow rate (kg/s)"] = total_crude_flowrate

        #=========================== Temp HE fluid (Cooling water )(K) ===========================
        cooling_water_base = 40 # in degree celsius
        temp_cooling_water_min = cooling_water_base - 20 
        temp_cooling_water_max = cooling_water_base + 20
        temp_cooling_water = random.uniform(temp_cooling_water_min, temp_cooling_water_max)
        self.data_to_be_updated["Temp HE fluid(K)"] = sigfig(celsius_to_kelvin(temp_cooling_water))

        #=========================== Flow rate HE fluid (Cooling water )(kg/s) ====================
        flow_rate_cooling_water_base = total_crude_flowrate / 1.5
        self.data_to_be_updated["Flow rate HE fluid(kg/s)"] = sigfig(flow_rate_cooling_water_base)

        #========================== mass fraction wash water =====================================
        flow_rate_neut_amine_base = 10 # in TPH
        flow_rate_neut_amine = sigfig(tph_to_kg_per_s(flow_rate_neut_amine_base))

        flow_rate_wash_water =  11.6 # in TPH
        flow_rate_wash_water = sigfig(tph_to_kg_per_s(flow_rate_wash_water))

        mf_wash_water  = flow_rate_wash_water  / (flow_rate_wash_water  + total_crude_flowrate + flow_rate_neut_amine)
        self.data_to_be_updated["mass fraction wash water"] = sigfig(mf_wash_water)

        #========================== mass fraction neutralizing amine ==============================
        mf_neut_amine = flow_rate_neut_amine / (flow_rate_wash_water  + total_crude_flowrate + flow_rate_neut_amine)
        self.data_to_be_updated["mass fraction neutralizing amine"] = sigfig(mf_neut_amine)

        #========================== Sulfur mass fraction ==============================
        mf_sulfur_min = 0.001 
        mf_sulfur_max = 0.05
        mf_sulfur = random.uniform(mf_sulfur_min, mf_sulfur_max)
        self.data_to_be_updated["Sulfur mass fraction"] = sigfig(mf_sulfur)

        #========================== H+ mass fraction ==============================
        mf_h_min = 0.0005 
        mf_h_max = 0.05
        mf_h = random.uniform(mf_h_min, mf_h_max)
        self.data_to_be_updated["H+ mass fraction"] = sigfig(mf_h)

        #======================= Crude tube ==============================
        mf_crude_tube = 1 - (mf_neut_amine + mf_sulfur + mf_wash_water + mf_h)
        self.data_to_be_updated["Crude tube"] = sigfig(mf_crude_tube)

        #======================= Crude shell +============================
        mf_crude_shell = 1 - ( mf_sulfur + mf_h)
        self.data_to_be_updated["Crude shell"] = sigfig(mf_crude_shell) 

        #======================= Flow rate in each tube(kg/s) ==============================
        flow_rate_in_each_tube = 6.178248652
        self.data_to_be_updated["Flow rate in each tube(kg/s)"] = sigfig(flow_rate_in_each_tube)
        

        # ================================ UPDATE THE DATABASE ===============================

        print(f"Data to be updated from icv113 contributor thread: {self.data_to_be_updated}")
        table_name = f"{self.year}_{self.month}_102_contributor"
        self.db_manager.update_a_row(self.db_name, table_name, "Date", self.yesterday_date, self.data_to_be_updated)

        data = self.db_manager.read_table(self.db_name, table_name)
        for row in data:
            print(row)

        logger.info("CRContributorICV102 completed for Date: %s", self.yesterday_date)


