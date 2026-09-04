
import logging
from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import  check_ip21_update, check_daily_lab_report_data_update, sigfig
from src.utils.unit_conversion import celsius_to_kelvin
import random
from src.utils.unit_conversion import tph_to_kg_per_s


# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class CRContributorICV112():
    
    def __init__(self, month: str, year: str, yesterday_date: str):
        """
        Initializes the CRContributorICV112 class.

        Args:
            yesterday_date (str): The date for which data needs to be fetched. dd/mm/yyyy
        """
        logger.debug("Initializing CRContributorICV112 for Date: %s", yesterday_date)
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
            logger.warning("CRContributorICV112 Aborted: Daily lab report data for '%s' is not updated for %s.", table, self.yesterday_date)
            return
        
        logger.debug("Lab report update verified. Proceeding with data fetch.")

        logger.debug("Checking IP21 update status for %s", self.yesterday_date)
        is_updated, table = check_ip21_update(self.yesterday_date)
        if not is_updated:
            logger.warning("CRContributorICV112 Aborted: IP21 data for '%s' is not updated for %s.", table, self.yesterday_date)
            return

        logger.debug("IP21 data verified. Proceeding with data fetch.")
        
        #=============================== GET THE REQUIRED IP21 DATA ===============================
        ip_data = self.db_manager.special_method_ip_21_get_rows_for_day(self.yesterday_date)
        temp_overhead_vapor = ip_data["Temp of overhead vapor from reflux drum °C"]
    
        average_temp_overhead_vapor = 0
        try:
            average_temp_overhead_vapor = sigfig(celsius_to_kelvin(sum(temp_overhead_vapor)/len(temp_overhead_vapor)))
        except:
            average_temp_overhead_vapor = None
        
        self.data_to_be_updated["Temperature Overhead drum (K)"] = average_temp_overhead_vapor

        #================================ GET THE REQUIRED CRUDE BLEND DATA =============================
        blend_property_mapping = {
            "Molecular weight(g/mol)": "MW(g/gmol)",
            "Thermal conductivity(W/m·K)": "k (W/m-k)",
            "DENSITY(g/mL)": "Density(kg/m3)",
            "Specific heat(J/kg·K)": "Cp (J/kg-K)",
            "Viscosity(Pa·s)": "Viscosity (Pa-s)",
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


        #==================================== Flow rate at inlet (kg/s) ===========================
        flow_rate_max = 352 * 1.2                                                   
        flow_rate_min = 352 / 8
        flow_rate_inlet = random.uniform(flow_rate_min, flow_rate_max)
        flow_rate_inlet = sigfig(tph_to_kg_per_s(flow_rate_inlet))
        self.data_to_be_updated["Flow rate at inlet (kg/s)"] = flow_rate_inlet

        #=================================== Crude temperature(K) =================================
        min_crude_temp =  340.313
        max_crude_temp = 414.3236
        crude_temp = random.uniform(min_crude_temp, max_crude_temp)
        self.data_to_be_updated["Crude temperature(K)"] = crude_temp

        #=================================== Inlet split ratio ====================================
        inlet_split_ratio = random.uniform(0.6252168859, 0.945525097)
        self.data_to_be_updated["Inlet split ratio"] = inlet_split_ratio

        #==================================== Flow rate crude inlet(kg/s) ==========================
        phase_split_ratio = random.uniform(0.6, 0.95)
        crude_flow_rate = flow_rate_inlet * phase_split_ratio
        self.data_to_be_updated["Flow rate crude inlet(kg/s)"] = crude_flow_rate

        #=================================  Flow rate ww inlet(kg/s) ================================
        flow_rate_w_inlet = crude_flow_rate - crude_flow_rate
        self.data_to_be_updated["Flow rate ww inlet(kg/s)"] = flow_rate_w_inlet

        #================================ Split ratio outlet 1 ====================================
        flow_rate_out1_base = flow_rate_inlet * 0.6 
        flow_rate_out2_base = flow_rate_inlet * 0.2
        flow_rate_out3_base = flow_rate_inlet * 0.2
        split_ration_outlet1 =  sigfig(flow_rate_out1_base / ( flow_rate_out1_base + flow_rate_out2_base + flow_rate_out3_base))
        split_out1_min = 0.5 * split_ration_outlet1
        split_out1_max = 1.2 * split_ration_outlet1
        split_ration_outlet1 = random.uniform(split_out1_min, split_out1_max)
        self.data_to_be_updated["Split ratio outlet 1"] = split_ration_outlet1

        #================================ Split ratio outlet 2 =====================================
        flow_rate_out1_base = flow_rate_inlet * 0.8
        flow_rate_out2_base = flow_rate_inlet * 0.2
        flow_rate_out3_base = flow_rate_inlet * 0.2
        split_ration_outlet2 =  sigfig(flow_rate_out1_base / ( flow_rate_out1_base + flow_rate_out2_base + flow_rate_out3_base))
        split_out2_min = 0.5 * split_ration_outlet2
        split_out2_max = 1.2 * split_ration_outlet2
        split_ration_outlet2 = random.uniform(split_out2_min, split_out2_max)
        self.data_to_be_updated["Split ratio outlet 2"] = split_ration_outlet2

        #================================= Flow rate outlet 1(kg/s) ===============================
        flow_rate_outlet_1 =  flow_rate_inlet * split_ration_outlet1
        self.data_to_be_updated["Flow rate outlet 1(kg/s)"] = flow_rate_outlet_1

        #================================= Flow rate outlet 2(kg/s) ===============================
        flow_rate_outlet_2 =  flow_rate_inlet * split_ration_outlet2
        self.data_to_be_updated["Flow rate outlet 2(kg/s)"] = flow_rate_outlet_2

        #================================= Flow rate outlet 3(kg/s) ===============================
        flow_rate_outlet_3 =  flow_rate_inlet - (flow_rate_outlet_1 + flow_rate_outlet_2) 
        self.data_to_be_updated["Flow rate outlet 3(kg/s)"] = flow_rate_outlet_3

        #================================= H+ =====================================================
        mf_H_plus_min = 0.0005 
        mf_H_plus_max = 0.05
        H_plus = random.uniform(mf_H_plus_min, mf_H_plus_max)
        self.data_to_be_updated["H+"] = H_plus

        #================================= Sulfur =====================================================
        mf_sulphur_min = 0.001
        mf_sulphur_max = 0.05
        sulphur_mf = random.uniform(mf_sulphur_min, mf_sulphur_max)
        self.data_to_be_updated["Sulfur"] = sulphur_mf

        #================================= crude inlet mf =====================================================
        crude_mf_inlet = 1 - (H_plus + sulphur_mf)
        self.data_to_be_updated["crude inlet mf"] = crude_mf_inlet

        logger.info(f"Data to be updated from icv112 contributor thread: {self.data_to_be_updated}")
        table_name = f"{self.year}_{self.month}_112_contributor"
        self.db_manager.update_a_row(self.db_name, table_name, "Date", self.yesterday_date, self.data_to_be_updated)

        # data = self.db_manager.read_table(self.db_name, table_name)
        # for row in data:
        #     print(row)

        logger.info("CRContributorICV112 completed for Date: %s", self.yesterday_date)


