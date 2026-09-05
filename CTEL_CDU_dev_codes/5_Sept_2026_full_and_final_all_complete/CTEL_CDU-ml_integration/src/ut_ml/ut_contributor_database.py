#### Created By ANURAG of IP21

import logging
from dataclasses import dataclass

logger = logging.getLogger("SentinelApp")

@dataclass
class BlendPropertiesResult:
    """Blend values in their source and contributor forms."""
    source_values: dict
    contributor_values: dict

class ContributorDatabaseManager:
    """
    Handles database operations associated with the UT
    contributor workflow.
    This class does not change the database API being used
    by the original UTThicknessContributor.
    """

    def __init__(self, db_manager, db_name, probe_id, month, year,
        yesterday_date,
        flags_column,
    ):
        self.db_manager = db_manager
        self.db_name = db_name
        self.probe_id = str(probe_id)
        self.month = str(month)
        self.year = str(year)
        self.yesterday_date = str(yesterday_date)
        self.flags_column = flags_column

    # ===================== TABLE NAMES =====================
    @property
    def thickness_table(self):
        return (f"ut_{self.probe_id}_{self.year}_{self.month}_thickness")

    @property
    def contributor_table(self):
        return (f"ut_{self.probe_id}_{self.year}_{self.month}_contributor")

    @property
    def blend_table(self):
        return (f"blend_properties_{self.year}_{self.month}")

    # ======================== WRITE FLAG=========================
    def write_flag(self, flag):
        table_name = self.thickness_table
        flag_data = {self.flags_column: flag}
        logger.info("[IP21] Writing recovery flag to thickness table '%s' for Date=%s.", table_name, self.yesterday_date)

        try:
            result = (self.db_manager.update_a_row(self.db_name,table_name,"Date",self.yesterday_date,flag_data))
            logger.info("[IP21] Thickness-table flag update result: %s", result)
            stored_flag = (self.db_manager.get_cell_value(self.db_name, table_name, self.flags_column,"Date", self.yesterday_date))

            if stored_flag != flag:
                logger.error("[IP21] FLAG VERIFICATION FAILED | Expected=%s | Stored=%s", flag, stored_flag)

                raise RuntimeError(f"Flag verification failed for {self.yesterday_date}. Expected={flag!r}, Stored={stored_flag!r}")

            logger.info("[IP21] FLAG SUCCESSFULLY STORED | Table=%s | Date=%s",   table_name, self.yesterday_date)
        except Exception:
            logger.exception("[IP21] Failed to write recovery flag to thickness table.")
            raise

    # ========================== BLEND PROPERTIES========================================

    def load_blend_properties(self,is_valid_number):
        logger.info("[Blend] Reading blend properties from %s for %s.", self.blend_table, self.yesterday_date)
        property_mapping = {
            "DENSITY(g/mL)": "Density(g/ml)",
            "API": "API",
            "SULPHUR%": "Sulphur%",
        }
        source_values = {}
        contributor_values = {}

        for source_column, target_column in (property_mapping.items()):
            value = (
                self.db_manager.get_cell_value(
                    self.db_name,
                    self.blend_table,
                    source_column,
                    "Date",
                    self.yesterday_date,
                )
            )
            if not is_valid_number(value):

                raise ValueError(f"Invalid/missing blend property '{source_column}' for {self.yesterday_date}. Value={value!r}")
            numeric_value = float(value)
            source_values[source_column] = numeric_value
            contributor_values[target_column] = numeric_value
            logger.info("[Blend] %s = %s",  target_column,  numeric_value)

        return BlendPropertiesResult(source_values=source_values, contributor_values=contributor_values)
    
    # =============================CONTRIBUTOR ROW=================================================
    def write_contributor_row(self, data_to_be_updated,  is_valid_number):

        required_inputs = ["Density(g/ml)", "API",  "Sulphur%"]
        for column in required_inputs:
            value = data_to_be_updated.get(column)
            if not is_valid_number(value):
                raise ValueError(f"Cannot update contributor table because {column} is missing/invalid for {self.yesterday_date}.")
        logger.info("[Contributor] Updating table: %s", self.contributor_table)
        result = (self.db_manager.update_a_row(self.db_name, self.contributor_table, "Date", self.yesterday_date, data_to_be_updated))
        logger.info("[Contributor] update_a_row result: %s", result)

        # -------------------------Verify stored values-----------------------------------------
        for column in required_inputs:
            stored_value = (self.db_manager.get_cell_value(self.db_name, self.contributor_table, column,"Date", self.yesterday_date))
            if not is_valid_number(stored_value):
                raise RuntimeError(f"Contributor database verification failed for {self.yesterday_date}. Table={self.contributor_table}, "
                    f"Column={column}, "
                    f"StoredValue={stored_value!r}"
                )
            logger.info("[Contributor] VERIFIED | %s = %s", column, stored_value)