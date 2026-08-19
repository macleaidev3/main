"""
Module: blend_properties_calculation.py

Calculates and updates crude blend properties for a given date.
"""

from datetime import datetime
import logging

from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import sigfig, month_short_name, format_date_long
from src.crude_blend.calculate_thermophysical_prop import ThermophysicalProperties

logger = logging.getLogger("SentinelApp")


class BlendPropertiesCalculation:
    """
    Calculates and updates blend properties for a given date.

    Blend properties include:

    - Blend Density
    - Blend API
    - Blend Sulphur
    - Blend VR
    - Specific Gravity
    - Molecular Weight
    - Specific Heat
    - Thermal Conductivity
    - Viscosity
    """
    IGNORED_ROWS = {
            "total crude",
            "total indigenous",
            "total ls",
            "total ms",
            "total hs",
            "total hs imported", "total ms imported", "total ls imported", "density", "api", "vr", "sulphur", "vr%", "", "total crude"

                        }

    def __init__(self) -> None:
        self.db_manager = DatabaseManager()
        self.thermophysical_properties = ThermophysicalProperties()
        self.db_name = "SentinelDB"


    def update_blend_properties(self, given_date: str) -> None:
        """
        Calculate and update blend properties for the supplied date.

        Parameters
        ----------
        given_date : str
            Date in DD/MM/YYYY format.
        """

        logger.info(
            "[BlendProperties] Started blend property calculation for %s",
            given_date,
        )

        # ----------------------------------------------------------
        # Parse date
        # ----------------------------------------------------------

        try:
            self.date, self.month, self.year = given_date.split("/")
            self.month = month_short_name()[int(self.month) - 1]
        except Exception:
            logger.exception(
                "[BlendProperties] Invalid date received: %s",
                given_date,
            )
            return

        self.table_name = f"blend_properties_{self.year}_{self.month}"
        blend_table = f"blend_{self.year}_{self.month}"

        logger.info(
            "[BlendProperties] Blend table : %s",
            blend_table,
        )

        # ----------------------------------------------------------
        # Read crude names and corresponding volumes
        # ----------------------------------------------------------
        crude_volume_dict = {}

        try:
            column_name = format_date_long(given_date)
            column_data = self.db_manager.read_columns(
                db_name=self.db_name,
                table_name=blend_table,
                column_names=[
                    "crude_name",
                    column_name,
                ],
            )

        except Exception:
            logger.exception(
                "[BlendProperties] Failed to read crude blend table."
            )
            return

        for crude_name, volume in column_data:
            if crude_name is None:
                continue

            if crude_name.strip().lower() in self.IGNORED_ROWS:
                continue

            if volume is None:
                continue

            if volume <= 0:
                continue

            crude_volume_dict[crude_name] = volume

        logger.info(
            "[BlendProperties] %d crude(s) found in blend.",
            len(crude_volume_dict),
        )

        if not crude_volume_dict:
            logger.warning(
                "[BlendProperties] No crude available for calculation."
            )
            return

        # ----------------------------------------------------------
        # Read crude properties
        # ----------------------------------------------------------

        # check if the crudes in blend table are in general crude table
        _crude_in_blend_table = set(crude_volume_dict.keys())
        _crude_in_general_crude_table = {
            crude[0].strip()
            for crude in self.db_manager.read_columns(
                db_name=self.db_name,
                table_name="crude_data",
                column_names=["Crude Name"],
            )
            if crude[0] is not None
        }

        _crude_not_in_general_crude_table = _crude_in_blend_table - _crude_in_general_crude_table

        if _crude_not_in_general_crude_table:
            logger.warning("========================= CRUDE NOT AVAILABLE! ============================")
            logger.warning(
                "[BlendProperties]Processing for date: %s. Crude(s) not in general crude table: %s. Please add them in general crude table.",
                given_date,
                ", ".join(_crude_not_in_general_crude_table),
            )
            logger.warning("===========================================================================")
            return
#====================================================================================================================================================
        crude_property_dict = {}

        for crude_name, volume in crude_volume_dict.items():

            try:

                density = self.db_manager.get_cell_value(
                    self.db_name,
                    "crude_data",
                    "SPG",
                    "Crude Name",
                    crude_name,
                )

                api = self.db_manager.get_cell_value(
                    self.db_name,
                    "crude_data",
                    "API",
                    "Crude Name",
                    crude_name,
                )

                sulphur = self.db_manager.get_cell_value(
                    self.db_name,
                    "crude_data",
                    "Sulphur, %",
                    "Crude Name",
                    crude_name,
                )

                vr = self.db_manager.get_cell_value(
                    self.db_name,
                    "crude_data",
                    "VR + SD %",
                    "Crude Name",
                    crude_name,
                )

            except Exception:
                logger.exception(
                    "[BlendProperties] Failed reading crude properties "
                    "for '%s'.",
                    crude_name,
                )
                continue

            crude_property_dict[crude_name] = {
                "Density": density,
                "API": api,
                "Sulphur": sulphur,
                "VR": vr,
                "volume": volume,
            }

        logger.info(
            "[BlendProperties] Successfully collected properties "
            "for %d crude(s).",
            len(crude_property_dict),
        )

        if not crude_property_dict:
            logger.error(
                "[BlendProperties] No crude properties available."
            )
            return

        missing_properties = []

        for crude_name, properties in crude_property_dict.items():
            for property_name, value in properties.items():
                if value is None:
                    missing_properties.append((crude_name, property_name))

        if missing_properties:
            logger.warning("=============== MISSING PROPERTIES ==================")
            for crude_name, property_name in missing_properties:
                logger.warning(
                    "Crude '%s' has no value for '%s'.",
                    crude_name,
                    property_name,
                )
            logger.warning("=======================================================")

            return

        

        # ----------------------------------------------------------
        # Calculate blend properties
        # ----------------------------------------------------------

        logger.info(
            "[BlendProperties] Calculating blend properties."
        )

        try:

            blend_density = sigfig(
                self._calculate_blend_density(crude_property_dict)
            )

            blend_api = sigfig(
                self._calculate_blend_api(crude_property_dict)
            )

            blend_vr = sigfig(
                self._calculate_blend_vr(crude_property_dict)
            )

            blend_sulphur = sigfig(
                self._calculate_blend_sulphur(crude_property_dict)
            )

        except Exception:
            logger.exception(
                "[BlendProperties] Failed while calculating blend properties."
            )
            return

        logger.info(
            "[BlendProperties] "
            "Density=%s API=%s Sulphur=%s VR=%s",
            blend_density,
            blend_api,
            blend_sulphur,
            blend_vr,
        )

        # ----------------------------------------------------------
        # Calculate thermophysical properties
        # ----------------------------------------------------------

        logger.info(
            "[BlendProperties] Calculating thermophysical properties."
        )

        try:

            thermo_props = (
                self.thermophysical_properties
                .get_thermophysical_properties(
                    blend_density,
                    blend_api,
                    blend_sulphur,
                    blend_vr,
                )
            )

            logger.info(
                "[BlendProperties] "
                "Specific Gravity=%s Molecular Weight=%s "
                "Specific Heat Cp=%s Thermal Conductivity=%s "
                "Viscosity mu=%s",
                thermo_props["Specific Gravity"],
                thermo_props["Molecular Weight"],
                thermo_props["Specific Heat Cp"],
                thermo_props["Thermal Conductivity"],
                thermo_props["Viscosity mu"],
            )

        except Exception:
            logger.exception(
                "[BlendProperties] "
                "Thermophysical property calculation failed."
            )
            return

        blend_properties = {

            "Blend Density": blend_density,

            "Blend API": blend_api,

            "Blend Sulphur": blend_sulphur,

            "Blend VR": blend_vr,

            "Specific Gravity":
                thermo_props["Specific Gravity"],

            "Molecular Weight":
                thermo_props["Molecular Weight"],

            "Specific Heat Cp":
                thermo_props["Specific Heat Cp"],

            "Thermal Conductivity":
                thermo_props["Thermal Conductivity"],

            "Viscosity mu":
                thermo_props["Viscosity mu"],
        }

        logger.info(
            "[BlendProperties] Thermophysical properties calculated successfully."
        )

        # ----------------------------------------------------------
        # Update Blend Property Table
        # ----------------------------------------------------------

        data_to_be_updated = {

            "DENSITY(g/mL)":
                blend_properties["Blend Density"],

            "API":
                blend_properties["Blend API"],

            "SULPHUR%":
                blend_properties["Blend Sulphur"],

            "VR%":
                blend_properties["Blend VR"],

            "Molecular weight(g/mol)":
                blend_properties["Molecular Weight"],

            "Specific heat(J/kg·K)":
                blend_properties["Specific Heat Cp"],

            "Thermal conductivity(W/m·K)":
                blend_properties["Thermal Conductivity"],

            "Viscosity(Pa·s)":
                blend_properties["Viscosity mu"],

            "Specific gravity":
                blend_properties["Specific Gravity"],
        }

        logger.info(
            "[BlendProperties] Updating blend property table."
        )

        try:

            self.db_manager.update_a_row(
                db_name=self.db_name,
                table=self.table_name,
                pk_column="Date",
                pk_value=given_date,
                data=data_to_be_updated,
            )

        except Exception:
            logger.exception(
                "[BlendProperties] Failed updating blend property table."
            )
            return

        logger.info(
            "[BlendProperties] Blend property table updated successfully."
        )

        # ----------------------------------------------------------
        # Read updated row (Debug)
        # ----------------------------------------------------------

        try:

            updated_rows = self.db_manager.read_table(
                self.db_name,
                self.table_name,
            )

            logger.debug(
                "[BlendProperties] Updated table contents:\n%s",
                updated_rows,
            )

        except Exception:
            logger.exception(
                "[BlendProperties] Unable to read updated blend property table."
            )

        logger.info(
            "[BlendProperties] Completed blend calculation for %s",
            given_date,
        )

    # ==================================================================
    # Blend Property Calculations
    # ==================================================================

    def _calculate_blend_density(
        self,
        crude_property_dict: dict,
    ) -> float:
        """
        Calculate volume-weighted blend density.
        """

        total_volume = 0.0
        weighted_density = 0.0

        for crude_name, props in crude_property_dict.items():

            try:

                volume = props["volume"]
                density = props["Density"]

                total_volume += volume
                weighted_density += volume * density

            except Exception:

                logger.exception(
                    "[BlendDensity] Error processing crude '%s'",
                    crude_name,
                )

        if total_volume == 0:

            logger.warning(
                "[BlendDensity] Total volume is zero."
            )

            return 0.0

        density = weighted_density / total_volume

        logger.debug(
            "[BlendDensity] Calculated density = %s",
            density,
        )

        return density

    def _calculate_blend_api(
        self,
        crude_property_dict: dict,
    ) -> float:
        """
        Calculate blend API using weighted Specific Gravity.
        """

        total_volume = 0.0
        weighted_sg = 0.0

        for crude_name, props in crude_property_dict.items():

            try:

                volume = props["volume"]

                api = props["API"]

                sg = 141.5 / (api + 131.5)

                total_volume += volume

                weighted_sg += volume * sg

            except Exception:

                logger.exception(
                    "[BlendAPI] Error processing crude '%s'",
                    crude_name,
                )

        if total_volume == 0:

            logger.warning(
                "[BlendAPI] Total volume is zero."
            )

            return 0.0

        blend_sg = weighted_sg / total_volume

        api = (141.5 / blend_sg) - 131.5

        logger.debug(
            "[BlendAPI] Calculated API = %s",
            api,
        )

        return api

    def _calculate_blend_sulphur(
            self,
            crude_property_dict: dict,
        ) -> float:
            """
            Calculate the volume-weighted sulphur content of the blend.
            """

            total_volume = 0.0
            weighted_sulphur = 0.0

            for crude_name, props in crude_property_dict.items():

                try:

                    volume = props["volume"]
                    sulphur = props["Sulphur"]

                    total_volume += volume
                    weighted_sulphur += volume * sulphur

                except Exception:

                    logger.exception(
                        "[BlendSulphur] Error processing crude '%s'.",
                        crude_name,
                    )

            if total_volume == 0:

                logger.warning(
                    "[BlendSulphur] Total blend volume is zero."
                )

                return 0.0

            sulphur = weighted_sulphur / total_volume

            logger.debug(
                "[BlendSulphur] Calculated blend sulphur = %.6f",
                sulphur,
            )

            return sulphur

    def _calculate_blend_vr(
        self,
        crude_property_dict: dict,
    ) -> float:
        """
        Calculate the volume-weighted Vacuum Residue (VR%) of the blend.
        """

        total_volume = 0.0
        weighted_vr = 0.0

        for crude_name, props in crude_property_dict.items():

            try:

                volume = props["volume"]
                vr = props["VR"]

                total_volume += volume
                weighted_vr += volume * vr

            except Exception:

                logger.exception(
                    "[BlendVR] Error processing crude '%s'.",
                    crude_name,
                )

        if total_volume == 0:

            logger.warning(
                "[BlendVR] Total blend volume is zero."
            )

            return 0.0

        vr = weighted_vr / total_volume

        logger.debug(
            "[BlendVR] Calculated blend VR = %.6f",
            vr,
        )

        return vr

    # ==================================================================
    # Utility Methods
    # ==================================================================

    @staticmethod
    def get_ordinal(day: int) -> str:
        """
        Convert a day number into its ordinal representation.

        Example
        -------
        1  -> 1st
        2  -> 2nd
        3  -> 3rd
        4  -> 4th
        21 -> 21st
        """

        if 11 <= day <= 13:
            suffix = "th"
        else:
            suffix = {
                1: "st",
                2: "nd",
                3: "rd",
            }.get(day % 10, "th")

        return f"{day}{suffix}"