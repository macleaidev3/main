from __future__ import annotations

import pandas as pd

class PredictionInputBuilder:
    """
    Builds the ML input DataFrame from the coordinate and contributor tables.
    """

    # ------------------------------------------------------------------
    # Coordinate table mapping
    # ML Feature -> Coordinate table column
    # ------------------------------------------------------------------

    COORDINATE_COLUMN_MAPPING = {
        "S no": "S no",
        "x-coordinate": "X",
        "y-coordinate": "Y",
        "z-coordinate": "Z",
        "r": "r",
        "theta": "theta",
        "phi": "phi",
    }
    
    # ------------------------------------------------------------------
    # Contributor table mapping
    # ML Feature -> Contributor table column
    # ------------------------------------------------------------------

    CONTRIBUTOR_COLUMN_MAPPING = {
        "Crude temperature(K)": "crude temperature(k)",
        "MW(g/gmol)": "mw(g/gmol)",
        "k (W/m-k)": "k (w/m-k)",
        "Density(kg/m3)": "density(kg/m3)",
        "Cp (J/kg-K)": "cp (j/kg-k)",
        "Viscosity (Pa-s)": "viscosity (pa-s)",
        "Sulfur": "sulfur mf",
        "H+": "h+ mf",
        "Temperature Overhead drum (K)": "temperature overhead drum (k)",
        "Flow rate crude inlet(kg/s)": "flow rate crude inlet(kg/s)",
        "Flow rate ww inlet(kg/s)": "flow rate ww inlet(kg/s)",
        "Flow rate outlet 1(kg/s)": "flow rate outlet 1(kg/s)",
        "Flow rate outlet 2(kg/s)": "flow rate outlet 2(kg/s)",
        "Crude inlet mf": "crude inlet mf",
    }

    def __init__(
        self,
        db_manager,
        db_name: str,
        year: int,
        month: str,
        equipment: str,
        prediction_date: str,
    ):
        """
        Parameters
        ----------
        equipment
            Example: "113", "112"
        prediction_date
            Date whose contributor values will be used.
        """

        self.db_manager = db_manager
        self.db_name = db_name
        self.year = year
        self.month = month
        self.equipment = equipment
        self.prediction_date = prediction_date

    def build(self) -> pd.DataFrame:
        """
        Creates the DataFrame required for the ML models.
        """

        coordinate_df = self._read_coordinate_data()
        

        self._append_contributor_data(coordinate_df)

        return coordinate_df

    def _read_coordinate_data(self) -> pd.DataFrame:
        """
        Read coordinate table.
        """

        coordinate_table = (
            f"{self.year}_{self.equipment}_cr"
        )

        db_columns = list(self.COORDINATE_COLUMN_MAPPING.values())

        coordinate_data = self.db_manager.read_columns(
            self.db_name,
            coordinate_table,
            db_columns,
        )
        
        coordinate_data = [tuple(row) for row in coordinate_data]

        df = pd.DataFrame(
            coordinate_data,
            columns=list(self.COORDINATE_COLUMN_MAPPING.keys()),
        )

        return df

    def _append_contributor_data(self, df: pd.DataFrame) -> None:
        """
        Read contributor values and append them to every row.
        """

        contributor_table = (
            f"{self.year}_{self.month}_{self.equipment}_contributor"
        )

        for feature_name, db_column in self.CONTRIBUTOR_COLUMN_MAPPING.items():

            value = self.db_manager.get_cell_value(
                self.db_name,
                contributor_table,
                db_column,
                "Date",
                self.prediction_date,
            )

            # Broadcast scalar to all rows
            df[feature_name] = value