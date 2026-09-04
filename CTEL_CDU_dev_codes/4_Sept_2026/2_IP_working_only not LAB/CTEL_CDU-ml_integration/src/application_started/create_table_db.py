import logging
from src.server_manager.operation_manager import DatabaseManager
from src.utils.table_columns import TABLE_COLUMNS
from src.utils.core_utility_functions import get_year_range, month_short_name, year_month_days_list_of_dict, get_days_in_month, resource_path
from PyQt6 import QtCore
import pandas as pd

class CreateAllDB(QtCore.QObject):

    progress = QtCore.pyqtSignal(int, str)  
    finished = QtCore.pyqtSignal()

    def __init__(self):
        """
        CREATE ALL DATABASE TABLES FOR SQL SERVER
        """
        super().__init__()
        
        # Define your single SQL Server database name here.
        # NOTE: This database must already be created in SQL Server Management Studio 
        # (e.g., executing "CREATE DATABASE SentinelDB").
        self.target_db = "SentinelDB"

        

    @QtCore.pyqtSlot()
    def run(self):
        logger = logging.getLogger("SentinelApp")
        logger.info("Starting database table initialization for target DB: %s", self.target_db)
        
        try:
            DB_MANAGER = DatabaseManager()
    
            year_range = get_year_range()
            month_short_names = month_short_name()

            # all_table = DB_MANAGER.list_tables(self.target_db)
            # for table in all_table:
            #     DB_MANAGER.drop_table(self.target_db, table)

            #======================== GENERAL CRUDE DATABASE ========================
            table_name = "crude_data"
            columns_data = TABLE_COLUMNS["crude_data"]
            logger.debug("Checking/Creating general crude database table: %s", table_name)
            DB_MANAGER.create_table(self.target_db, table_name, columns_data)

            for i, year in enumerate(year_range):
                pct = int((i / len(year_range)) * 100)
                self.progress.emit(pct, f"Initializing Sentinel database tables for {year}...(it may take a while on first launch)")
                
                logger.info("Initializing database tables for year: %s (%d%% complete)", year, pct)
                year_month_day_dict = year_month_days_list_of_dict(year)

                #======================== CRUDE BLEND DATABASE ========================
                logger.debug("Generating crude blend tables for year %s...", year)
                for month_name in month_short_names:
                    _db_column = [("crude_name", "NVARCHAR(255)")]
                    _days_in_month = year_month_day_dict[year][month_name] 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "FLOAT"))
                    
                    # Flat naming: blend_2026_Jan
                    table_name = f"blend_{year}_{month_name}"
                    DB_MANAGER.create_table(self.target_db, table_name, _db_column)

                #========================== CRUDE BLEND PROPERTIES TABLE CREATION ==========================
                logger.debug("Generating crude blend properties tables for year %s...", year)
                crude_blend_db_columns = TABLE_COLUMNS.get("crude_blend_properties")

                month_dict = {sort_month: idx + 1 for idx, sort_month in enumerate(month_short_names)}
                
                for month in month_short_names:
                    # Flat naming: blend_properties_2026_Jan
                    blend_properties_table = f"blend_properties_{year}_{month}"
                    DB_MANAGER.create_table(self.target_db, table_name=blend_properties_table, columns=crude_blend_db_columns)
                    # DB_MANAGER.drop_table(self.target_db, table_name=blend_properties_table)
                    
                    # if year == 2025 and month == "Dec":
                    #     data = DB_MANAGER.read_table(self.target_db, blend_properties_table)
                    #     for d in data:
                    #         print(d)

                    if DB_MANAGER.row_count(self.target_db, table_name=blend_properties_table) == 0:
                        logger.debug("Populating initial date column for %s", blend_properties_table)
                        date_list = get_days_in_month(year, month_dict[month])
                        DB_MANAGER.insert_column(
                            db_name=self.target_db,
                            table_name=blend_properties_table,
                            column_name="Date",
                            values=date_list
                        )
                                

                # ======================= UT THICKNESS CONTRIBUTORS AND THICKNESS =======================
                contributor_db_columns = TABLE_COLUMNS.get("ut_thickness_contributor")
                thickness_db_columns = TABLE_COLUMNS.get("ut_thickness")

                month_dict = {sort_month: idx + 1 for idx, sort_month in enumerate(month_short_names)}
                probes_id_list = ["00001", "00003", "00004", "00005", "00006", "00029", "00030"]

                logger.debug("Generating UT thickness and contributor tables for %s probes...", len(probes_id_list))
                for probe_id in probes_id_list:
                    for month in month_short_names:
                        # Flat naming: ut_00001_2026_Jan_contributor
                        contributor_table = f"ut_{probe_id}_{year}_{month}_contributor"
                        DB_MANAGER.create_table(self.target_db, table_name=contributor_table, columns=contributor_db_columns) ##coment out
                        # DB_MANAGER.drop_table(self.target_db, table_name=contributor_table)

                        if DB_MANAGER.row_count(self.target_db, table_name=contributor_table) == 0: ##coment out
                            logger.debug("Populating initial date column for %s", contributor_table)
                            date_list = get_days_in_month(year, month_dict[month])
                            DB_MANAGER.insert_column(
                                db_name=self.target_db,
                                table_name=contributor_table,
                                column_name="Date",
                                values=date_list
                            )  

                        # Flat naming: ut_00001_2026_Jan_thickness 
                        thickness_table = f"ut_{probe_id}_{year}_{month}_thickness"
                        DB_MANAGER.create_table(self.target_db, table_name=thickness_table, columns=thickness_db_columns) ##coment out
                        # DB_MANAGER.drop_table(self.target_db, table_name=thickness_table)

                        if DB_MANAGER.row_count(self.target_db, table_name=thickness_table) == 0: ##coment out
                            logger.debug("Populating initial date column for %s", thickness_table)
                            date_list = get_days_in_month(year, month_dict[month]) 
                            DB_MANAGER.insert_column(
                                db_name=self.target_db,
                                table_name=thickness_table,
                                column_name="Date",
                                values=date_list
                            )
                

                #========================== LAB REPORT DATABASE CREATION ==========================
                table_name_list = ["after_desalter_stage_1", "after_desalter_stage_2", "crude_before_desalter", "sour_water_icv112", "sour_water_icv113", "stripped_water"]
                
                logger.debug("Generating lab report tables for year %s...", year)
                for month in month_short_names:
                    for base_table_name in table_name_list:
                        columns_data = TABLE_COLUMNS.get(base_table_name)
                        # Flat naming: lab_2026_Jan_after_desalter_stage_1
                        full_table_name = f"lab_{year}_{month}_{base_table_name}"
                        DB_MANAGER.create_table(self.target_db, table_name=full_table_name, columns=columns_data)


                # ======================= DATABASE CREATION IP21 =======================
                column_name = TABLE_COLUMNS.get("ip21_data")
                logger.debug("Generating IP21 data tables for year %s...", year)
                for month in month_short_names:
                    # Flat naming: ip21_2026_Jan
                    table_name = f"ip21_{year}_{month}"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=column_name)


                # ======================= IC-V-113 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("icv113_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_113_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    ## ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_113_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICV113 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/icv113_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]
                r_df = df["r"]
                theta_df = df["theta"]
                phi_df = df["phi"]

                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))


                table_name = f"{year}_113_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)

                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df, "r": r_df, "theta": theta_df, "phi": phi_df})

                #========================================================================
                # ======================= IC-V-112 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("icv112_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_112_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    ## ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_112_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICV112 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/icv112_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]
                r_df = df["r"]
                theta_df = df["theta"]
                phi_df = df["phi"]

                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))


                table_name = f"{year}_112_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)

                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df, "r": r_df, "theta": theta_df, "phi": phi_df})

                # =======================102-161 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("102_to_161_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_102_to_161_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    # DB_MANAGER.drop_table(self.target_db, table_name)
                    # ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_102_to_161_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICE102-161 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/102_to_161_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]
              
                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))


                table_name = f"{year}_102_to_161_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)

                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df})


                # =======================101-102 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("101_to_102_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_101_to_102_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    # DB_MANAGER.drop_table(self.target_db, table_name)
                    # ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_101_to_102_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICE102-161 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/101_to_102_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]
              
                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))


                table_name = f"{year}_101_to_102_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)

                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df})



                # =======================112-162 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("112_to_162_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_112_to_162_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    # DB_MANAGER.drop_table(self.target_db, table_name)
                    # ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_112_to_162_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICE112-162 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/112_to_162_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]
                

                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))


                table_name = f"{year}_112_to_162_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)

                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df})
                

                # =======================126-113 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("126_to_113_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_126_to_113_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    # DB_MANAGER.drop_table(self.target_db, table_name)
                    # ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_126_to_113_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICE126-113 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/126_to_113_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]

                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))


                table_name = f"{year}_126_to_113_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)

                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df})

                # =======================161-112 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("161_to_112_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_161_to_112_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    # DB_MANAGER.drop_table(self.target_db, table_name)
                    # ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_161_to_112_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICE161-112 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/161_to_112_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]
                
                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))
                
                table_name = f"{year}_161_to_112_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)
        
                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df})
   
                # =======================162-126 TABLE CREATION =======================
                # get the database columns
                contributor_db_columns = TABLE_COLUMNS.get("162_to_126_cr_contributor")
                for month in month_short_names:
                    # create contributor table
                    table_name = f"{year}_{month}_162_to_126_contributor"
                    DB_MANAGER.create_table(self.target_db, table_name=table_name, columns=contributor_db_columns)
                    # DB_MANAGER.drop_table(self.target_db, table_name)
                    # ---- Check if date column already has data ----
                    row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                    if row_count == 0:
                        table_info = DB_MANAGER.get_table_info(self.target_db, table_name)
                        col_length = len(table_info)
                        
                        remaining_col = col_length - 1
                        filling_tuple = (None,) * remaining_col

                        date_list = get_days_in_month(year, month_dict[month])
                        rows = []
                        for d in date_list:
                            rows.append(tuple((d,)) + filling_tuple)

                        DB_MANAGER.insert_rows(
                            db_name=self.target_db,
                            table_name=f"{year}_{month}_162_to_126_contributor",
                            rows = rows
                        )
                    else:
                        # print(f"Table '{month}' already initialized → skipping insertion")
                        pass

                # =======================    DATABASE CREATION OF ICE162-126 CORROSION PROFILES    =======================
                _db_column = [("S no", "INT"), ("X", "FLOAT"), ("Y", "FLOAT"), ("Z", "FLOAT"), ("r", "FLOAT"),("theta", "FLOAT"),("phi", "FLOAT")]
                csv_path = resource_path("assets/model_coordinalte_excels/162_to_126_coordinates.csv")
                df = pd.read_csv(csv_path)

                x_df = df["x-coordinate"]
                y_df = df["y-coordinate"]
                z_df = df["z-coordinate"]
                

                for month_name in month_short_names:
                    
                    _days_in_month = year_month_day_dict[year][month_name] # act as columns of a particular month table 
                    
                    for day in _days_in_month:
                        _db_column.append((f"{day}", "NVARCHAR(50)"))
                
                
                table_name = f"{year}_162_to_126_cr"
                DB_MANAGER.create_table(self.target_db, table_name, _db_column)
                # DB_MANAGER.drop_table(self.target_db, table_name)
        
                # ---- Check if the coordinate columns already have data ----
                row_count = DB_MANAGER.row_count(self.target_db, table_name=table_name)

                if row_count == 0:
                    DB_MANAGER.insert_columns(db_name=self.target_db, table_name=table_name, column_data={"X": x_df, "Y": y_df, "Z": z_df})

                
            
            logger.info("Database table initialization completed successfully.")
            self.progress.emit(100, "Initialization complete.")
            self.finished.emit()

        except Exception:
            # Catching and logging any unexpected errors during thread execution
            logger.exception("A critical error occurred during database initialization.")
            self.progress.emit(100, "Initialization failed. Check logs.")
            self.finished.emit()

            