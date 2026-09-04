USE kr_db;
GO

-- 1. Drop the table if it already exists
DROP TABLE IF EXISTS ip21_table;
GO

-- 2. Recreate the table with the Primary Key included
CREATE TABLE ip21_table (
    [col_Time] NVARCHAR(255) PRIMARY KEY,
    [col_CDU_col_Top_temp_°C] FLOAT,
    [col_CDU_col_Top_press_kg_cm2g] FLOAT,
    [col_Flow_of_CDU_reflux_to_col_TPH] FLOAT,
    [col_Stripping_steam_to_column_TPH] FLOAT,
    [col_Temp_of_Crude_to_IC_E_102_temp_°C] FLOAT,
    [col_Temp_of_Crude_from_IC_E_102_temp_°C] FLOAT,
    [col_Temp_of_Overhead_vap_from_IC_E_102A_°C] FLOAT,
    [col_Temp_of_Overhead_vap_from_IC_E_102B_°C] FLOAT,
    [col_Temp_of_Overhead_vap_from_IC_E_102C_°C] FLOAT,
    [col_Temp_of_Overhead_vap_from_IC_E_102D_°C] FLOAT,
    [col_Temp_of_Wash_water_to_IC_E_102_°C] FLOAT,
    [col_Press_Of_Wash_water_to_IC_E_102_kg_cm2g] FLOAT,
    [col_Flow_of_Wash_water_to_IC_E_102A_TPH] FLOAT,
    [col_Flow_of_Wash_water_to_IC_E_102B_TPH] FLOAT,
    [col_Flow_of_Wash_water_to_IC_E_102C_TPH] FLOAT,
    [col_Flow_of_Wash_water_to_IC_E_102D_TPH] FLOAT,
    [col_Temp_of_overhead_vapor_from_air_cooler_°C] FLOAT,
    [col_Pressure_at_reflux_drum_IC_V_112_kg_cm2g] FLOAT,
    [col_Temp_of_overhead_vapor_from_reflux_drum_°C] FLOAT,
    [col_Flow_of_sour_water_from_IC_V_112_TPH] FLOAT,
    [col_Temp_of_sour_water_from_IC_V_112_°C] FLOAT,
    [col_Flow_of_reflux_to_IC_V_101_TPH] FLOAT,
    [col_Temp_of_reflux_from_reflux_drum_°C] FLOAT,
    [col_Temp_of_o_h_naphtha_from_IC_E_162_°C] FLOAT,
    [col_Wash_water_to_IC_E_162__1__TPH] FLOAT,
    [col_Wash_water_to_IC_E_162__2__TPH] FLOAT,
    [col_Wash_water_to_IC_E_162__3__TPH] FLOAT,
    [col_Wash_water_to_IC_E_162__4__TPH] FLOAT,
    [col_Wash_water_to_IC_E_162__5__TPH] FLOAT,
    [col_Wash_water_to_IC_E_162__6__TPH] FLOAT,
    [col_Wash_water_to_IC_E_162__7__TPH] FLOAT,
    [col_Wash_water_to_IC_E_162__8__TPH] FLOAT,
    [col_O_h_naphtha_temp_°C] FLOAT,
    [col_Temp_of_Naphtha_from_IC_E_126A_B_°C] FLOAT,
    [col_Temp_of_Naphtha_from_IC_E_126C_D_°C] FLOAT,
    [col_Temp_of_Naphtha_from_IC_E_126_°C] FLOAT,
    [col_Flow_of_sour_water_from_IC_V_113_TPH] FLOAT,
    [col_Temp_of_unstab_naphtha_from_IC_V_113_°C] FLOAT,
    [col_Unstab_naphtha_bypass_TPH] FLOAT,
    [col_Flow_of_unstab_naphtha_to_IC_V_106_TPH] FLOAT,
    [col_Pressure_in_IC_V_113_kg_cm2g] FLOAT
);
GO