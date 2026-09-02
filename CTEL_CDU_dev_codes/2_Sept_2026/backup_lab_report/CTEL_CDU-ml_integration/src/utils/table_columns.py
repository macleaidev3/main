"""
table_columns.py
Centralized definitions of all table column names. 
Used by both UI (PyQt) and database logic.
"""
TABLE_COLUMNS = {

    "crude_data": [
        ("Crude Name", "NVARCHAR(255)"),
        ("PIMS code", "NVARCHAR(255)"),
        ("Origin", "NVARCHAR(255)"),
        ("SPG", "FLOAT"),
        ("API", "FLOAT"),
        ("Sulphur, %", "FLOAT"),
        ("Sulphur category", "NVARCHAR(255)"),
        ("PIMS", "NVARCHAR(255)"),
        ("Pour Point OC", "FLOAT"),
        ("KV Cst @ 40OC", "FLOAT"),
        ("TAN", "FLOAT"),
        ("MCRT", "FLOAT"),
        ("FG+LPG%", "FLOAT"),
        ("SN%", "FLOAT"),
        ("HN%", "FLOAT"),
        ("Kero / Lt Kero %", "FLOAT"),
        ("Hy Kero + Diesel %", "FLOAT"),
        ("Kero + Diesel %", "FLOAT"),
        ("VGO %", "FLOAT"),
        ("VR + SD %", "FLOAT"),
        ("VGO, S%", "FLOAT"),
        ("VR, S%", "FLOAT"),
        ("Kero Merc, ppm", "FLOAT"),
        ("Diesel Base, S, ppm", "FLOAT"),
        ("Diesel Base, CI", "FLOAT"),
        ("VGO - Total N2, ppm", "FLOAT"),
        ("VGO - Basic N2, ppm", "FLOAT"),
        ("VR, MCRT (wt%)", "FLOAT"),
        ("C5-60 (wt%)", "FLOAT"),
        ("60-90 (wt%)", "FLOAT"),
        ("90-110 (wt%)", "FLOAT"),
        ("110-140 (wt%)", "FLOAT"),
        ("C5-60 SPG", "FLOAT"),
        ("60-90 SPG", "FLOAT"),
        ("90-110 SPG", "FLOAT"),
        ("110-140 SPG", "FLOAT"),
        ("SN, P - vol%", "FLOAT"),
        ("C5-60 P - vol%", "FLOAT"),
        ("60-90 P - vol%", "FLOAT"),
        ("90-110 P - vol%", "FLOAT"),
        ("110-140 P - vol%", "FLOAT"),
        ("SN, SPG", "FLOAT"),
        ("N+2A, 90-140 cut", "FLOAT"),
        ("Naphthenes, wt%", "FLOAT"),
        ("Aromatics, wt%", "FLOAT"),
    ],

    "ip21_data": [
        ("Time", "NVARCHAR(255)"),  # Use DATETIME2 if this represents an exact timestamp
        ("CDU col Top temp °C", "FLOAT"),
        ("CDU col Top press kg/cm2g", "FLOAT"),
        ("Flow of CDU reflux to col TPH", "FLOAT"),
        ("Stripping steam to column TPH", "FLOAT"),
        ("Temp of Crude to IC-E-102 temp °C", "FLOAT"),
        ("Temp of Crude from IC-E-102 temp °C", "FLOAT"),
        ("Temp of Overhead vap from IC-E-102A °C", "FLOAT"),
        ("Temp of Overhead vap from IC-E-102B °C", "FLOAT"),
        ("Temp of Overhead vap from IC-E-102C °C", "FLOAT"),
        ("Temp of Overhead vap from IC-E-102D °C", "FLOAT"),
        ("Temp of Wash water to IC-E-102 °C", "FLOAT"),
        ("Press Of Wash water to IC-E-102 kg/cm2g", "FLOAT"),
        ("Flow of Wash water to IC-E-102A TPH", "FLOAT"),
        ("Flow of Wash water to IC-E-102B TPH", "FLOAT"),
        ("Flow of Wash water to IC-E-102C TPH", "FLOAT"),
        ("Flow of Wash water to IC-E-102D TPH", "FLOAT"),
        ("Temp of overhead vapor from air cooler °C", "FLOAT"),
        ("Pressure at reflux drum IC-V-112 kg/cm2g", "FLOAT"),
        ("Temp of overhead vapor from reflux drum °C", "FLOAT"),
        ("Flow of sour water from IC-V-112 TPH", "FLOAT"),
        ("Temp of sour water from IC-V-112 °C", "FLOAT"),
        ("Flow of reflux to IC-V-101 TPH", "FLOAT"),
        ("Temp of reflux from reflux drum °C", "FLOAT"),
        ("Temp of o/h naphtha from IC-E-162 °C", "FLOAT"),
        ("Wash water to IC-E-162 (1) TPH", "FLOAT"),
        ("Wash water to IC-E-162 (2) TPH", "FLOAT"),
        ("Wash water to IC-E-162 (3) TPH", "FLOAT"),
        ("Wash water to IC-E-162 (4) TPH", "FLOAT"),
        ("Wash water to IC-E-162 (5) TPH", "FLOAT"),
        ("Wash water to IC-E-162 (6) TPH", "FLOAT"),
        ("Wash water to IC-E-162 (7) TPH", "FLOAT"),
        ("Wash water to IC-E-162 (8) TPH", "FLOAT"),
        ("O/h naphtha temp °C", "FLOAT"),
        ("Temp of Naphtha from IC-E-126A/B °C", "FLOAT"),
        ("Temp of Naphtha from IC-E-126C/D °C", "FLOAT"),
        ("Temp of Naphtha from IC-E-126 °C", "FLOAT"),
        ("Flow of sour water from IC-V-113 TPH", "FLOAT"),
        ("Temp of unstab naphtha from IC-V-113 °C", "FLOAT"),
        ("Unstab naphtha bypass TPH", "FLOAT"),
        ("Flow of unstab naphtha to IC-V-106 TPH", "FLOAT"),
        ("Pressure in IC-V-113 kg/cm2g", "FLOAT")
    ],

    "ut_thickness_contributor": [
        ("Date", "NVARCHAR(255)"),
        ("Density(g/ml)", "FLOAT"),
        ("API", "FLOAT"),
        ("Sulphur%", "FLOAT")
    ],

    "ut_thickness": [
        ("Date", "NVARCHAR(255)"),
        ("Status", "NVARCHAR(255)"), # Consider changing to NVARCHAR if Status is a word instead of a number
        ("Check on", "NVARCHAR(255)"),
        ("Thickness(mm)", "FLOAT"),
        ("Flag", "NVARCHAR(255)")
    ],

#     "global_min_max_cr": [
#         # ("primary_key", "INT PRIMARY KEY IDENTITY(1,1)"),
#         ("global_min", "FLOAT"),
#         ("global_max", "FLOAT")
#     ],

    "after_desalter_stage_1": [
        # ("ID", "INT PRIMARY KEY IDENTITY(1,1)"),
        ("Date", "NVARCHAR(255)"),
        ("Density(g/ml)", "FLOAT"),
        ("Salt(PTB)", "FLOAT"),
        ("BSW(%vol)", "FLOAT"),
        ("Sulphur(% by mass)", "FLOAT")
    ],
    
    "after_desalter_stage_2": [
        # ("ID", "INT PRIMARY KEY IDENTITY(1,1)"),
        ("Date", "NVARCHAR(255)"),
        ("Salt(PTB)", "FLOAT"),
        ("BSW(%vol)", "FLOAT"),
    ],

    "crude_before_desalter": [
        # ("ID", "INT PRIMARY KEY IDENTITY(1,1)"),
        ("Date", "NVARCHAR(255)"),
        ("Density(g/ml)", "FLOAT"),
        ("Salt(PTB)", "FLOAT"),
        ("KV40(cSt)", "FLOAT"),
        ("BSW(%vol)", "FLOAT"),
        ("Fiterable Solids(%wt)", "FLOAT"),
        ("McR(%wt)", "FLOAT"),
        ("Sulphur(% by mass)", "FLOAT"),
        ("Sulphur(%wt)", "FLOAT")
    ],

    "sour_water_icv112": [
        # ("ID", "INT PRIMARY KEY IDENTITY(1,1)"),
        ("Date", "NVARCHAR(255)"),
        ("Remark", "NVARCHAR(MAX)"), # MAX allows for long notes
        ("pH", "FLOAT"),
        ("Iron(mg/L)", "FLOAT"),
        ("Chloride(ppm)", "FLOAT"),
        ("Sulphides(mg/L)", "FLOAT"),
        ("Ammonia(ppm)", "FLOAT"),
        ("Oil", "FLOAT"),
        ("Free Oil(%vol)", "FLOAT")
    ],

    "sour_water_icv113": [
        # ("ID", "INT PRIMARY KEY IDENTITY(1,1)"),
        ("Date", "NVARCHAR(255)"),
        ("Remark", "NVARCHAR(MAX)"),
        ("pH", "FLOAT"),
        ("Iron(mg/L)", "FLOAT"),
        ("Fe(ppm)", "FLOAT"),
        ("Chloride(ppm)", "FLOAT"),
        ("Sulphides(mg/L)", "FLOAT"),
        ("Ammonia(ppm)", "FLOAT"),
        ("NH3(Mol ppm)", "FLOAT"),
    ],

    "stripped_water": [
        # ("ID", "INT PRIMARY KEY IDENTITY(1,1)"),
        ("Date", "NVARCHAR(255)"),
        ("Remark", "NVARCHAR(MAX)"),
        ("pH", "FLOAT"),
        ("Iron(mg/L)", "FLOAT"),
        ("Chloride content(mg/L)", "FLOAT"),
    ],

    "crude_blend_properties": [
        ("Date", "NVARCHAR(10)"),
        ("DENSITY(g/mL)", "FLOAT"),
        ("API", "FLOAT"),
        ("SULPHUR%", "FLOAT"),
        ("VR%", "FLOAT"),
        ("Molecular weight(g/mol)", "FLOAT"),
        ("Specific heat(J/kg·K)", "FLOAT"),
        ("Thermal conductivity(W/m·K)", "FLOAT"),
        ("Viscosity(Pa·s)", "FLOAT"),
        ("Specific gravity", "FLOAT"),
        
    ],

    "icv113_cr_contributor": [

        ("Date", "NVARCHAR(255)"),
        ("crude temperature(k)", "FLOAT"),
        ("mw(g/gmol)", "FLOAT"),
        ("k (w/m-k)", "FLOAT"),
        ("density(kg/m3)", "FLOAT"),
        ("cp (j/kg-k)", "FLOAT"),
        ("viscosity (pa-s)", "FLOAT"),
        ("sulfur mf", "FLOAT"),
        ("h+ mf", "FLOAT"),
        ("temperature overhead drum (k)", "FLOAT"),
        ("flow rate crude inlet(kg/s)", "FLOAT"),
        ("flow rate ww inlet(kg/s)", "FLOAT"),
        ("flow rate outlet 1(kg/s)", "FLOAT"),
        ("flow rate outlet 2(kg/s)", "FLOAT"),
        ("crude inlet mf", "FLOAT")
    ],

    "icv112_cr_contributor": [
            ("Date", "NVARCHAR(255)"),
            ("Flow rate at inlet (kg/s)", "FLOAT"),
            ("Inlet split ratio", "FLOAT"),
            ("Crude temperature(K)", "FLOAT"),
            ("Split ratio outlet 1", 'FLOAT'),
            ("Split ratio outlet 2", 'FLOAT'),
            ("MW(g/gmol)", "FLOAT"),
            ("k (W/m-k)", "FLOAT"),
            ("Density(kg/m3)", "FLOAT"),
            ("Cp (J/kg-K)", "FLOAT"),
            ("Viscosity (Pa-s)", "FLOAT"),
            ("Sulfur", "FLOAT"),
            ("H+", "FLOAT"),
            ("Temperature Overhead drum (K)", "FLOAT"),
            ("Flow rate crude inlet(kg/s)", "FLOAT"),
            ("Flow rate ww inlet(kg/s)", "FLOAT"),
            ("Flow rate outlet 1(kg/s)", "FLOAT"),
            ("Flow rate outlet 2(kg/s)", "FLOAT"),
            ("Flow rate outlet 3(kg/s)", "FLOAT"),
            ("crude inlet mf", "FLOAT")
        ],


    "102_to_161_cr_contributor":[
        ("Date", "NVARCHAR(255)"),
        ("DENSITY",  "FLOAT"),
        ("API",  "FLOAT"),
        ("Sulphur",  "FLOAT"),
        ("VR%",  "FLOAT"),
        ("Cp",  "FLOAT"),
        ("Viscosity",  "FLOAT"),
        ("Molecular Weight",  "FLOAT"),
        ("Thermal Conductivity",  "FLOAT")

    ],

    "101_to_102_cr_contributor":[
        ("Date", "NVARCHAR(255)"),
        ("DENSITY",  "FLOAT"),
        ("API",  "FLOAT"),
        ("Sulphur",  "FLOAT"),
        ("VR%",  "FLOAT"),
        ("Cp",  "FLOAT"),
        ("Viscosity",  "FLOAT"),
        ("Molecular Weight",  "FLOAT"),
        ("Thermal Conductivity",  "FLOAT")
    ],


    "112_to_162_cr_contributor": [
            ("Date", "NVARCHAR(255)"),
            ("DENSITY",  "FLOAT"),
            ("API",  "FLOAT"),
            ("Sulphur",  "FLOAT"),
            ("VR%",  "FLOAT"),
            ("Cp",  "FLOAT"),
            ("Viscosity",  "FLOAT"),
            ("Molecular Weight",  "FLOAT"),
            ("Thermal Conductivity",  "FLOAT")
    ],

    "126_to_113_cr_contributor":[
            ("Date", "NVARCHAR(255)"),
            ("DENSITY",  "FLOAT"),
            ("API",  "FLOAT"),
            ("Sulphur",  "FLOAT"),
            ("VR%",  "FLOAT"),
            ("Cp",  "FLOAT"),
            ("Viscosity",  "FLOAT"),
            ("Molecular Weight",  "FLOAT"),
            ("Thermal Conductivity",  "FLOAT")

    ],

    "161_to_112_cr_contributor":[
            ("Date", "NVARCHAR(255)"),
            ("DENSITY",  "FLOAT"),
            ("API",  "FLOAT"),
            ("Sulphur",  "FLOAT"),
            ("VR%",  "FLOAT"),
            ("Cp",  "FLOAT"),
            ("Viscosity",  "FLOAT"),
            ("Molecular Weight",  "FLOAT"),
            ("Thermal Conductivity",  "FLOAT")
    ],

    "162_to_126_cr_contributor": [
        ("Date", "NVARCHAR(255)"),
        ("DENSITY",  "FLOAT"),
        ("API",  "FLOAT"),
        ("Sulphur",  "FLOAT"),
        ("VR%",  "FLOAT"),
        ("Cp",  "FLOAT"),
        ("Viscosity",  "FLOAT"),
        ("Molecular Weight",  "FLOAT"),
        ("Thermal Conductivity",  "FLOAT")
                    
    ]


}