"""
Mappings used during Lab database synchronization.
"""

# ============================================================
# Sample -> Local Table
# ============================================================

SAMPLE_TABLE_MAPPING = {

    "cr_icv110a": "after_desalter_stage_1",

    "cr_icv110b": "after_desalter_stage_2",

    "cr_bf_cdu3": "crude_before_desalter",

    "cd3_icv112": "sour_water_icv112",

    "cd3_icv113": "sour_water_icv113",

    "str_w_cdu3": "stripped_water",

}


# ============================================================
# Client Column -> Local Database Column
# ============================================================

COLUMN_MAPPING = {

    "sampledate": "Date",

    "sample date": "Date",

    "salt (ptb)": "Salt(PTB)",

    "bsw (%vol)": "BSW(%vol)",

    "density (g/ml)": "Density(g/ml)",

    "chloride (ppm)": "Chloride(ppm)",

    "chloride content (mg/l)": "Chloride content(mg/L)",

    "iron (mg/l)": "Iron(mg/L)",

    "ph": "pH",

}