import pandas as pd
import numpy as np
import os


# ============================================================
# NOZZLE CSV FILES
# ============================================================

nozzle_files = [

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\Nozzle_1_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\Nozzle_2_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\Nozzle_3_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\Nozzle_4_Red_Strip.csv"

]


# ============================================================
# ICV DATASET
# ============================================================

icv_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works"
    r"\3D visualization\All new\SSTL\STL file\112"
    r"\ICV_112_CASE_1_with_r_theta_phi_unseen_report.csv"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_folder = os.path.dirname(nozzle_files[0])


# ============================================================
# MAXIMUM ALLOWED DISTANCE
# ============================================================
#
# Distance is in metres because your coordinates are in metres.
#
# 0.001 m = 1 mm
#
# Start with 1 mm.
#
# ============================================================

MAX_DISTANCE = 0.001


# ============================================================
# LOAD ICV DATASET
# ============================================================

print("\n==================================================")
print("LOADING ICV DATASET")
print("==================================================")

icv_df = pd.read_csv(icv_file)

icv_df.columns = icv_df.columns.str.strip()

print("ICV dataset loaded.")
print("Rows:", len(icv_df))

print("\nICV columns:")
print(icv_df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "x-coordinate",
    "y-coordinate",
    "z-coordinate",
    "predicted"
]

for column in required_columns:

    if column not in icv_df.columns:

        raise ValueError(
            f"Column '{column}' was not found in ICV dataset."
        )


# ============================================================
# CONVERT ICV DATA TO NUMERIC
# ============================================================

icv_df["x-coordinate"] = pd.to_numeric(
    icv_df["x-coordinate"],
    errors="coerce"
)

icv_df["y-coordinate"] = pd.to_numeric(
    icv_df["y-coordinate"],
    errors="coerce"
)

icv_df["z-coordinate"] = pd.to_numeric(
    icv_df["z-coordinate"],
    errors="coerce"
)

icv_df["predicted"] = pd.to_numeric(
    icv_df["predicted"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

icv_df = icv_df.dropna(
    subset=[
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "predicted"
    ]
).reset_index(drop=True)


print(
    "\nValid ICV rows:",
    len(icv_df)
)


# ============================================================
# CREATE NUMPY ARRAY
# ============================================================

icv_coordinates = icv_df[
    [
        "x-coordinate",
        "y-coordinate",
        "z-coordinate"
    ]
].to_numpy()


# ============================================================
# PROCESS EACH NOZZLE
# ============================================================

for nozzle_file in nozzle_files:

    print("\n")
    print("==================================================")
    print(
        "PROCESSING:",
        os.path.basename(nozzle_file)
    )
    print("==================================================")


    # ========================================================
    # LOAD NOZZLE CSV
    # ========================================================

    nozzle_df = pd.read_csv(nozzle_file)

    nozzle_df.columns = nozzle_df.columns.str.strip()


    # ========================================================
    # CHECK COLUMNS
    # ========================================================

    required_nozzle_columns = [
        "X[m]",
        "Y[m]",
        "Z[m]"
    ]

    for column in required_nozzle_columns:

        if column not in nozzle_df.columns:

            raise ValueError(
                f"Column '{column}' was not found in "
                f"{os.path.basename(nozzle_file)}"
            )


    # ========================================================
    # CONVERT TO NUMERIC
    # ========================================================

    nozzle_df["X[m]"] = pd.to_numeric(
        nozzle_df["X[m]"],
        errors="coerce"
    )

    nozzle_df["Y[m]"] = pd.to_numeric(
        nozzle_df["Y[m]"],
        errors="coerce"
    )

    nozzle_df["Z[m]"] = pd.to_numeric(
        nozzle_df["Z[m]"],
        errors="coerce"
    )


    # ========================================================
    # CREATE OUTPUT DATAFRAME
    # ========================================================

    result_rows = []


    # ========================================================
    # PROCESS EVERY RED STRIP COORDINATE
    # ========================================================

    for index, row in nozzle_df.iterrows():

        x = row["X[m]"]
        y = row["Y[m]"]
        z = row["Z[m]"]


        # ----------------------------------------------------
        # Skip invalid coordinates
        # ----------------------------------------------------

        if pd.isna(x) or pd.isna(y) or pd.isna(z):

            continue


        # ----------------------------------------------------
        # Current red-strip coordinate
        # ----------------------------------------------------

        nozzle_coordinate = np.array([
            x,
            y,
            z
        ])


        # ----------------------------------------------------
        # Calculate 3D distance to every ICV coordinate
        # ----------------------------------------------------

        difference = (
            icv_coordinates
            - nozzle_coordinate
        )


        distance = np.sqrt(
            np.sum(
                difference ** 2,
                axis=1
            )
        )


        # ----------------------------------------------------
        # Find nearest ICV coordinate
        # ----------------------------------------------------

        nearest_index = np.argmin(distance)

        nearest_distance = distance[nearest_index]


        # ----------------------------------------------------
        # Get matching ICV row
        # ----------------------------------------------------

        icv_row = icv_df.iloc[nearest_index]


        # ----------------------------------------------------
        # Get corrosion rate
        # ----------------------------------------------------

        corrosion_rate = icv_row["predicted"]


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result_rows.append({

            "X[m]": x,

            "Y[m]": y,

            "Z[m]": z,

            "predicted": corrosion_rate,

            "ICV_X[m]": icv_row["x-coordinate"],

            "ICV_Y[m]": icv_row["y-coordinate"],

            "ICV_Z[m]": icv_row["z-coordinate"],

            "Distance[m]": nearest_distance

        })


    # ========================================================
    # CREATE RESULT DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(result_rows)


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\nTotal red-strip coordinates:",
        len(result_df)
    )


    if len(result_df) > 0:

        print(
            "\nMinimum coordinate distance:",
            result_df["Distance[m]"].min()
        )

        print(
            "Maximum coordinate distance:",
            result_df["Distance[m]"].max()
        )

        print(
            "Average coordinate distance:",
            result_df["Distance[m]"].mean()
        )


    # ========================================================
    # CHECK MAXIMUM DISTANCE
    # ========================================================

    if len(result_df) > 0:

        outside_limit = (
            result_df["Distance[m]"]
            > MAX_DISTANCE
        ).sum()

        print(
            "\nCoordinates farther than",
            MAX_DISTANCE,
            "m:",
            outside_limit
        )


    # ========================================================
    # OUTPUT FILE NAME
    # ========================================================

    nozzle_name = os.path.basename(
        nozzle_file
    ).replace(
        "_Red_Strip.csv",
        ""
    )


    output_filename = (
        f"Corrosion_{nozzle_name}_Red_Strip.csv"
    )


    output_path = os.path.join(
        output_folder,
        output_filename
    )


    # ========================================================
    # SAVE
    # ========================================================

    result_df.to_csv(
        output_path,
        index=False
    )


    print(
        "\nOutput saved:"
    )

    print(
        output_path
    )


    # ========================================================
    # SHOW SAMPLE
    # ========================================================

    print(
        "\nFirst 10 results:"
    )

    print(
        result_df.head(10).to_string(
            index=False
        )
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("==================================================")
print("CORROSION RATE EXTRACTION COMPLETE")
print("==================================================")

print(
    "The 'predicted' corrosion rate from the nearest"
)

print(
    "ICV coordinate has been saved for every red-strip point."
)

print(
    "\nOutput folder:"
)

print(output_folder)