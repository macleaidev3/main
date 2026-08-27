import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

red_strip_files = [
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\162_dataset-12\COnverted_Nozzle_1_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\162_dataset-12\COnverted_Nozzle_2_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\162_dataset-12\COnverted_Nozzle_3_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\162_dataset-12\COnverted_Nozzle_4_Red_Strip.csv"
]


# ICV prediction file
icv_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\162_dataset-12\162_P_12.2.xlsx_report.csv"
)


# ============================================================
# LOAD ICV PREDICTION FILE
# ============================================================

print("\nLoading ICV prediction file...")

icv_df = pd.read_csv(icv_file)

print("ICV file loaded successfully.")
print("Number of rows:", len(icv_df))

print("\nICV columns:")
print(icv_df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_icv_columns = [
    "X[m]",
    "Y[m]",
    "Z[m]",
    "predicted_CR"
]

for column in required_icv_columns:
    if column not in icv_df.columns:
        raise ValueError(
            f"Column '{column}' was not found in the ICV file."
        )


# ============================================================
# CREATE COORDINATE KEY
# ============================================================
#
# We combine X, Y and Z into one key.
#
# Example:
#
# X = 1.234
# Y = 2.345
# Z = 3.456
#
# becomes:
#
# (1.234, 2.345, 3.456)
#
# ============================================================

icv_df["_coordinate_key"] = list(
    zip(
        icv_df["X[m]"],
        icv_df["Y[m]"],
        icv_df["Z[m]"]
    )
)


# Create lookup dictionary
prediction_lookup = dict(
    zip(
        icv_df["_coordinate_key"],
        icv_df["predicted_CR"]
    )
)


print("\nPrediction lookup created.")
print("Number of coordinate entries:", len(prediction_lookup))


# ============================================================
# PROCESS EACH RED STRIP FILE
# ============================================================

for red_file in red_strip_files:

    print("\n" + "=" * 70)
    print("Processing:")
    print(os.path.basename(red_file))
    print("=" * 70)

    # --------------------------------------------------------
    # Load Red Strip CSV
    # --------------------------------------------------------

    red_df = pd.read_csv(red_file)

    print("Rows in Red Strip file:", len(red_df))

    print("\nRed Strip columns:")
    print(red_df.columns.tolist())


    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_red_columns = [
        "X",
        "Y",
        "Z"
    ]

    for column in required_red_columns:

        if column not in red_df.columns:
            raise ValueError(
                f"Column '{column}' was not found in "
                f"{os.path.basename(red_file)}"
            )


    # --------------------------------------------------------
    # Create coordinate keys for Red Strip file
    # --------------------------------------------------------

    red_df["_coordinate_key"] = list(
        zip(
            red_df["X"],
            red_df["Y"],
            red_df["Z"]
        )
    )


    # --------------------------------------------------------
    # Match coordinates and retrieve predicted_CR
    # --------------------------------------------------------

    red_df["Predicted_Corrosion_Rate"] = (
        red_df["_coordinate_key"].map(prediction_lookup)
    )


    # --------------------------------------------------------
    # Count matches
    # --------------------------------------------------------

    total_rows = len(red_df)

    matched_rows = red_df[
        "Predicted_Corrosion_Rate"
    ].notna().sum()

    unmatched_rows = red_df[
        "Predicted_Corrosion_Rate"
    ].isna().sum()


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nMatching results:")
    print("Total coordinates :", total_rows)
    print("Matched coordinates:", matched_rows)
    print("Unmatched coordinates:", unmatched_rows)


    # --------------------------------------------------------
    # Show unmatched coordinates if any
    # --------------------------------------------------------

    if unmatched_rows > 0:

        print("\nWARNING: Some coordinates were not found.")

        unmatched = red_df[
            red_df["Predicted_Corrosion_Rate"].isna()
        ][["X", "Y", "Z"]]

        print("\nUnmatched coordinates:")
        print(unmatched)


    # --------------------------------------------------------
    # Remove temporary coordinate key
    # --------------------------------------------------------

    red_df.drop(
        columns=["_coordinate_key"],
        inplace=True
    )


    # --------------------------------------------------------
    # Save updated CSV
    # --------------------------------------------------------

    red_df.to_csv(
        red_file,
        index=False
    )

    print("\nUpdated file saved successfully:")
    print(red_file)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ALL FOUR RED STRIP FILES HAVE BEEN PROCESSED.")
print("=" * 70)