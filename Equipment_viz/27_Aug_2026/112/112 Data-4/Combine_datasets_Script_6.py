import pandas as pd
import numpy as np
import os

from scipy.spatial import cKDTree


# ============================================================
# FILE PATHS
# ============================================================

red_strip_files = [

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\112 Data-4\COnverted_Nozzle_1_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\112 Data-4\COnverted_Nozzle_2_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\112 Data-4\COnverted_Nozzle_3_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\112 Data-4\COnverted_Nozzle_4_Red_Strip.csv"

    # Add more Red Strip files here if required
]


# ============================================================
# ICV PREDICTION FILE
# ============================================================

icv_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\112 Data-4\ICV_112_CASE_4_with_r_theta_phi_unseen_report.csv"
)


# ============================================================
# MATCHING TOLERANCE
# ============================================================
#
# IMPORTANT:
#
# We are NOT rounding coordinates.
#
# Instead, we find the nearest ICV coordinate in 3D.
#
# MAX_DISTANCE defines the maximum allowed distance between
# a Red Strip coordinate and its nearest ICV coordinate.
#
# Start with 1e-5.
#
# 1e-5 = 0.00001
#
# If your coordinates are in metres, this means:
#
# 0.00001 m = 0.01 mm
#
# ============================================================

MAX_DISTANCE = 1e-5


# ============================================================
# LOAD ICV PREDICTION FILE
# ============================================================

print("\n" + "=" * 80)
print("LOADING ICV PREDICTION FILE")
print("=" * 80)

icv_df = pd.read_csv(icv_file)

print("ICV file loaded successfully.")
print("Number of rows:", len(icv_df))

print("\nICV columns:")
print(icv_df.columns.tolist())


# ============================================================
# CHECK REQUIRED ICV COLUMNS
# ============================================================

required_icv_columns = [
    "x-coordinate",
    "y-coordinate",
    "z-coordinate",
    "predicted"
]

for column in required_icv_columns:

    if column not in icv_df.columns:

        raise ValueError(
            f"Column '{column}' was not found in the ICV file."
        )


# ============================================================
# CONVERT ICV COORDINATES TO NUMERIC
# ============================================================

print("\nConverting ICV coordinates to numeric...")

icv_df["x-coordinate"] = pd.to_numeric(
    icv_df["x-coordinate"],
    errors="raise"
)

icv_df["y-coordinate"] = pd.to_numeric(
    icv_df["y-coordinate"],
    errors="raise"
)

icv_df["z-coordinate"] = pd.to_numeric(
    icv_df["z-coordinate"],
    errors="raise"
)

icv_df["predicted"] = pd.to_numeric(
    icv_df["predicted"],
    errors="coerce"
)


# ============================================================
# CHECK FOR INVALID ICV COORDINATES
# ============================================================

invalid_icv_coordinates = icv_df[
    [
        "x-coordinate",
        "y-coordinate",
        "z-coordinate"
    ]
].isna().any(axis=1)

if invalid_icv_coordinates.any():

    invalid_count = invalid_icv_coordinates.sum()

    raise ValueError(
        f"ICV file contains {invalid_count} rows with "
        f"invalid X/Y/Z coordinates."
    )


# ============================================================
# CREATE ICV COORDINATE ARRAY
# ============================================================
#
# IMPORTANT:
#
# NO ROUNDING IS DONE HERE.
#
# We preserve the complete floating-point precision from
# the ICV file.
#
# ============================================================

icv_coordinates = icv_df[
    [
        "x-coordinate",
        "y-coordinate",
        "z-coordinate"
    ]
].to_numpy(dtype=np.float64)


# ============================================================
# CREATE 3D KD-TREE
# ============================================================
#
# The KD-tree allows us to efficiently find the nearest
# ICV coordinate for every Red Strip coordinate.
#
# ============================================================

print("\nCreating 3D coordinate search tree...")

icv_tree = cKDTree(icv_coordinates)

print("3D coordinate search tree created successfully.")


# ============================================================
# DISPLAY ICV COORDINATE INFORMATION
# ============================================================

print("\n" + "-" * 80)
print("ICV COORDINATE INFORMATION")
print("-" * 80)

print("Number of ICV coordinates:", len(icv_coordinates))

print("\nFirst 10 ICV coordinates:")

print(
    icv_df[
        [
            "x-coordinate",
            "y-coordinate",
            "z-coordinate",
            "predicted"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# PROCESS EACH RED STRIP FILE
# ============================================================

for red_file in red_strip_files:

    print("\n\n" + "=" * 80)
    print("PROCESSING RED STRIP FILE")
    print("=" * 80)

    print("File:")
    print(os.path.basename(red_file))

    print("=" * 80)


    # ========================================================
    # LOAD RED STRIP CSV
    # ========================================================

    red_df = pd.read_csv(red_file)

    print("\nRows in Red Strip file:", len(red_df))

    print("\nRed Strip columns:")
    print(red_df.columns.tolist())


    # ========================================================
    # CHECK REQUIRED RED STRIP COLUMNS
    # ========================================================

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


    # ========================================================
    # CONVERT RED STRIP COORDINATES TO NUMERIC
    # ========================================================

    print("\nConverting Red Strip coordinates to numeric...")

    red_df["X"] = pd.to_numeric(
        red_df["X"],
        errors="raise"
    )

    red_df["Y"] = pd.to_numeric(
        red_df["Y"],
        errors="raise"
    )

    red_df["Z"] = pd.to_numeric(
        red_df["Z"],
        errors="raise"
    )


    # ========================================================
    # CHECK FOR INVALID RED STRIP COORDINATES
    # ========================================================

    invalid_red_coordinates = red_df[
        ["X", "Y", "Z"]
    ].isna().any(axis=1)

    if invalid_red_coordinates.any():

        invalid_count = invalid_red_coordinates.sum()

        raise ValueError(
            f"{os.path.basename(red_file)} contains "
            f"{invalid_count} rows with invalid X/Y/Z coordinates."
        )


    # ========================================================
    # CREATE RED STRIP COORDINATE ARRAY
    # ========================================================
    #
    # IMPORTANT:
    #
    # NO ROUNDING IS DONE.
    #
    # The original full-precision coordinates are used.
    #
    # ========================================================

    red_coordinates = red_df[
        ["X", "Y", "Z"]
    ].to_numpy(dtype=np.float64)


    # ========================================================
    # FIND NEAREST ICV COORDINATE
    # ========================================================
    #
    # For every Red Strip coordinate:
    #
    #     X,Y,Z
    #       |
    #       v
    #    KD-Tree
    #       |
    #       v
    # nearest ICV coordinate
    #
    # distances = distance to nearest ICV point
    #
    # indices = row number of nearest ICV point
    #
    # ========================================================

    print("\nFinding nearest ICV coordinates...")

    distances, indices = icv_tree.query(
        red_coordinates,
        k=1
    )


    # ========================================================
    # STORE MATCHED DISTANCE
    # ========================================================
    #
    # This is extremely useful for checking whether the
    # coordinate matching is physically sensible.
    #
    # ========================================================

    red_df["Matched_Distance"] = distances


    # ========================================================
    # DETERMINE VALID MATCHES
    # ========================================================
    #
    # Only accept the nearest ICV coordinate if the distance
    # is <= MAX_DISTANCE.
    #
    # ========================================================

    valid_matches = distances <= MAX_DISTANCE


    # ========================================================
    # CREATE CORROSION RATE COLUMN
    # ========================================================

    red_df["Predicted_Corrosion_Rate"] = np.nan


    # ========================================================
    # ASSIGN CORROSION RATE
    # ========================================================
    #
    # Only valid nearest-neighbour matches receive a
    # Predicted_Corrosion_Rate.
    #
    # ========================================================

    valid_indices = indices[valid_matches]

    red_df.loc[
        valid_matches,
        "Predicted_Corrosion_Rate"
    ] = icv_df.iloc[
        valid_indices
    ]["predicted"].to_numpy()


    # ========================================================
    # COUNT MATCHES
    # ========================================================

    total_rows = len(red_df)

    matched_rows = valid_matches.sum()

    unmatched_rows = total_rows - matched_rows


    # ========================================================
    # DISPLAY MATCHING RESULTS
    # ========================================================

    print("\n" + "-" * 80)
    print("MATCHING RESULTS")
    print("-" * 80)

    print(
        "Total Red Strip coordinates :",
        total_rows
    )

    print(
        "Matched coordinates         :",
        matched_rows
    )

    print(
        "Unmatched coordinates       :",
        unmatched_rows
    )


    if total_rows > 0:

        matching_percentage = (
            matched_rows / total_rows
        ) * 100

        print(
            "Matching percentage         :",
            f"{matching_percentage:.2f}%"
        )


    # ========================================================
    # DISTANCE STATISTICS
    # ========================================================
    #
    # These statistics help us determine whether the selected
    # MAX_DISTANCE is appropriate.
    #
    # ========================================================

    if matched_rows > 0:

        matched_distances = distances[
            valid_matches
        ]

        print("\nMatched distance statistics:")

        print(
            "Minimum distance :",
            f"{matched_distances.min():.12f}"
        )

        print(
            "Maximum distance :",
            f"{matched_distances.max():.12f}"
        )

        print(
            "Mean distance    :",
            f"{matched_distances.mean():.12f}"
        )

        print(
            "Median distance  :",
            f"{np.median(matched_distances):.12f}"
        )


    # ========================================================
    # SHOW UNMATCHED COORDINATES
    # ========================================================

    if unmatched_rows > 0:

        print("\n" + "-" * 80)
        print("WARNING: UNMATCHED COORDINATES")
        print("-" * 80)

        unmatched = red_df[
            ~valid_matches
        ][
            [
                "X",
                "Y",
                "Z",
                "Matched_Distance"
            ]
        ]

        print(
            "\nUnmatched coordinates "
            "and nearest ICV distance:"
        )

        print(
            unmatched.to_string(index=False)
        )


    # ========================================================
    # SHOW FIRST 20 SUCCESSFUL MATCHES
    # ========================================================

    print("\n" + "-" * 80)
    print("FIRST 20 COORDINATE MATCHES")
    print("-" * 80)

    preview_columns = [
        "X",
        "Y",
        "Z",
        "Matched_Distance",
        "Predicted_Corrosion_Rate"
    ]

    print(
        red_df[
            preview_columns
        ].head(20).to_string(index=False)
    )


    # ========================================================
    # SAVE UPDATED CSV
    # ========================================================
    #
    # IMPORTANT:
    #
    # We are NOT rounding X/Y/Z before matching.
    #
    # The original coordinate precision is retained in the
    # DataFrame.
    #
    # float_format controls how the numbers are written.
    #
    # Here we use 10 decimal places rather than 5 so that
    # important coordinate precision is not destroyed.
    #
    # ========================================================

    red_df.to_csv(
        red_file,
        index=False,
        float_format="%.10f"
    )


    # ========================================================
    # CONFIRM FILE SAVED
    # ========================================================

    print("\nUpdated file saved successfully:")

    print(red_file)


# ============================================================
# COMPLETED
# ============================================================

print("\n\n" + "=" * 80)
print("ALL RED STRIP FILES HAVE BEEN PROCESSED")
print("=" * 80)

print("\nMatching method:")
print("3D nearest-neighbour coordinate matching using cKDTree")

print("\nMaximum accepted coordinate distance:")
print(f"{MAX_DISTANCE}")

print("\nNo coordinate rounding was used during matching.")

print("\nProcessing completed successfully.")
print("=" * 80)