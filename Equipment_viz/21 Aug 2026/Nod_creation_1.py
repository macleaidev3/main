import pandas as pd
import numpy as np


# ============================================================
# 1. INPUT CSV FILE
# ============================================================

FILE_NAME = "161_to_112P_1.2.xlsx_report.csv"


# ============================================================
# 2. READ CSV
# ============================================================

df = pd.read_csv(FILE_NAME)


# ============================================================
# 3. REMOVE ROWS WITH MISSING X, Y, Z
# ============================================================

df = df.dropna(
    subset=[
        "X[m]",
        "Y[m]",
        "Z[m]"
    ]
).reset_index(drop=True)


# ============================================================
# 4. CREATE NODE NUMBERS
# ============================================================

df["Node"] = np.arange(1, len(df) + 1)


# ============================================================
# 5. CHECK cpredicted_CR COLUMN
# ============================================================

if "predicted_CR" not in df.columns:
    raise ValueError(
        "ERROR: 'predicted_CR' column was not found in the CSV file."
    )


# ============================================================
# 6. CREATE NODE + COORDINATES + CORROSION RATE TABLE
# ============================================================

node_corrosion = df[
    [
        "Node",
        "X[m]",
        "Y[m]",
        "Z[m]",
        "predicted_CR"
    ]
].copy()


# ============================================================
# 7. PRINT NODE, COORDINATES AND CORROSION RATE
# ============================================================

print("\n==============================================================")
print("NODE, COORDINATES AND CORROSION RATE")
print("==============================================================\n")

print(
    node_corrosion.to_string(index=False)
)


# ============================================================
# 8. SAVE TO CSV
# ============================================================

OUTPUT_FILE = "Node_161_to_112P_1.2.csv"

node_corrosion.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. CONFIRMATION
# ============================================================

print("\n==============================================================")
print("DONE")
print("==============================================================")

print(f"Total number of nodes: {len(node_corrosion)}")

print(f"Output file saved as: {OUTPUT_FILE}")