# Wall Shear Error Distribution Plot Script
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# INPUT DIRECTORY
# =========================================================
REPORT_DIR = r"C:\Users\intel1\Desktop\copy_cdu\experiments\162AirFinCooler\wall-shear\hybrid_wallshear_reports\loso_reports"


print("Folder Exists:", os.path.exists(REPORT_DIR))

if os.path.exists(REPORT_DIR):
    print("\nFolder Contents:")
    for f in os.listdir(REPORT_DIR):
        print(f)
# =========================================================
# SETTINGS
# =========================================================
ERROR_COLUMN = "pct_error"
CASE_COLUMN = None   # Auto detect from filename

# Error bins
bins = [0, 10, 25, 50, 100, 200, 500, np.inf]
labels = [
    "0-10%",
    "10-25%",
    "25-50%",
    "50-100%",
    "100-200%",
    "200-500%",
    ">500%"
]

# =========================================================
# LOAD FILES
# =========================================================
csv_files = sorted(glob.glob(os.path.join(REPORT_DIR, "*.csv")))

# =========================================================
# SELECT TOP 5 BEST PERFORMING FILES
# =========================================================

# Score and select top 5 CSV files
file_scores = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        if ERROR_COLUMN not in df_temp.columns:
            continue
        errors_temp = pd.to_numeric(df_temp[ERROR_COLUMN], errors='coerce')
        errors_temp = errors_temp.replace([np.inf, -np.inf], np.nan)
        errors_temp = errors_temp.dropna()
        if len(errors_temp) == 0:
            continue
        score = (errors_temp < 10).mean() * 100
        file_scores.append((file, score))
    except Exception as e:
        print(f"Skipping {file}: {e}")

file_scores = sorted(file_scores, key=lambda x: x[1], reverse=True)
csv_files = [x[0] for x in file_scores[:5]]

print("\nTop 5 Performing Files:\n")
for rank, (f, s) in enumerate(file_scores[:5], start=1):
    print(f"{rank}. {os.path.basename(f)} --> {s:.2f}% points below 10% error")

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in: {REPORT_DIR}")

print(f"Found {len(csv_files)} CSV files")

# =========================================================
# PLOT SETTINGS
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

all_percentages = []
all_counts = []

# =========================================================
# PROCESS EACH FILE
# =========================================================

for idx, file in enumerate(csv_files[:5]):
    df = pd.read_csv(file)

    if ERROR_COLUMN not in df.columns:
        print(f"Skipping {file} -> '{ERROR_COLUMN}' not found")
        continue

    # Remove invalid values
    error_values = pd.to_numeric(df[ERROR_COLUMN], errors='coerce')
    error_values = error_values.replace([np.inf, -np.inf], np.nan)
    error_values = error_values.dropna()

    total_points = len(error_values)

    # Histogram binning
    counts, _ = np.histogram(error_values, bins=bins)
    percentages = (counts / total_points) * 100

    all_percentages.append(percentages)
    all_counts.append(counts)

    # Case name from filename
    case_name = os.path.splitext(os.path.basename(file))[0]

    ax = axes[idx]

    # Colors
    colors = []
    for i in range(len(percentages)):
        if i == 0:
            colors.append('#38c56b')
        elif i == 1:
            colors.append('#2cad5d')
        elif i == 2:
            colors.append('#e6b422')
        else:
            colors.append('#d9534f')

    bars = ax.bar(labels, percentages, color=colors)

    # Labels on bars
    for bar, pct, count in zip(bars, percentages, counts):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{pct:.1f}%\n({count})",
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Metrics
    below_50 = (error_values < 50).mean() * 100
    below_100 = (error_values < 100).mean() * 100

    ax.text(
        0.98,
        0.98,
        f"<50% err: {below_50:.1f}%\n<100% err: {below_100:.1f}%",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=11
    )

    ax.set_title(case_name, fontsize=15)
    ax.set_xlabel("Percentage Error Range", fontsize=11)
    ax.set_ylabel("% of Points", fontsize=11)

    ax.tick_params(axis='x', rotation=30)
    ax.set_ylim(0, max(percentages) + 10)

# =========================================================
# OVERALL AVERAGE PLOT
# =========================================================
if len(all_percentages) > 0:

    avg_percentages = np.mean(all_percentages, axis=0)
    avg_counts = np.mean(all_counts, axis=0).astype(int)

    ax = axes[5]

    colors = []
    for i in range(len(avg_percentages)):
        if i == 0:
            colors.append('#38c56b')
        elif i == 1:
            colors.append('#2cad5d')
        elif i == 2:
            colors.append('#e6b422')
        else:
            colors.append('#d9534f')

    bars = ax.bar(labels, avg_percentages, color=colors)

    for bar, pct, count in zip(bars, avg_percentages, avg_counts):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{pct:.1f}%\n({count})",
            ha='center',
            va='bottom',
            fontsize=10
        )

    overall_below_50 = np.sum(avg_percentages[:3])
    overall_below_100 = np.sum(avg_percentages[:4])

    ax.text(
        0.98,
        0.98,
        f"<50% err: {overall_below_50:.1f}%\n<100% err: {overall_below_100:.1f}%",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=11
    )

    ax.set_title("ALL UNSEEN (Average)", fontsize=15)
    ax.set_xlabel("Percentage Error Range", fontsize=11)
    ax.set_ylabel("% of Points", fontsize=11)

    ax.tick_params(axis='x', rotation=30)
    ax.set_ylim(0, max(avg_percentages) + 10)

# =========================================================
# MAIN TITLE
# =========================================================
plt.suptitle(
    "Error Distribution — 5 Truly Unseen Simulations (Wall Shear Model)",
    fontsize=22,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# =========================================================
# SAVE
# =========================================================
out_path = os.path.join(REPORT_DIR, "wall_shear_error_distribution.png")

plt.savefig(out_path, dpi=300, bbox_inches='tight')

print(f"Saved plot to: {out_path}")

plt.show()
