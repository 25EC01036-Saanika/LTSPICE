
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# S5 LTspice Assignment
# 14 experiments, 3 subplots per figure
#
# 1. ID-VG Linear: VDS = 0.4 V
# 2. ID-VG Log:    VDS = 0.4 V
# 3. ID-VD Output: VGS = 1.1 V
#
# VTH MID is used as the shared transfer reference.
# All figures are displayed AND saved as PNG files.
# ============================================================

FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = FOLDER / "S5_Graphs"
OUTPUT_FOLDER.mkdir(exist_ok=True)

PARAMETERS = {
    1: ("VTH", "VTH"),
    2: ("TOXE", "TOXE"),
    3: ("L", "L"),
    4: ("XL", "XL"),
    5: ("LINT", "LINT"),
    6: ("W", "W"),
    7: ("WINT", "WINT"),
    8: ("NDEP", "NDEP"),
    9: ("NSD", "NSD"),
    10: ("XJ", "XJ"),
    11: ("NGATE", "NGATE"),
    12: ("U0", "U0"),
    13: ("VSAT", "VSAT"),
    14: ("RDSW", "RDSW"),
}

# Filename tuple: (VDS = 0.4 V, VDS = 1.1 V)
# This script uses the 0.4 V files for transfer plots.
FILES = {
    "VTH": {
        "Low": ("S5_VTHlow AT 0.4.txt", "S5_VTHLOW AT 1.1.txt"),
        "High": ("S5_VTHHIGH AT 0.4.txt", "S5_VTHHIGH AT 1.1.txt"),
    },
    "TOXE": {
        "Low": ("S5_TOXELOW AT 0,4.txt", "S5_TOXELOW AT 1.1.txt"),
        "High": ("S5_TOXEHIGH AT 0.4.txt", "S5_TOXEHIGH AT 1.1.txt"),
    },
    "L": {
        "Low": ("S5_LLOW AT 0.4.txt", "S5_LLOW AT 1.1.txt"),
        "High": ("S5_LHIGH AT 0.4.txt", "S5_LHIGH AT 1.1.txt"),
    },
    "XL": {
        "Low": ("S5_XLLOW AT 0.4.txt", "S5_XLLOW AT 1.1.txt"),
        "High": ("S5_XLHIGH AT 0.4.txt", "S5_XLHIGH AT 1.1.txt"),
    },
    "LINT": {
        "Low": ("S5_LINTLOW AT 0.4.txt", "S5_LINTLOW AT 1.1.txt"),
        "High": ("S5_LINTHIGH AT 0.4.txt", "S5_LINTHIGH AT 1.1.txt"),
    },
    "W": {
        "Low": ("S5_WLOW AT 0.4.txt", "S5_WLOW AT 1.1.txt"),
        "High": ("S5_WHIGH AT 0.4.txt", "S5_WHIGH AT 1.1.txt"),
    },
    "WINT": {
        "Low": ("S5_WINTLOW AT 0.4.txt", "S5_WINTLOW AT 1.1.txt"),
        "High": ("S5_WINTHIGH AT 0.4.txt", "S5_WINTHIGH AT 1.1.txt"),
    },
    "NDEP": {
        "Low": ("S5_NDEPlow at 0.4.txt", "S5_NDEPLOW AT 1.1.txt"),
        "High": ("S5_NDEPHIGH AT 0.4.txt", "S5_NDEPHIGH AT 1.1.txt"),
    },
    "NSD": {
        "Low": ("S5_NSDLOW AT 0.4.txt", "S5_NSDLOW AT 1.1.txt"),
        "High": ("S5_NSDHIGH AT 0.4.txt", "S5_NSDHIGH AT 1.1.txt"),
    },
    "XJ": {
        "Low": ("S5_XJLOW AT 0.4.txt", "S5_XJLOW AT 1.1.txt"),
        "High": ("S5_XJHIGH AT 0.4.txt", "S5_XJHIGH AT 1.1.txt"),
    },
    "NGATE": {
        "Low": ("S5_NGATELOW AT 0.4.txt", "S5_NGATELOW AT 1.1.txt"),
        "High": ("S5_NGATEHIGH AT 0.4.txt", "S5_NGATEHIGH AT 1.1.txt"),
    },
    "U0": {
        "Low": ("S5_U0LOW AT 0.4.txt", "S5_U0LOW AT 1.1.txt"),
        "High": ("S5_U0HIGH AT 0.4.txt", "S5_U0HIGH AT 1.1.txt"),
    },
    "VSAT": {
        "Low": ("S5_VSATLOW AT 0.4.txt", "S5_VSATLOW AT 1.1.txt"),
        "High": ("S5_VSATHIGH AT 0.4.txt", "S5_VSATHIGH AT 1.1.txt"),
    },
    "RDSW": {
        "Low": ("S5_RDSWLOW AT 0.4.txt", "S5_RDSWLOW AT 1.1.txt"),
        "High": ("S5_RDSWHIGH AT 0.4.txt", "S5_RDSWHIGH AT 1.1.txt"),
    },
}

MID_REFERENCE = "S5_VTH MID at 0.4.txt"

# -------------------- Find files --------------------

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

def find_file(filename):
    if not filename:
        return None

    target = normalize(filename)

    for path in FOLDER.rglob("*"):
        if path.is_file() and path.name.lower() == filename.lower():
            return path

    for path in FOLDER.rglob("*"):
        if path.is_file() and normalize(path.name) == target:
            return path

    return None

# -------------------- Read transfer data --------------------

def read_transfer(path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]

    gate_col = next(
        (c for c in df.columns if c in ("gate", "vgs")),
        None
    )
    current_col = next(
        (c for c in df.columns
         if c in ("id(m2)", "idrain", "i(drain)", "id")),
        None
    )

    if gate_col is None or current_col is None:
        raise ValueError(
            f"Cannot identify columns in {path.name}: {list(df.columns)}"
        )

    gate = pd.to_numeric(df[gate_col], errors="coerce").to_numpy()
    current = pd.to_numeric(df[current_col], errors="coerce").to_numpy()

    valid = np.isfinite(gate) & np.isfinite(current)
    return gate[valid], np.abs(current[valid]) * 1e6

# -------------------- Plot ID-VG transfer curves --------------------

def plot_transfer(ax_linear, ax_log, parameter):
    for case in ("Low", "High"):
        filenames = FILES.get(parameter, {}).get(case)

        if not filenames:
            continue

        path = find_file(filenames[0])  # VDS = 0.4 V

        if path is None:
            print(f"Missing transfer file: {filenames[0]}")
            continue

        try:
            vg, current = read_transfer(path)
            ax_linear.plot(vg, current, label=case)

            positive = current > 0
            ax_log.plot(vg[positive], current[positive], label=case)

        except Exception as exc:
            print(f"Error reading {path.name}: {exc}")

    # Shared VTH MID reference, as requested.
    mid_path = find_file(MID_REFERENCE)

    if mid_path is not None:
        try:
            vg, current = read_transfer(mid_path)

            ax_linear.plot(
                vg, current, "--", label="VTH MID reference"
            )

            positive = current > 0
            ax_log.plot(
                vg[positive], current[positive],
                "--", label="VTH MID reference"
            )
        except Exception as exc:
            print(f"Error reading MID reference: {exc}")
    else:
        print(f"Missing shared reference: {MID_REFERENCE}")

    ax_linear.set_title("ID-VG Linear (VDS = 0.4 V)")
    ax_linear.set_xlabel("Gate Voltage VGS (V)")
    ax_linear.set_ylabel("Drain Current |ID| (µA)")
    ax_linear.grid(True, alpha=0.3)
    ax_linear.legend(fontsize=7)

    ax_log.set_title("ID-VG Log (VDS = 0.4 V)")
    ax_log.set_xlabel("Gate Voltage VGS (V)")
    ax_log.set_ylabel("Drain Current |ID| (µA)")
    ax_log.set_yscale("log")
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.legend(fontsize=7)

# -------------------- Read output sweeps --------------------

def read_output(path):
    curves = []
    label = "Output"
    points = []

    def finish_curve():
        if points:
            arr = np.asarray(points, dtype=float)
            curves.append(
                (label, arr[:, 0], np.abs(arr[:, 1]) * 1e6)
            )

    with open(path, "r", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.lower().startswith("step information"):
                finish_curve()
                points.clear()
                label = line.split(":", 1)[-1].strip()
                continue

            parts = re.split(r"\s+", line)

            try:
                vd = float(parts[0])
                current = float(parts[1])
                points.append((vd, current))
            except (ValueError, IndexError):
                continue

    finish_curve()
    return curves

# -------------------- Plot ID-VD output curves --------------------

def plot_output(ax, number):
    path = find_file(f"S5_{number}_vd.txt")

    if path is None:
        ax.text(
            0.5, 0.5,
            f"Missing S5_{number}_vd.txt",
            ha="center", va="center",
            transform=ax.transAxes
        )
    else:
        try:
            curves = read_output(path)

            for label, vd, current in curves:
                ax.plot(vd, current, label=label)

            if curves:
                ax.legend(fontsize=7)

        except Exception as exc:
            ax.text(
                0.5, 0.5, f"Error reading output:\n{exc}",
                ha="center", va="center",
                transform=ax.transAxes
            )

    # Special scale for Experiment 9 only.
    if number == 9:
        ax.set_ylim(0, 600)
        ax.set_yticks(np.arange(0, 601, 50))
        ax.set_xlim(0, 1.1)

    ax.set_title("ID-VD Output (VGS = 1.1 V)")
    ax.set_xlabel("Drain Voltage VDS (V)")
    ax.set_ylabel("Drain Current |ID| (µA)")
    ax.grid(True, alpha=0.3)

# -------------------- Main loop --------------------

for number, (parameter, display_name) in PARAMETERS.items():
    print(f"Displaying Experiment {number}: {display_name}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # Exactly three subplots per experiment.
    plot_transfer(axes[0], axes[1], parameter)
    plot_output(axes[2], number)

    fig.suptitle(
        f"S5 Experiment {number}: {display_name} Variation",
        fontsize=14
    )

    fig.tight_layout()

    # Save with descriptive names.
    filename = (
        f"S5_experiment_{number}_{display_name}_Variation.png"
    )
    save_path = OUTPUT_FOLDER / filename

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # Display the figure too.
    plt.show()
    plt.close(fig)

print(f"\nAll graphs saved in: {OUTPUT_FOLDER}")