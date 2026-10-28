import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# S4: PHYSICS CHECK
# Compare the original model with:
# 1. UA = UB = 0
# 2. VSAT multiplied by 100
# ==================================================

FILES = {
    "Default model": {
        0.05: "S4_DEFAULT AT 0.05.txt",
        1.1:  "S4_DEFAULT AT 1.1.txt"
    },
    "UA = UB = 0": {
        0.05: "S4_UAUB FOR 0.05.txt",
        1.1:  "S4_UAUBREM AT 1.1.txt"
    },
    "VSAT x 100": {
        0.05: "S4_VSAT FOR 0.05.txt",
        1.1:  "S4_VSAT AT 1.1.txt"
    }
}


def read_ltspice(filename):
    rows = []

    with open(filename, "r", encoding="utf-8-sig",
              errors="ignore") as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            # Skip LTspice step-information lines
            if line.lower().startswith("step information"):
                continue

            try:
                values = [float(x) for x in line.split()]
                rows.append(values)
            except ValueError:
                continue

    if not rows:
        raise ValueError(
            f"No numeric data found in '{filename}'. "
            "Check the filename and exported data."
        )

    # Keep rows with a consistent number of columns
    ncols = max(len(row) for row in rows)
    rows = [row for row in rows if len(row) == ncols]

    data = np.asarray(rows, dtype=float)

    # Standard LTspice export:
    # Column 1 = swept gate voltage
    # Column 5 = drain current
    #
    # If your export has a different format,
    # these column indices must be adjusted.

    if data.shape[1] < 5:
        raise ValueError(
            f"'{filename}' has only {data.shape[1]} columns. "
            "Check the exported file format."
        )

    vg = data[:, 0]
    id_current = np.abs(data[:, 4])

    order = np.argsort(vg)

    return vg[order], id_current[order]


# ==================================================
# LOAD ALL THREE CASES AT BOTH DRAIN VOLTAGES
# ==================================================

curves = {}

for case, voltage_files in FILES.items():

    for vds, filename in voltage_files.items():

        vg, current = read_ltspice(filename)

        curves[(case, vds)] = (vg, current)

        print(
            f"Loaded: {filename} | "
            f"VDS = {vds} V | "
            f"Points = {len(vg)}"
        )


# ==================================================
# FUNCTION TO PLOT THE THREE CASES
# ==================================================

def plot_comparison(vds, logarithmic=False):

    plt.figure(figsize=(9, 6))

    for case in FILES:

        vg, current = curves[(case, vds)]

        if logarithmic:
            plt.semilogy(
                vg,
                np.maximum(current, 1e-15),
                label=case
            )
        else:
            plt.plot(
                vg,
                current * 1e3,
                label=case
            )

    plt.xlabel("Gate Voltage VGS (V)")

    if logarithmic:
        plt.ylabel("Drain Current ID (A)")
        scale = "Logarithmic"
    else:
        plt.ylabel("Drain Current ID (mA)")
        scale = "Linear"

    plt.title(
        f"S4: Physics Check | VDS = {vds} V | {scale} Scale"
    )

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    if vds == 0.05 and not logarithmic:
        filename = "S4_VDS_005_Linear.png"

    elif vds == 0.05 and logarithmic:
        filename = "S4_VDS_005_Log.png"

    elif vds == 1.1 and not logarithmic:
        filename = "S4_VDS_11_Linear.png"

    else:
        filename = "S4_VDS_11_Log.png"

    plt.savefig(filename, dpi=300)
    plt.show()

    print("Saved:", filename)


# ==================================================
# GENERATE ALL FOUR GRAPHS
# ==================================================

# VDS = 0.05 V
plot_comparison(0.05, logarithmic=False)
plot_comparison(0.05, logarithmic=True)

# VDS = 1.1 V
plot_comparison(1.1, logarithmic=False)
plot_comparison(1.1, logarithmic=True)


# ==================================================
# COMPARE ON-CURRENT AT THE HIGHEST VGS
# ==================================================

print("\n========== S4 CURRENT COMPARISON ==========")

for vds in [0.05, 1.1]:

    print(f"\nVDS = {vds} V")

    for case in FILES:

        vg, current = curves[(case, vds)]

        # Current at the highest available gate voltage
        index = np.argmax(vg)

        print(
            f"{case:20s} | "
            f"VGS = {vg[index]:.4f} V | "
            f"ID = {current[index]:.6e} A"
        )


print("\n========== S4 COMPLETE ==========")
print("Generated four comparison graphs.")