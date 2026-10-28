import numpy as np
import matplotlib.pyplot as plt
import re

# ==================================================
# S3: NMOS OUTPUT CHARACTERISTICS
# ==================================================

FILENAME = "S3_FINALLLLL FOR PYTHON.txt"

W_UM = 1.0
TARGET_VGS = 1.5
TARGET_VDS = 1.1
WINDOW = 5

# ==================================================
# 1. READ LTSPICE DATA
# ==================================================

curves = {}
current_vgs = None

multipliers = {
    "": 1.0,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9
}

with open(FILENAME, "r", encoding="utf-8-sig", errors="ignore") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        # Detect the gate-voltage step
        if line.lower().startswith("step information"):
            match = re.search(
                r"V1\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(meg|m|u|n|k|g)?",
                line,
                re.IGNORECASE
            )

            if not match:
                raise ValueError(
                    f"Cannot extract VGS from:\n{line}"
                )

            value = float(match.group(1))
            suffix = (match.group(2) or "").lower()

            current_vgs = value * multipliers[suffix]
            curves[current_vgs] = []
            continue

        # Read numeric rows
        try:
            values = [float(x) for x in line.split()]
        except ValueError:
            continue

        # Column 1: swept drain voltage
        # Column 5: drain current
        if current_vgs is not None and len(values) >= 5:
            vd = values[0]
            id_current = abs(values[4])
            curves[current_vgs].append((vd, id_current))

# ==================================================
# 2. VALIDATE DATA
# ==================================================

curves = {
    vgs: points
    for vgs, points in curves.items()
    if len(points) > 1
}

if not curves:
    raise ValueError(
        "No usable data found. Check the filename and export format."
    )

print("VGS values found:", sorted(curves.keys()))

# ==================================================
# 3. PLOT ID VS VDS
# ==================================================

plt.figure(figsize=(9, 6))

for vgs in sorted(curves):
    data = np.asarray(curves[vgs])
    order = np.argsort(data[:, 0])

    vd = data[order, 0]
    id_current = data[order, 1]

    plt.plot(
        vd,
        id_current * 1e3,
        label=f"VGS = {vgs:g} V"
    )

plt.xlabel("Drain Voltage VDS (V)")
plt.ylabel("Drain Current ID (mA)")
plt.title("S3: NMOS Output Characteristics")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S3_output_characteristics.png", dpi=300)
plt.show()

# ==================================================
# 4. PLOT GDS VS VDS
# ==================================================

plt.figure(figsize=(9, 6))

for vgs in sorted(curves):
    data = np.asarray(curves[vgs])
    order = np.argsort(data[:, 0])

    vd = data[order, 0]
    id_current = data[order, 1]

    # gds = dID/dVDS
    gds = np.gradient(id_current, vd)

    plt.plot(
        vd,
        gds * 1e3,
        label=f"VGS = {vgs:g} V"
    )

plt.xlabel("Drain Voltage VDS (V)")
plt.ylabel("Drain Conductance gds (mS)")
plt.title("S3: Drain Conductance vs Drain Voltage")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S3_gds_vs_vds.png", dpi=300)
plt.show()

# ==================================================
# 5. PLOT RO VS VDS
# ==================================================

plt.figure(figsize=(9, 6))

for vgs in sorted(curves):
    data = np.asarray(curves[vgs])
    order = np.argsort(data[:, 0])

    vd = data[order, 0]
    id_current = data[order, 1]

    gds = np.gradient(id_current, vd)

    ro_kohm = np.full_like(gds, np.nan)
    valid = gds > 0
    ro_kohm[valid] = 1 / gds[valid] / 1000

    plt.plot(
        vd,
        ro_kohm,
        label=f"VGS = {vgs:g} V"
    )

plt.xlabel("Drain Voltage VDS (V)")
plt.ylabel("Output Resistance ro (kΩ)")
plt.title("S3: Output Resistance vs Drain Voltage")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S3_ro_vs_vds.png", dpi=300)
plt.show()

# ==================================================
# 6. EXTRACT GDS AND RO AT HIGHEST VGS
# ==================================================

if TARGET_VGS not in curves:
    raise ValueError(
        f"VGS = {TARGET_VGS} V was not found in the file."
    )

data = np.asarray(curves[TARGET_VGS])
order = np.argsort(data[:, 0])

vd = data[order, 0]
id_current = data[order, 1]

# Find VDS nearest 1.0 V
index = np.argmin(np.abs(vd - TARGET_VDS))

# Local window around the extraction point
start = max(0, index - WINDOW)
end = min(len(vd), index + WINDOW + 1)

if end - start < 3:
    raise ValueError("Not enough points for local linear fitting.")

# Linear fit: ID = gds*VDS + intercept
gds, intercept = np.polyfit(
    vd[start:end],
    id_current[start:end],
    1
)

# ==================================================
# 7. PRINT FINAL EXTRACTION
# ==================================================

print("\n========================================")
print("S3 FINAL EXTRACTION")
print("========================================")

print(f"Highest VGS = {TARGET_VGS:.1f} V")
print(f"VDS         = {vd[index]:.4f} V")
print(f"ID          = {id_current[index]:.6e} A")

if gds > 0:
    ro_ohms = 1 / gds

    # Normalize resistance by transistor width
    ro_kohm_um = ro_ohms * W_UM / 1000

    print(f"gds         = {gds:.6e} S")
    print(f"ro          = {ro_ohms / 1000:.6f} kΩ")
    print(f"ro × W      = {ro_kohm_um:.6f} kΩ·µm")
else:
    print("The fitted conductance is non-positive.")
    print("Check the data and the selected saturation region.")

# ==================================================
# 8. GENERATED FILES
# ==================================================

print("\nGenerated graphs:")
print("1. S3_output_characteristics.png")
print("2. S3_gds_vs_vds.png")
print("3. S3_ro_vs_vds.png")