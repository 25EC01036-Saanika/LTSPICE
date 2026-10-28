import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# S2: FILES AND PARAMETERS
# =====================================================

FILE_LOW = "S2_FINALL AT 0.05.txt"
FILE_HIGH = "S2_FINALAT 1.1.txt"

VDS_LOW = 0.05
VDS_HIGH = 1.1

I_THRESHOLD = 1.5e-6  # 1.5 uA constant-current criterion


# =====================================================
# LOAD DATA
# =====================================================

def load_data(filename):
    df = pd.read_csv(filename, sep="\t")
    df.columns = df.columns.str.strip()

    df["GATE"] = pd.to_numeric(df["GATE"], errors="coerce")
    df["Id(M2)"] = pd.to_numeric(df["Id(M2)"], errors="coerce")

    df = df.dropna(subset=["GATE", "Id(M2)"])
    df = df.sort_values("GATE").reset_index(drop=True)

    return df


low = load_data(FILE_LOW)
high = load_data(FILE_HIGH)


# =====================================================
# VTH: CONSTANT-CURRENT METHOD
# =====================================================

def extract_vth_constant_current(df):
    gate = df["GATE"].to_numpy()
    current = np.abs(df["Id(M2)"].to_numpy())

    order = np.argsort(gate)
    gate = gate[order]
    current = current[order]

    crossings = np.where(
        (current[:-1] <= I_THRESHOLD) &
        (current[1:] >= I_THRESHOLD)
    )[0]

    if len(crossings) == 0:
        return np.nan

    i = crossings[0]

    return np.interp(
        I_THRESHOLD,
        current[i:i + 2],
        gate[i:i + 2]
    )


# =====================================================
# VTH: LINEAR-EXTRAPOLATION METHOD
# =====================================================

def extract_vth_linear(df):
    gate = df["GATE"].to_numpy()
    current = np.abs(df["Id(M2)"].to_numpy())

    order = np.argsort(gate)
    gate = gate[order]
    current = current[order]

    # Calculate transconductance
    gm = np.gradient(current, gate)

    # Locate maximum transconductance
    idx = np.argmax(gm)

    # Fit a straight line around the maximum-gm point
    start = max(0, idx - 2)
    end = min(len(gate), idx + 3)

    slope, intercept = np.polyfit(
        gate[start:end],
        current[start:end],
        1
    )

    if slope <= 0:
        return np.nan, slope, intercept

    # Extrapolate the line to ID = 0
    vth_linear = -intercept / slope

    return vth_linear, slope, intercept


# =====================================================
# ION AND IOFF
# =====================================================

def get_current_at_gate(df, gate_voltage):
    gate = df["GATE"].to_numpy()
    current = np.abs(df["Id(M2)"].to_numpy())

    order = np.argsort(gate)

    return np.interp(
        gate_voltage,
        gate[order],
        current[order]
    )


# =====================================================
# CALCULATE PARAMETERS
# =====================================================

# Constant-current VTH
vth_low = extract_vth_constant_current(low)
vth_high = extract_vth_constant_current(high)

# Linear-extrapolation VTH
vth_lin_low, slope_low, intercept_low = extract_vth_linear(low)
vth_lin_high, slope_high, intercept_high = extract_vth_linear(high)

# ION at VGS = 1.1 V
ion_low = get_current_at_gate(low, 1.1)
ion_high = get_current_at_gate(high, 1.1)

# IOFF at VGS = 0 V
ioff_low = get_current_at_gate(low, 0.0)
ioff_high = get_current_at_gate(high, 0.0)

# DIBL using constant-current VTH
dibl = (vth_low - vth_high) / (VDS_HIGH - VDS_LOW)


# =====================================================
# PRINT RESULTS
# =====================================================

print("\n========== S2 RESULTS ==========")

print(f"VTH at VDS = 50 mV : {vth_low:.4f} V")
print(f"VTH at VDS = 1.1 V : {vth_high:.4f} V")

print(f"DIBL                : {dibl * 1000:.2f} mV/V")

print(f"ION                 : {ion_high:.6e} A")
print(f"IOFF                : {ioff_high:.6e} A")

print("\n--- LINEAR-EXTRAPOLATION VTH ---")

print(f"VTH,lin at VDS = 50 mV : {vth_lin_low:.4f} V")
print(f"VTH,lin at VDS = 1.1 V : {vth_lin_high:.4f} V")

print("================================\n")


# =====================================================
# PLOT 1: LINEAR ID-VGS
# =====================================================

plt.figure(figsize=(9, 6))

plt.plot(
    low["GATE"],
    np.abs(low["Id(M2)"]),
    label="VDS = 50 mV"
)

plt.plot(
    high["GATE"],
    np.abs(high["Id(M2)"]),
    label="VDS = 1.1 V"
)

plt.xlabel("Gate Voltage VGS (V)")
plt.ylabel("|Drain Current ID| (A)")
plt.title("S2: ID-VGS Linear Scale")
plt.grid(True)
plt.legend()
plt.tight_layout()


# =====================================================
# PLOT 2: LOG ID-VGS
# =====================================================

plt.figure(figsize=(9, 6))

plt.semilogy(
    low["GATE"],
    np.maximum(np.abs(low["Id(M2)"]), 1e-15),
    label="VDS = 50 mV"
)

plt.semilogy(
    high["GATE"],
    np.maximum(np.abs(high["Id(M2)"]), 1e-15),
    label="VDS = 1.1 V"
)

plt.xlabel("Gate Voltage VGS (V)")
plt.ylabel("|Drain Current ID| (A)")
plt.title("S2: ID-VGS Log Scale")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()


# =====================================================
# PLOT 3: CONSTANT-CURRENT VTH
# =====================================================

plt.figure(figsize=(9, 6))

plt.semilogy(
    low["GATE"],
    np.maximum(np.abs(low["Id(M2)"]), 1e-15),
    label="VDS = 50 mV"
)

plt.semilogy(
    high["GATE"],
    np.maximum(np.abs(high["Id(M2)"]), 1e-15),
    label="VDS = 1.1 V"
)

plt.axhline(
    I_THRESHOLD,
    linestyle="--",
    label="Threshold current = 1.5 uA"
)

if np.isfinite(vth_low):
    plt.scatter(
        vth_low,
        I_THRESHOLD,
        s=60,
        label=f"VTH,cc (50 mV) = {vth_low:.4f} V"
    )

if np.isfinite(vth_high):
    plt.scatter(
        vth_high,
        I_THRESHOLD,
        s=60,
        label=f"VTH,cc (1.1 V) = {vth_high:.4f} V"
    )

plt.xlabel("Gate Voltage VGS (V)")
plt.ylabel("|Drain Current ID| (A)")
plt.title("S2: Constant-Current VTH Extraction")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()


# =====================================================
# PLOT 4: LINEAR-EXTRAPOLATION VTH
# =====================================================

plt.figure(figsize=(9, 6))

for df, label in [
    (low, "VDS = 50 mV"),
    (high, "VDS = 1.1 V")
]:
    gate = df["GATE"].to_numpy()
    current = np.abs(df["Id(M2)"].to_numpy())

    order = np.argsort(gate)
    gate = gate[order]
    current = current[order]

    gm = np.gradient(current, gate)
    idx = np.argmax(gm)

    start = max(0, idx - 2)
    end = min(len(gate), idx + 3)

    slope, intercept = np.polyfit(
        gate[start:end],
        current[start:end],
        1
    )

    plt.plot(
        gate,
        current,
        label=f"{label} data"
    )

    if slope > 0:
        vth_linear = -intercept / slope

        # Extend the fitted line to the VTH intercept
        x_line = np.linspace(
            min(gate.min(), vth_linear),
            gate.max(),
            300
        )

        y_line = slope * x_line + intercept

        plt.plot(
            x_line,
            y_line,
            "--",
            label=f"{label} linear extrapolation"
        )

        plt.scatter(
            vth_linear,
            0,
            marker="x",
            s=80
        )

plt.xlabel("Gate Voltage VGS (V)")
plt.ylabel("|Drain Current ID| (A)")
plt.title("S2: VTH by Linear Extrapolation")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()