import pandas as pd
import matplotlib.pyplot as plt

# 1. Load TCAD files
tcad_idvg = pd.read_csv('e5_b.csv')
tcad_idvd_09 = pd.read_csv('S6_idvd_0.9V.csv')

# 2. Load SPICE exported files
spice_idvg = pd.read_csv('S6-exp3_IDVG.txt', sep=r'\s+')
spice_idvd = pd.read_csv('S6-exp3_IDVD.txt', sep=r'\s+')  # Make sure filename matches your explorer!

# --- Plot 1: ID - VG Overlay ---
plt.figure(figsize=(7, 5))
plt.plot(tcad_idvg['drain TotalCurrent(IdVg_n35_des) X'], 
         tcad_idvg['drain TotalCurrent(IdVg_n35_des) Y'] * 1e6, 
         'ro', markevery=5, label='TCAD (Sentaurus)')

# Dynamically pick SPICE Vg and Id columns
vg_col_spice = spice_idvg.columns[0]
id_col_spice1 = spice_idvg.columns[1]
plt.plot(spice_idvg[vg_col_spice], spice_idvg[id_col_spice1] * 1e6, 'b-', linewidth=2, label='SPICE (nfit)')

plt.xlabel('Gate Voltage $V_G$ (V)')
plt.ylabel(r'Drain Current $I_D$ ($\mu$A/$\mu$m)')
plt.title(r'$I_D - V_G$ Matching at $V_{DS} = 0.4$ V')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('S6_IdVg_exp3.png')

# --- Plot 2: ID - VD Overlay ---
plt.figure(figsize=(7, 5))

# Dynamically handle TCAD columns
x_col = 'Drain_Voltage_V' if 'Drain_Voltage_V' in tcad_idvd_09.columns else 'drain TotalCurrent(IdVd_n367_des) X'
y_col = 'Drain_Current_A' if 'Drain_Current_A' in tcad_idvd_09.columns else 'drain TotalCurrent(IdVd_n367_des) Y'

plt.plot(tcad_idvd_09[x_col], 
         tcad_idvd_09[y_col] * 1e3, 
         'ro', markevery=5, label='TCAD (Sentaurus)')

# Dynamically pick SPICE Vd and Id columns (Fixes KeyError: 'V2')
vd_col_spice = spice_idvd.columns[0]
id_col_spice2 = spice_idvd.columns[1]
plt.plot(spice_idvd[vd_col_spice], spice_idvd[id_col_spice2] * 1e3, 'b-', linewidth=2, label='SPICE (nfit)')

plt.xlabel('Drain Voltage $V_D$ (V)')
plt.ylabel(r'Drain Current $I_D$ (mA/$\mu$m)')
plt.title(r'$I_D - V_D$ Matching at $V_{GS} = 0.9$ V')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('S6_IdVd_exp3.png')

# Show both plots together
plt.show()