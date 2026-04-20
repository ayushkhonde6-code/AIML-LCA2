import pandas as pd
import numpy as np

# Configuration for a typical mid-range EV
m = 1850        # Mass (kg) + payload
Cd = 0.24       # Drag Coefficient
A = 2.35        # Frontal Area (m^2)
Crr = 0.012     # Rolling Resistance
rho = 1.225     # Air Density
g = 9.81        # Gravity
v_nom = 380     # Nominal Battery Voltage (V)
batt_cap_kwh = 75 

# Generate WLTP-like Velocity Profile (1800 seconds)
t = np.arange(0, 1801, 1)
velocity_kmh = 40 * (1 - np.cos(2 * np.pi * t / 600)) + 20 * np.sin(2 * np.pi * t / 1800)
velocity_kmh = np.maximum(0, velocity_kmh + np.random.normal(0, 1.5, 1801)) # Add noise
v_ms = velocity_kmh / 3.6
accel = np.gradient(v_ms)

# Power Calculations
F_rolling = Crr * m * g
F_drag = 0.5 * rho * Cd * A * v_ms**2
F_accel = m * accel
P_wheel = (F_rolling + F_drag + F_accel) * v_ms

# Powertrain Efficiency & Regen
P_elec = np.where(P_wheel >= 0, P_wheel / 0.88, P_wheel * 0.70)
current = P_elec / v_nom
voltage = v_nom - (current * 0.05) # Voltage sag

# State of Charge (SoC) Calculation
energy_consumed_j = np.cumsum(P_elec)
soc = 95 - (energy_consumed_j / (batt_cap_kwh * 3600000) * 100)

# Create Dataset
ev_data = pd.DataFrame({
    'timestamp_s': t,
    'speed_kmh': velocity_kmh.round(2),
    'accel_ms2': accel.round(3),
    'motor_torque_nm': (P_wheel / (v_ms + 0.1) * 0.3 / 9).round(2), # Simplified
    'battery_voltage_v': voltage.round(2),
    'battery_current_a': current.round(2),
    'inverter_temp_c': 35 + np.cumsum(np.abs(current) * 0.001) + np.random.normal(0, 0.1, 1801),
    'soc_perc': soc.round(4)
})

# Save to CSV
ev_data.to_csv('ev_powertrain_test_data.csv', index=False)
print("File 'ev_powertrain_test_data.csv' has been generated.")