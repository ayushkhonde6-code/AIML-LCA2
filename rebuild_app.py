app_code = """import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="EV Digital Twin Dashboard")
st.title("⚡ EV Powertrain Digital Twin Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("ev_powertrain_test_data.csv")

df = load_data()
df['power_w'] = df['battery_voltage_v'] * df['battery_current_a']

st.sidebar.header("Controls")
time_range = st.sidebar.slider("Select Time Range", 0, len(df), (0, len(df)))
df_f = df.iloc[time_range[0]:time_range[1]]

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔋 SOC", f"{df_f['soc_perc'].iloc[-1]:.2f}%")
col2.metric("⚡ Avg Power", f"{df_f['power_w'].mean():.2f} W")
col3.metric("🚗 Avg Speed", f"{df_f['speed_kmh'].mean():.2f} km/h")
col4.metric("🌡️ Max Temp", f"{df_f['inverter_temp_c'].max():.2f} °C")

st.subheader("📊 Real-Time Powertrain Data")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(df_f, x='timestamp_s', y='speed_kmh', title="Speed"))
    st.plotly_chart(px.line(df_f, x='timestamp_s', y='motor_torque_nm', title="Torque"))
with c2:
    st.plotly_chart(px.line(df_f, x='timestamp_s', y='battery_voltage_v', title="Voltage"))
    st.plotly_chart(px.line(df_f, x='timestamp_s', y='battery_current_a', title="Current"))

st.subheader("🔋 Battery Intelligence")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(df_f, x='timestamp_s', y='soc_perc', title="SOC"))
with c2:
    st.plotly_chart(px.line(df_f, x='timestamp_s', y='power_w', title="Power"))

@st.cache_resource
def train_models(data):
    features = ['speed_kmh', 'accel_ms2', 'battery_voltage_v', 'battery_current_a', 'soc_perc']
    X = data[features]
    y = data['speed_kmh']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(data[['speed_kmh', 'accel_ms2', 'motor_torque_nm']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_s)
    return model, iso, scaler, kmeans

model, iso, scaler, kmeans = train_models(df)

st.subheader("⚠️ Anomaly Detection")
features = ['speed_kmh', 'accel_ms2', 'battery_voltage_v', 'battery_current_a', 'soc_perc']
df['anom'] = iso.predict(df[features])
st.plotly_chart(px.scatter(df_f, x='timestamp_s', y='battery_current_a', color=df_f['anom'].map({1: "Normal", -1: "Anomaly"}), title="Anomaly"))
st.warning(f"Detected {len(df[df['anom'] == -1])} anomalies")

st.subheader("🚗 Driving Pattern")
X_s = scaler.transform(df[['speed_kmh', 'accel_ms2', 'motor_torque_nm']])
df['mode'] = kmeans.predict(X_s)
st.plotly_chart(px.scatter(df_f, x='speed_kmh', y='accel_ms2', color=df_f['mode'], title="Clusters"))

st.subheader("🌡️ Thermal Monitoring")
st.plotly_chart(px.line(df_f, x='timestamp_s', y='inverter_temp_c', title="Temperature"))

st.subheader("🎛️ Digital Twin Simulation")
c1, c2, c3 = st.columns(3)
with c1:
    u_speed = st.slider("Speed", 0, 150, int(df['speed_kmh'].mean()))
    u_acc = st.slider("Acceleration", -5, 5, 0)
with c2:
    u_volt = st.slider("Voltage", int(df['battery_voltage_v'].min()), int(df['battery_voltage_v'].max()), int(df['battery_voltage_v'].mean()))
    u_curr = st.slider("Current", int(df['battery_current_a'].min()), int(df['battery_current_a'].max()), int(df['battery_current_a'].mean()))
with c3:
    u_soc = st.slider("SOC", 0, 100, 50)
    u_temp = st.slider("Temp", int(df['inverter_temp_c'].min()), int(df['inverter_temp_c'].max()), int(df['inverter_temp_c'].mean()))

inp = pd.DataFrame([{'speed_kmh': u_speed, 'accel_ms2': u_acc, 'battery_voltage_v': u_volt, 'battery_current_a': u_curr, 'soc_perc': u_soc}])
pred = model.predict(inp)[0]
anom = iso.predict(inp)[0]
drv = kmeans.predict(scaler.transform([[u_speed, u_acc, u_speed]]))[0]
pwr = u_volt * u_curr

st.subheader("📊 Simulation Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Speed", f"{pred:.1f} km/h")
c2.metric("Power", f"{pwr:.0f} W")
c3.metric("Mode", f"{drv}")
c4.metric("Temp", f"{u_temp}°C")

st.subheader("🚨 Alerts")
alerts = []
if u_temp > 80: alerts.append("🔥 High Temp")
if u_curr > df['battery_current_a'].mean() * 2: alerts.append("⚡ High Current")
if u_soc < 20: alerts.append("🔋 Low Battery")
if anom == -1: alerts.append("⚠️ Anomaly")
if alerts:
    for a in alerts: st.error(a)
else:
    st.success("Normal ✅")
"""

with open('app.py', 'w') as f:
    f.write(app_code)
print("app.py rebuilt successfully!")
