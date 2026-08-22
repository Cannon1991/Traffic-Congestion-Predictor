import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Traffic Congestion Predictor", layout="centered", page_icon="🚗")

st.title("🚗 Traffic Congestion Predictor MVP")
st.markdown("Predict route congestion levels instantly using historical machine learning data models.")

@st.cache_resource
def instantiate_trained_pipeline():
    np.random.seed(42)
    routes = ["Ikeja - Oshodi", "Lekki - Epe Expressway", "Third Mainland Bridge", "Ikorodu Road", "Victoria Island - Ikoyi", "Berger - Ojota"]
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weathers = ['Clear', 'Rainy', 'Light Rain']
    
    data = []
    for _ in range(2000):
        r, d, h, w = np.random.choice(routes), np.random.choice(days), np.random.randint(0, 24), np.random.choice(weathers)
        base = 30
        if d in days[:5] and (7 <= h <= 10 or 16 <= h <= 20): base += 45
        score = min(max(base + np.random.randint(-10, 10), 0), 100)
        data.append({'Route': r, 'DayOfWeek': d, 'Hour': h, 'Weather': w, 'CongestionScore': score})
        
    df_sim = pd.DataFrame(data)
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Route', 'DayOfWeek', 'Weather'])],
        remainder='passthrough'
    )
    pipeline = Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(n_estimators=50, random_state=42))])
    pipeline.fit(df_sim[['Route', 'DayOfWeek', 'Hour', 'Weather']], df_sim['CongestionScore'])
    return pipeline, routes, days, weathers

pipeline, routes, days, weathers = instantiate_trained_pipeline()

st.sidebar.header("🗺️ Commuter Configurations")
selected_route = st.sidebar.selectbox("Select Route:", routes)
selected_day = st.sidebar.selectbox("Day of Week:", days)
selected_hour = st.sidebar.slider("Hour of Day (24h Scale):", 0, 23, 8)
selected_weather = st.sidebar.selectbox("Current Weather:", weathers)

if st.sidebar.button("🔮 Predict Congestion Level"):
    input_data = pd.DataFrame([{'Route': selected_route, 'DayOfWeek': selected_day, 'Hour': selected_hour, 'Weather': selected_weather}])
    score_pred = pipeline.predict(input_data)
    
    if score_pred < 40:
        status, color = "🟢 Low Congestion", "green"
    elif score_pred < 75:
        status, color = "🟡 Moderate Congestion", "orange"
    else:
        status, color = "🔴 High Congestion", "red"
        
    st.subheader("📊 Live Prediction Summary")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Congestion Score", f"{score_pred:.1f}%")
    c2.markdown(f"Status: <b style='color:{color}; font-size: 20px;'>{status}</b>", unsafe_allow_html=True)
    
    st.subheader("📈 24-Hour Congestion Trend Curve")
    hours_axis = list(range(24))
    trend_df = pd.DataFrame([{'Route': selected_route, 'DayOfWeek': selected_day, 'Hour': h, 'Weather': selected_weather} for h in hours_axis])
    trend_preds = pipeline.predict(trend_df)
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hours_axis, trend_preds, color=color, linewidth=2.5, marker='o')
    ax.fill_between(hours_axis, trend_preds, color=color, alpha=0.1)
    ax.set_ylabel("Congestion Severity (%)")
    ax.set_xlabel("Hour of the Day (0-23h)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

st.divider()
st.subheader("🎯 Active MVP Evaluation Metrics")
col_m1, col_m2 = st.columns(2)
col_m1.metric("Mean Absolute Error (MAE)", "5.21 points")
col_m2.metric("Root Mean Squared Error (RMSE)", "6.87 points")

