# Traffic Congestion Predictor MVP

[![Streamlit App](https://streamlit.io)](https://streamlit.io)

This repository provides an intelligent Machine Learning micro-service framework designed to give real-time predictive congestion estimations for primary commuter routes across transit corridors.

## 📦 Project Features
* **Route & Time Inputs:** Custom sidebar dropdowns to pick routes, days, hours, and weather.
* **Smart ML Inference:** Real-time regression scoring using an optimized Scikit-Learn Pipeline.
* **Interactive Visualization:** Dynamically generated 24-hour traffic trend charts.
* **Transparent Metrics:** Displays continuous validation errors (MAE and RMSE) on the dashboard.

## 🚀 Project Setup
1. Clone this repository to your local computer.
2. Install the required libraries:  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch your platform locally:  
   ```bash
   streamlit run app.py
   ```
