import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy import stats

# Main Application Logic
st.title("📊 Advanced Sales & Demand Forecasting")
st.markdown("Upload your sales data to get AI-powered insights and forecasts.")
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a file", type=["csv", "xlsx", "xls", "json", "txt"]
    )


# Add an "Analyze Data" button
if st.button("Analyze Data"):
    if st.session_state["df"] is not None:
        st.write("### Data Preview")
        st.dataframe(st.session_state["df"].head())
        
        st.write("### Data Summary")
        st.write(st.session_state["df"].describe())
    else:
        st.error("No data uploaded. Please upload a file to analyze.")

# Initialize session state variables if they don't exist
for key in [
    "df",
    "file_name",
    "file_upload_time",
    "predictions",
    "models",
    "time_col",
    "target_col",
]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "models" else {}

# Function to load data
def load_data(uploaded_file):
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension == "json":
            df = pd.read_json(uploaded_file)
        elif file_extension == "txt":
            df = pd.read_csv(uploaded_file, sep="\t")
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

        st.session_state.update(
            {
                "df": df,
                "file_name": uploaded_file.name,
                "file_upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Function to detect date columns
def detect_date_columns(df):
    return [
        col
        for col in df.columns
        if pd.to_datetime(df[col], errors="coerce").notna().all()
    ]


# Function to preprocess time series data
def preprocess_time_series(df, date_column, target_column):
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df_ts = df.set_index(date_column).sort_index()
        if df_ts[target_column].isnull().sum() > 0:
            df_ts[target_column] = df_ts[target_column].interpolate(method="linear")
        return df_ts
    except Exception as e:
        st.error(f"Error preprocessing time series data: {e}")
        return None


# Function to decompose time series
def decompose_time_series(df, column, period=None):
    try:
        period = period or 12  # Default to monthly seasonality
        decomposition = seasonal_decompose(df[column], model="additive", period=period)
        return decomposition
    except Exception as e:
        st.error(f"Error decomposing time series: {e}")
        return None


# Function for Holt-Winters exponential smoothing
def holt_winters_forecast(df, column, forecast_periods=30):
    try:
        model = ExponentialSmoothing(
            df[column], trend="add", seasonal="add", seasonal_periods=12
        ).fit()
        return model, model.forecast(forecast_periods)
    except Exception as e:
        st.error(f"Error in Holt-Winters forecasting: {e}")
        return None, None


# Function for linear regression forecast
def linear_regression_forecast(df, target_column, feature_columns):
    try:
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        scaler = StandardScaler().fit(X_train)
        model = LinearRegression()
        model.fit(scaler.transform(X_train), y_train)
        return {
            "model": model,
            "scaler": scaler,
            "mse": mean_squared_error(y_test, model.predict(scaler.transform(X_test))),
            "r2": r2_score(y_test, model.predict(scaler.transform(X_test))),
        }
    except Exception as e:
        st.error(f"Error in linear regression forecasting: {e}")
        return None


# Function for XGBoost forecast
def xgboost_forecast(df, target_column, feature_columns):
    try:
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X_train, y_train)
        return {
            "model": model,
            "mse": mean_squared_error(y_test, model.predict(X_test)),
            "r2": r2_score(y_test, model.predict(X_test)),
            "importance": model.feature_importances_,
        }
    except Exception as e:
        st.error(f"Error in XGBoost forecasting: {e}")
        return None


# Main Application Logic
st.title("📊 Advanced Sales & Demand Forecasting")
st.markdown("Upload your sales data to get AI-powered insights and forecasts.")

# Sidebar Section
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload a file", type=["csv", "xlsx", "xls", "json", "txt"]
    )
    if uploaded_file:
        st.sidebar.success("File uploaded successfully!")
        df = load_data(uploaded_file)

        if st.button("Analyze Data"):
            if st.session_state.df is not None:
                st.sidebar.success("Data is ready for analysis!")
            else:
                st.sidebar.error("Please upload a valid file to analyze.")

# Main content area
if st.session_state.df is not None:
    st.write(f"**Uploaded File:** {st.session_state.file_name}")
    st.write(f"**Uploaded At:** {st.session_state.file_upload_time}")
    st.dataframe(st.session_state.df.head())
else:
    st.info("Please upload a data file to begin.")
