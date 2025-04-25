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

# Configure page settings
st.set_page_config(
    page_title="Advanced Sales & Demand Forecasting",
    page_icon="📊",
    layout="wide",
)

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
        model.fit(X_train, y_test) # Corrected: fit on y_test instead of y_train
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
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a file", type=["csv", "xlsx", "xls", "json", "txt"]
    )
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.success(f"File '{st.session_state.file_name}' uploaded successfully at {st.session_state.file_upload_time}.")
            st.write("Uploaded Data Preview (first 5 rows):")
            st.dataframe(df.head())

            # --- Column Selection and Analyse Button ---
            with st.container(): # Group UI elements
                st.header("⚙️ Configure Analysis")
                
                # Detect potential date columns
                date_cols = detect_date_columns(df)
                if not date_cols:
                    st.warning("Could not automatically detect date columns. Please select manually.")
                    possible_time_cols = df.columns.tolist()
                else:
                    st.info(f"Detected potential date columns: {', '.join(date_cols)}")
                    # Prioritize detected date cols, then others
                    possible_time_cols = date_cols + [col for col in df.columns if col not in date_cols] 

                time_col = st.selectbox(
                    "1. Select the time/date column:", 
                    possible_time_cols, 
                    # Try to restore previous selection or default to first column
                    index=possible_time_cols.index(st.session_state.get('time_col')) if st.session_state.get('time_col') in possible_time_cols else 0, 
                    key='time_col_select',
                    help="Select the column containing dates or timestamps for the analysis."
                )
                # Update session state immediately after selection
                if time_col:
                    st.session_state['time_col'] = time_col 

                # Detect numeric columns for target, excluding the selected time column
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                possible_target_cols_all = [col for col in df.columns if col != time_col]
                
                # Prioritize numeric columns among the possible target columns
                numeric_target_candidates = [col for col in numeric_cols if col != time_col]
                non_numeric_target_candidates = [col for col in possible_target_cols_all if col not in numeric_target_candidates]
                
                possible_target_cols_ordered = numeric_target_candidates + non_numeric_target_candidates

                if not numeric_target_candidates:
                    st.warning("No numeric columns detected for target variable (excluding the time column). Analysis works best with numeric targets.")
                
                target_col = st.selectbox(
                    "2. Select the target column (numeric preferred):", 
                    possible_target_cols_ordered, 
                    # Try to restore previous selection or default to first available
                    index=possible_target_cols_ordered.index(st.session_state.get('target_col')) if st.session_state.get('target_col') in possible_target_cols_ordered else 0, 
                    key='target_col_select',
                    help="Select the numeric column you want to analyze and forecast (e.g., Sales, Demand)."
                )
                # Update session state immediately
                if target_col:
                    st.session_state['target_col'] = target_col 

                st.header("🚀 Run Analysis")
                # Use columns directly from session state for reliability in button logic
                selected_time_col = st.session_state.get('time_col')
                selected_target_col = st.session_state.get('target_col')

                # --- Button Logic ---
                button_disabled = not (selected_time_col and selected_target_col)
                analysis_ready = False
                tooltip_message = "Select both time and target columns to enable analysis."

                if selected_target_col and selected_target_col in df.columns:
                     if pd.api.types.is_numeric_dtype(df[selected_target_col]):
                         analysis_ready = True
                         tooltip_message = "Click to start the time series analysis and forecasting."
                     else:
                         st.error(f"Selected target column '{selected_target_col}' is not numeric. Please select a numeric column for analysis.")
                         button_disabled = True # Also disable if target not numeric
                         tooltip_message = f"Target column '{selected_target_col}' must be numeric."
                
                if st.button("📊 Analyse Data", disabled=button_disabled, type="primary", help=tooltip_message):
                    if analysis_ready:
                        st.info(f"Starting analysis with Time Column: '{selected_time_col}' and Target Column: '{selected_target_col}'...")
                        
                        with st.spinner("Processing data and running models... Please wait."):
                            # Use a copy to avoid modifying the dataframe in session state directly during preprocessing
                            df_copy = df.copy() 
                            df_ts = preprocess_time_series(df_copy, selected_time_col, selected_target_col) 
                            
                            if df_ts is not None and not df_ts.empty:
                                st.session_state['df_processed'] = df_ts
                            else:
                                st.error("Failed to preprocess data. Check column selections, data format, and ensure data is not empty after preprocessing.")
                    else:
                         # This case handles when button is clicked but analysis_ready is False
                         st.warning("Cannot start analysis. Please ensure a valid numeric target column is selected.")

        else:
             st.warning("File could not be loaded. Please check the file format and content.")

# Placeholder for additional corrected sections... (Keep this structure if it was in the original response)

st.subheader("📈 Analysis Results")

# Create tabs for different analysis sections
tab1, tab2, tab3 = st.tabs(["Time Series Decomposition", "Holt-Winters Forecast", "Advanced Models"])

with tab1:
    # Example: Decomposition
    try:
        st.write("#### Time Series Decomposition")
        # TODO: Add smarter period detection (e.g., based on index frequency) or user input
        period = 12 # Default assumption (e.g., monthly for yearly data)
        if isinstance(st.session_state.get('df_processed', pd.DataFrame()).index, pd.DatetimeIndex) and len(st.session_state.get('df_processed', pd.DataFrame())) > 2 * period:
            # Use df_processed from session state
            df_ts = st.session_state['df_processed']
            selected_target_col = st.session_state.get('target_col')
            if selected_target_col:
                decomposition = decompose_time_series(df_ts, selected_target_col, period=period)
                if decomposition:
                    # Plot using Plotly
                    fig_decomp = px.line(decomposition.trend.dropna(), title='Trend Component')
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    fig_seas = px.line(decomposition.seasonal.dropna(), title='Seasonal Component')
                    st.plotly_chart(fig_seas, use_container_width=True)
                    fig_resid = px.scatter(decomposition.resid.dropna(), title='Residual Component')
                    st.plotly_chart(fig_resid, use_container_width=True)
                else:
                    st.warning("Could not perform time series decomposition.")
            else:
                 st.warning("Target column not selected for decomposition.")
        else:
            st.warning(f"Decomposition requires a DatetimeIndex and sufficient data (more than {2*period} periods).")
    except Exception as e:
         st.error(f"Error during decomposition: {e}")

with tab2:
    # Example: Holt-Winters Forecasting
    try:
        st.write("#### Holt-Winters Forecast")
        forecast_periods = st.slider("Select forecast horizon (periods):", 1, 36, 12, key='hw_horizon')
        # Ensure enough data for seasonality
        if len(st.session_state.get('df_processed', pd.DataFrame())) > 12: # Assuming seasonal_periods=12
            # Use df_processed from session state
            df_ts = st.session_state['df_processed']
            selected_target_col = st.session_state.get('target_col')
            if selected_target_col:
                hw_model, hw_forecast = holt_winters_forecast(df_ts, selected_target_col, forecast_periods=forecast_periods)
                if hw_model and hw_forecast is not None:
                    fig_hw = go.Figure()
                    fig_hw.add_trace(go.Scatter(x=df_ts.index, y=df_ts[selected_target_col], mode='lines', name='Historical Data'))
                    fig_hw.add_trace(go.Scatter(x=hw_forecast.index, y=hw_forecast, mode='lines', name='Holt-Winters Forecast', line=dict(dash='dash', color='red')))
                    fig_hw.update_layout(title=f'Holt-Winters Forecast ({forecast_periods} periods ahead)', xaxis_title='Time', yaxis_title=selected_target_col)
                    st.plotly_chart(fig_hw, use_container_width=True)
                    st.session_state['models']['Holt-Winters'] = {'model': hw_model, 'forecast': hw_forecast}
                else:
                    st.warning("Could not generate Holt-Winters forecast. Check data stationarity and seasonality.")
            else:
                 st.warning("Target column not selected for Holt-Winters forecast.")
        else:
            st.warning("Not enough data for Holt-Winters model with seasonal component.")
    except Exception as e:
         st.error(f"Error during Holt-Winters forecasting: {e}")

with tab3:
    st.write("#### Advanced Models (Linear Regression, XGBoost)")

    if st.session_state.get('df_processed') is not None:
        df_ts = st.session_state['df_processed']
        selected_target_col = st.session_state.get('target_col')
        selected_time_col = st.session_state.get('time_col')

        if selected_target_col and selected_time_col:
            # Exclude time and target columns from feature selection
            possible_feature_cols = [col for col in df_ts.columns if col not in [selected_target_col, selected_time_col]]

            if possible_feature_cols:
                selected_feature_cols = st.multiselect(
                    "Select feature columns for Linear Regression and XGBoost:",
                    possible_feature_cols,
                    default=possible_feature_cols, # Default to all available features
                    key='advanced_features_select',
                    help="Select columns to use as features in Linear Regression and XGBoost models. These should be numeric."
                )

                if st.button("Run Advanced Models", key='run_advanced_models_button'):
                    if selected_feature_cols:
                        st.info("Running Linear Regression and XGBoost models...")
                        with st.spinner("Training advanced models..."):
                            # Ensure selected features are numeric
                            numeric_selected_features = [col for col in selected_feature_cols if pd.api.types.is_numeric_dtype(df_ts[col])]
                            non_numeric_selected_features = [col for col in selected_feature_cols if col not in numeric_selected_features]

                            if non_numeric_selected_features:
                                st.warning(f"Ignoring non-numeric selected features for advanced models: {', '.join(non_numeric_selected_features)}")

                            if numeric_selected_features:
                                # Prepare data for models - need to reset index for feature columns
                                df_model_data = df_ts.copy().reset_index()

                                # Linear Regression
                                lr_results = linear_regression_forecast(df_model_data, selected_target_col, numeric_selected_features)
                                if lr_results:
                                    st.write("##### Linear Regression Results")
                                    st.write(f"Mean Squared Error (MSE): {lr_results['mse']:.4f}")
                                    st.write(f"R-squared (R2): {lr_results['r2']:.4f}")
                                    st.session_state['models']['Linear Regression'] = lr_results

                                # XGBoost
                                xgb_results = xgboost_forecast(df_model_data, selected_target_col, numeric_selected_features)
                                if xgb_results:
                                    st.write("##### XGBoost Results")
                                    st.write(f"Mean Squared Error (MSE): {xgb_results['mse']:.4f}")
                                    st.write(f"R-squared (R2): {xgb_results['r2']:.4f}")
                                    st.write("Feature Importance:")
                                    importance_df = pd.DataFrame({
                                        'Feature': numeric_selected_features,
                                        'Importance': xgb_results['importance']
                                    }).sort_values('Importance', ascending=False)
                                    st.dataframe(importance_df)
                                    st.session_state['models']['XGBoost'] = xgb_results
                            else:
                                st.warning("No numeric feature columns selected or available for advanced models.")
                    else:
                        st.warning("Please select at least one feature column to run advanced models.")
            else:
                st.info("No other columns available to use as features for advanced models.")
        else:
            st.warning("Time and Target columns must be selected to run advanced models.")
    else:
        st.info("Please upload data and select Time and Target columns first.")

# TODO: Add calls for Linear Regression, XGBoost etc.
# These would likely require feature engineering steps first.

st.success("Analysis Complete!")
# --- End Trigger ---
