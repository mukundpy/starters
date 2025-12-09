# INSTALL FIRST (run this separately)
# pip install streamlit prophet yfinance plotly cmdstanpy


import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 Stock Price Forecast App")


# ---------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------
stocks = ("GOOG", "AAPL", "MSFT", "GME")
selected_stock = st.selectbox("Select stock for prediction", stocks)

n_years = st.slider("Years to forecast:", 1, 4)
period = n_years * 365


# ---------------------------------------------------
# SAFE DATA LOADER — Handles all Yahoo issues
# ---------------------------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)

    # 1. Check for empty data
    if data is None or data.empty:
        st.error("❌ Yahoo Finance returned no data. Try another stock or later.")
        st.stop()

    # 2. Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce")

    # 3. Drop rows with invalid dates
    data = data.dropna()

    # 4. Ensure essential columns exist
    required_cols = ["Open", "High", "Low", "Close"]
    for col in required_cols:
        if col not in data.columns:
            st.error(f"❌ Missing '{col}' column in downloaded data. Cannot continue.")
            st.stop()

    # 5. Reset index → Convert Date index to column
    data = data.reset_index()
    data.rename(columns={"index": "Date"}, inplace=True)

    return data


# LOAD DATA
st.text("Loading data…")
data = load_data(selected_stock)
st.text("Data loaded!")


# ---------------------------------------------------
# SHOW RAW DATA
# ---------------------------------------------------
st.subheader("Raw data")
st.write(data.tail())


# ---------------------------------------------------
# PLOT RAW DATA
# ---------------------------------------------------
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Open Price"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close Price"))
    fig.layout.update(title="Stock Price Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()


# ---------------------------------------------------
# PREPARE DATA FOR PROPHET
# ---------------------------------------------------
df_train = data[["Date", "Close"]].copy()
df_train.columns = ["ds", "y"]

# Convert types safely
df_train["ds"] = pd.to_datetime(df_train["ds"], errors="coerce")
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")

# Drop rows with invalid data
df_train = df_train.dropna(subset=["ds", "y"])

# Final safety check
if df_train.empty:
    st.error("❌ Error: Cleaned training data is empty. Cannot train Prophet.")
    st.stop()


# ---------------------------------------------------
# TRAIN PROPHET MODEL
# ---------------------------------------------------
m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


# ---------------------------------------------------
# DISPLAY FORECAST OUTPUT
# ---------------------------------------------------
st.subheader("📊 Forecast Data")
st.write(forecast.tail())

st.subheader(f"📉 Forecast Plot for {n_years} Years")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("🔎 Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
