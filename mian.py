import streamlit as st
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Indian Stock Forecast",
    page_icon="📈",
    layout="wide"
)

# App Header
st.title("📈 Indian Stock Price Forecast")
st.markdown("---")

# Indian Stock Tickers with NS suffix
indian_tickers = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS", 
    "ICICIBANK.NS", "ITC.NS", "HINDUNILVR.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "LT.NS", "MARUTI.NS", "SBIN.NS", "BAJAJ-AUTO.NS", "HCLTECH.NS", "WIPRO.NS",
    "ULTRACEMCO.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "TITAN.NS", "ONGC.NS",
    "BAJAJFINSV.NS", "POWERGRID.NS", "JSWSTEEL.NS", "DMART.NS", "NESTLEIND.NS",
    "HINDALCO.NS", "COALINDIA.NS", "BPCL.NS", "EICHERMOT.NS", "DRREDDY.NS",
    "BHARATFORG.NS", "ADANIPORTS.NS", "TATAMOTORS.NS", "BRITANNIA.NS", "M&M.NS",
    "DIVISLAB.NS", "IOC.NS", "CIPLA.NS", "TECHM.NS", "VEDL.NS"
]

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    selected_stock = st.selectbox(
        "Select Indian Stock:",
        indian_tickers,
        index=0
    )
    
    # Date range
    end_date = date.today()
    start_date = end_date - timedelta(days=3*365)
    
    start_date_input = st.date_input("Start Date", start_date)
    end_date_input = st.date_input("End Date", end_date)
    
    # Forecast settings
    n_years = st.slider("Years to forecast:", 1, 3, 1)
    period = n_years * 365
    
    if st.button("🔄 Refresh"):
        st.cache_data.clear()

# Data loading function
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        if data.empty:
            return None
            
        data = data.reset_index()
        
        # Handle column names
        if 'Date' not in data.columns:
            data = data.rename(columns={data.columns[0]: 'Date'})
        
        # Ensure we have Close price
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        
        data['Date'] = pd.to_datetime(data['Date'])
        
        return data
    
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Load data
with st.spinner("Loading stock data..."):
    data = load_data(selected_stock, start_date_input, end_date_input)

if data is None or data.empty:
    st.error("Could not load data. Please try a different stock or date range.")
    
    # Create sample data for demo
    dates = pd.date_range(start=start_date_input, end=end_date_input, freq='B')
    prices = 1000 + np.random.randn(len(dates)).cumsum() * 10
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98
    })
    
    st.warning("Showing sample data for demonstration.")

# Display current price
if not data.empty:
    current_price = data['Close'].iloc[-1]
    st.metric(f"Current Price ({selected_stock})", f"₹{current_price:.2f}")

# Plot price chart
st.subheader("Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='lines',
    name='Price',
    line=dict(color='blue', width=2)
))
fig.update_layout(
    title=f"{selected_stock} Price",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Prepare for forecasting
if len(data) > 30:
    df_train = data[['Date', 'Close']].copy()
    df_train.columns = ['ds', 'y']
    
    st.subheader(f"Forecast for next {n_years} year(s)")
    
    if st.button("Generate Forecast"):
        with st.spinner("Training model..."):
            try:
                model = Prophet()
                model.fit(df_train)
                
                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)
                
                # Plot forecast
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(
                    title=f"{selected_stock} {n_years}-Year Forecast",
                    height=500
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Show forecast summary
                st.subheader("Forecast Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
                
                with col2:
                    forecast_price = forecast['yhat'].iloc[-1]
                    change = ((forecast_price - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
                    st.metric(
                        f"Forecast Price",
                        f"₹{forecast_price:.2f}",
                        delta=f"{change:.1f}%"
                    )
                
            except Exception as e:
                st.error(f"Forecast error: {e}")
else:
    st.warning("Need at least 30 days of data for forecasting")

# Show raw data
with st.expander("View Raw Data"):
    st.dataframe(data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Data from Yahoo Finance | For educational purposes only*")
