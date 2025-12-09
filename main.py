import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Page Configuration
st.set_page_config(
    page_title="Indian Stock Forecast",
    page_icon="📈",
    layout="wide"
)

# App Header
st.title("📈 Indian Stock Price Forecast App")
st.markdown("---")

# Indian Stock Tickers
indian_tickers = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS", 
    "ICICIBANK.NS", "ITC.NS", "HINDUNILVR.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "LT.NS", "MARUTI.NS", "SBIN.NS", "BAJAJ-AUTO.NS", "HCLTECH.NS", "WIPRO.NS",
    "ULTRACEMCO.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "TITAN.NS", "ONGC.NS",
    "BAJAJFINSV.NS", "POWERGRID.NS", "JSWSTEEL.NS", "DMART.NS", "NESTLEIND.NS",
    "HINDALCO.NS", "COALINDIA.NS", "BPCL.NS", "EICHERMOT.NS", "DRREDDY.NS",
    "BHARATFORG.NS", "ADANIPORTS.NS", "TATAMOTORS.NS", "BRITANNIA.NS", "M&M.NS"
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
    start_date = end_date - timedelta(days=365*3)  # 3 years
    
    col1, col2 = st.columns(2)
    with col1:
        start_date_input = st.date_input("Start Date", start_date)
    with col2:
        end_date_input = st.date_input("End Date", end_date)
    
    # Forecast settings
    n_years = st.slider("Years to forecast:", 1, 3, 1)
    period = n_years * 365
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

# Data loading function
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        # Download data
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        if data.empty:
            return None
        
        # Reset index
        data = data.reset_index()
        
        # Ensure Date column exists
        if 'Date' not in data.columns:
            # Rename the first column to Date
            data = data.rename(columns={data.columns[0]: 'Date'})
        
        # Convert Date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Use Adj Close if available, otherwise Close
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        elif 'close' in [col.lower() for col in data.columns]:
            for col in data.columns:
                if 'close' in col.lower():
                    data['Close'] = data[col]
                    break
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
with st.spinner(f"Loading {selected_stock} data..."):
    data = load_data(selected_stock, start_date_input, end_date_input)

if data is None or data.empty:
    st.error("❌ Could not load data. Please try a different stock or date range.")
    
    # Create sample data for demo
    st.info("Showing sample data for demonstration...")
    dates = pd.date_range(start=start_date_input, end=end_date_input, freq='B')
    np.random.seed(42)
    base_price = 1000
    returns = np.random.randn(len(dates)) * 0.015
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices * 0.995,
        'High': prices * 1.015,
        'Low': prices * 0.985
    })

# Display current price
if 'Close' in data.columns:
    current_price = data['Close'].iloc[-1]
    st.metric(f"Current Price ({selected_stock})", f"₹{current_price:,.2f}")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Price Chart", "🔮 Forecast", "📋 Data"])

with tab1:
    # Price chart
    st.subheader("Price Movement")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Closing Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"{selected_stock} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Forecasting
    st.subheader("Price Forecast")
    
    if len(data) < 30:
        st.warning(f"Need at least 30 days of data for forecasting. Currently: {len(data)} days")
    else:
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Training forecast model..."):
                try:
                    # Prepare data for Prophet
                    df_train = data[['Date', 'Close']].copy()
                    df_train.columns = ['ds', 'y']
                    df_train['ds'] = pd.to_datetime(df_train['ds'])
                    
                    # Train model
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False
                    )
                    
                    model.fit(df_train)
                    
                    # Create future dataframe
                    future = model.make_future_dataframe(periods=period)
                    
                    # Generate forecast
                    forecast = model.predict(future)
                    
                    # Plot forecast
                    fig_forecast = plot_plotly(model, forecast)
                    fig_forecast.update_layout(
                        title=f"{selected_stock} {n_years}-Year Forecast",
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Price (₹)"
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Show forecast summary
                    st.subheader("Forecast Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        last_price = df_train['y'].iloc[-1]
                        st.metric("Current Price", f"₹{last_price:,.2f}")
                    
                    with col2:
                        forecast_price = forecast['yhat'].iloc[-1]
                        change_pct = ((forecast_price - last_price) / last_price) * 100
                        st.metric(
                            f"Forecast Price ({n_years} years)",
                            f"₹{forecast_price:,.2f}",
                            delta=f"{change_pct:.1f}%"
                        )
                    
                except Exception as e:
                    st.error(f"Forecast error: {str(e)}")
                    st.info("Try with more historical data.")

with tab3:
    # Data tab
    st.subheader("Historical Data")
    
    # Show data statistics
    if not data.empty:
        st.write(f"**Data Range:** {data['Date'].min().date()} to {data['Date'].max().date()}")
        st.write(f"**Total Days:** {len(data)}")
        
        # Data preview
        st.dataframe(
            data.tail(20),
            use_container_width=True
        )
        
        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_stock.replace('.NS', '')}_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>📊 Indian Stock Forecast App • Data from Yahoo Finance • Powered by Prophet</p>
    <p><em>⚠️ For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)
