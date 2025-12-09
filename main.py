import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Indian Stocks", layout="wide")

# Title
st.title("📈 Indian Stock Analysis")
st.markdown("---")

# Indian stocks
indian_stocks = [
    ("Reliance", "RELIANCE.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("TCS", "TCS.NS"),
    ("Infosys", "INFY.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("ITC", "ITC.NS"),
    ("HUL", "HINDUNILVR.NS"),
    ("Axis Bank", "AXISBANK.NS"),
    ("Bajaj Finance", "BAJFINANCE.NS"),
    ("SBI", "SBIN.NS"),
    ("HCL Tech", "HCLTECH.NS"),
    ("Wipro", "WIPRO.NS"),
    ("Sun Pharma", "SUNPHARMA.NS"),
    ("Asian Paints", "ASIANPAINT.NS"),
    ("Titan", "TITAN.NS"),
    ("ONGC", "ONGC.NS"),
    ("Power Grid", "POWERGRID.NS"),
    ("Nestle", "NESTLEIND.NS"),
    ("Coal India", "COALINDIA.NS"),
    ("Tata Motors", "TATAMOTORS.NS")
]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Create selection with names
    stock_names = [s[0] for s in indian_stocks]
    selected_name = st.selectbox("Select Company", stock_names)
    
    # Get ticker
    selected_ticker = [s[1] for s in indian_stocks if s[0] == selected_name][0]
    
    # Time period
    period = st.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    # Chart type
    chart_type = st.radio("Chart Type", ["Line", "Candlestick"])

# Main content
st.subheader(f"{selected_name} ({selected_ticker})")

# Load data
@st.cache_data
def load_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    except:
        return None

# Load with spinner
with st.spinner("Loading stock data..."):
    data = load_stock_data(selected_ticker, period)

if data is None or data.empty:
    st.error("❌ Could not load data. Please try again.")
    
    # Show sample data
    st.info("Showing sample data for demonstration...")
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    prices = 1000 + pd.Series(range(100)).cumsum() * 5 + pd.Series(np.random.randn(100).cumsum() * 20)
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
else:
    st.success(f"✅ Loaded {len(data)} days of data")

# Display metrics
if not data.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['Close'].iloc[-1]
        st.metric("Current Price", f"₹{current_price:,.2f}")
    
    with col2:
        if len(data) > 1:
            prev_price = data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            st.metric("Daily Change", f"₹{change:,.2f}", f"{change_pct:.2f}%")
    
    with col3:
        st.metric("52 Week High", f"₹{data['High'].max():,.2f}")
    
    with col4:
        st.metric("52 Week Low", f"₹{data['Low'].min():,.2f}")

# Create chart
st.subheader("Price Chart")

if chart_type == "Line":
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
else:
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    )])

fig.update_layout(
    title=f"{selected_name} Price",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    height=500,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Show data table
with st.expander("📊 View Data Table"):
    st.dataframe(data.tail(20), use_container_width=True)

# Additional analysis
st.subheader("📈 Analysis")

if not data.empty and len(data) > 20:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Returns Analysis**")
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) > 0:
            avg_return = returns.mean() * 100
            volatility = returns.std() * 100
            total_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            
            st.write(f"Average Daily Return: {avg_return:.2f}%")
            st.write(f"Daily Volatility: {volatility:.2f}%")
            st.write(f"Total Period Return: {total_return:.2f}%")
    
    with col2:
        st.markdown("**Volume Analysis**")
        if 'Volume' in data.columns:
            avg_volume = data['Volume'].mean()
            recent_volume = data['Volume'].iloc[-1]
            
            st.write(f"Average Volume: {avg_volume:,.0f}")
            st.write(f"Recent Volume: {recent_volume:,.0f}")
            
            # Volume chart
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume'
            ))
            fig_vol.update_layout(height=300, title="Trading Volume")
            st.plotly_chart(fig_vol, use_container_width=True)

# Download button
if not data.empty:
    csv = data.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv,
        file_name=f"{selected_name.replace(' ', '_')}_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>📊 Indian Stock Analysis • Data from Yahoo Finance</p>
    <p><em>For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)
