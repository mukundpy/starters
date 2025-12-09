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
    page_title="Indian Stock Forecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #FF9933 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF9933;
        margin-bottom: 1rem;
    }
    .stock-name {
        color: #138808;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF9933;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Indian Stock Tickers with NS/BSE suffix
indian_tickers = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS", 
    "ICICIBANK.NS", "HDFC.NS", "KOTAKBANK.NS", "ITC.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "LT.NS", "MARUTI.NS", "SBIN.NS", 
    "BAJAJ-AUTO.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
    "ASIANPAINT.NS", "TITAN.NS", "ONGC.NS", "BAJAJFINSV.NS", "POWERGRID.NS",
    "JSWSTEEL.NS", "HDFC-LIFE.NS", "DMART.NS", "NESTLEIND.NS", "HINDALCO.NS",
    "COALINDIA.NS", "BPCL.NS", "EICHERMOT.NS", "GRASIM.NS", "DRREDDY.NS",
    "BHARATFORG.NS", "ADANIPORTS.NS", "ADANIGREEN.NS", "RELIANCEIND.NS",
    "HAVELLS.NS", "GAIL.NS", "BHARTI.NS", "ZEEL.NS", "UPL.NS", "TATAMOTORS.NS",
    "TATACONSUM.NS", "BRITANNIA.NS", "M&M.NS", "DIVISLAB.NS", "IOC.NS",
    "SHREECEM.NS", "SBILIFE.NS", "TECHM.NS", "CIPLA.NS", "LTIM.NS",
    "JSWENERGY.NS", "VEDL.NS", "SRF.NS", "BERGEPAINT.NS", "HDFCLIFE.NS",
    "COFORGE.NS", "ADANIENT.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "TATASTEEL.NS"
]

# Stock names mapping
stock_names = {
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ICICIBANK.NS": "ICICI Bank",
    "HDFC.NS": "HDFC Ltd",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ITC.NS": "ITC Ltd",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "LT.NS": "Larsen & Toubro",
    "MARUTI.NS": "Maruti Suzuki",
    "SBIN.NS": "State Bank of India",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "HCLTECH.NS": "HCL Technologies",
    "WIPRO.NS": "Wipro",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "ASIANPAINT.NS": "Asian Paints",
    "TITAN.NS": "Titan Company",
    "ONGC.NS": "Oil & Natural Gas Corp",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "POWERGRID.NS": "Power Grid Corp",
    "JSWSTEEL.NS": "JSW Steel",
    "HDFC-LIFE.NS": "HDFC Life Insurance",
    "DMART.NS": "Avenue Supermarts",
    "NESTLEIND.NS": "Nestle India",
    "HINDALCO.NS": "Hindalco Industries",
    "COALINDIA.NS": "Coal India",
    "BPCL.NS": "Bharat Petroleum",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "DRREDDY.NS": "Dr Reddy's Labs",
    "BHARATFORG.NS": "Bharat Forge",
    "ADANIPORTS.NS": "Adani Ports",
    "ADANIGREEN.NS": "Adani Green Energy",
    "RELIANCEIND.NS": "Reliance Industries",
    "HAVELLS.NS": "Havells India",
    "GAIL.NS": "GAIL India",
    "BHARTI.NS": "Bharti Airtel",
    "ZEEL.NS": "Zee Entertainment",
    "UPL.NS": "UPL Ltd",
    "TATAMOTORS.NS": "Tata Motors",
    "TATACONSUM.NS": "Tata Consumer",
    "BRITANNIA.NS": "Britannia Industries",
    "M&M.NS": "Mahindra & Mahindra",
    "DIVISLAB.NS": "Divis Laboratories",
    "IOC.NS": "Indian Oil Corp",
    "SHREECEM.NS": "Shree Cement",
    "SBILIFE.NS": "SBI Life Insurance",
    "TECHM.NS": "Tech Mahindra",
    "CIPLA.NS": "Cipla",
    "LTIM.NS": "LTI Mindtree",
    "JSWENERGY.NS": "JSW Energy",
    "VEDL.NS": "Vedanta Ltd",
    "SRF.NS": "SRF Ltd",
    "BERGEPAINT.NS": "Berger Paints",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "COFORGE.NS": "Coforge Ltd",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPOWER.NS": "Adani Power",
    "TATAPOWER.NS": "Tata Power",
    "TATASTEEL.NS": "Tata Steel"
}

# App Header
st.markdown('<h1 class="main-header">📈 Indian Stock Forecast Pro</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🇮🇳 Indian Stock Analysis")
    
    # Stock selection
    selected_ticker = st.selectbox(
        "Select Stock:",
        indian_tickers,
        format_func=lambda x: f"{stock_names.get(x, x.split('.')[0])} ({x.split('.')[0]})"
    )
    
    stock_name = stock_names.get(selected_ticker, selected_ticker.split('.')[0])
    
    # Date range
    st.markdown("### 📅 Date Range")
    end_date = date.today()
    start_date = end_date - timedelta(days=3*365)  # 3 years default
    
    col1, col2 = st.columns(2)
    with col1:
        start_date_input = st.date_input("From", start_date)
    with col2:
        end_date_input = st.date_input("To", end_date)
    
    # Ensure end date is after start date
    if end_date_input <= start_date_input:
        st.error("End date must be after start date!")
        end_date_input = start_date_input + timedelta(days=1)
    
    # Forecast settings
    st.markdown("### 🔮 Forecast Settings")
    n_years = st.slider("Years to forecast:", 1, 5, 2)
    period = n_years * 365
    
    confidence_level = st.select_slider(
        "Confidence Level:",
        options=[80, 85, 90, 95],
        value=90
    )
    
    # Actions
    st.markdown("### ⚡ Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("📊 Update Charts", use_container_width=True):
        st.rerun()

# Cache data loading function
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(_ticker, _start_date, _end_date):
    """Load Indian stock data from Yahoo Finance"""
    try:
        # Convert dates to string
        start_str = _start_date.strftime("%Y-%m-%d")
        end_str = _end_date.strftime("%Y-%m-%d")
        
        # Download data
        data = yf.download(
            _ticker,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=True
        )
        
        if data is None or data.empty:
            return None
        
        # Reset index
        data = data.reset_index()
        
        # Ensure proper column names
        if 'Date' not in data.columns:
            # Rename the first column to Date
            data = data.rename(columns={data.columns[0]: 'Date'})
        
        # Convert Date to datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])
        
        # Handle column names (yahoo sometimes uses 'Adj Close' for Indian stocks)
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        elif 'adjclose' in [col.lower() for col in data.columns]:
            for col in data.columns:
                if 'adj' in col.lower():
                    data['Close'] = data[col]
                    break
        
        # Ensure Close column exists
        if 'Close' not in data.columns:
            # Try to find closing price
            for col in data.columns:
                if 'close' in col.lower():
                    data['Close'] = data[col]
                    break
        
        if 'Close' not in data.columns and len(data.columns) > 1:
            # Use last numeric column as Close
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['Close'] = data[numeric_cols[-1]]
        
        # Drop rows without Close price
        data = data.dropna(subset=['Close'])
        
        if data.empty:
            return None
        
        # Sort by date
        data = data.sort_values('Date')
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main content area
st.markdown(f"### 📊 **{stock_name}** ({selected_ticker.split('.')[0]})")

# Load data with progress
with st.spinner(f"Loading {stock_name} data..."):
    data = load_stock_data(selected_ticker, start_date_input, end_date_input)

if data is None or data.empty:
    st.error(f"❌ Could not load data for {selected_ticker}")
    
    # Generate sample data for demo
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
        'Low': prices * 0.985,
        'Volume': np.random.randint(100000, 10000000, len(dates))
    })
    
    st.warning("⚠️ Using simulated data. Real market data unavailable.")

# Display key metrics
st.markdown("### 📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_price = data['Close'].iloc[-1]
    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}",
        help="Latest closing price"
    )

with col2:
    if 'Open' in data.columns:
        daily_change = current_price - data['Open'].iloc[-1]
        change_pct = (daily_change / data['Open'].iloc[-1]) * 100
        st.metric(
            "Today's Change",
            f"₹{daily_change:,.2f}",
            delta=f"{change_pct:.2f}%"
        )
    else:
        st.metric("Total Days", len(data))

with col3:
    start_price = data['Close'].iloc[0]
    total_return_pct = ((current_price - start_price) / start_price) * 100
    st.metric(
        "Total Return",
        f"₹{current_price - start_price:,.2f}",
        delta=f"{total_return_pct:.2f}%"
    )

with col4:
    if len(data) > 1:
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric(
            "Annual Volatility",
            f"{volatility:.1f}%",
            help="Measure of price fluctuation risk"
        )
    else:
        st.metric("Data Points", len(data))

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "🔮 Forecast", "📈 Technicals", "📋 Data"])

with tab1:
    # Price chart
    st.subheader("Price Movement")
    
    # Chart type selector
    chart_col1, chart_col2 = st.columns([3, 1])
    with chart_col2:
        chart_type = st.radio(
            "Chart Type:",
            ["Line", "Candlestick"],
            label_visibility="collapsed"
        )
    
    # Create chart
    if chart_type == "Line":
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name='Closing Price',
            line=dict(color='#FF9933', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 153, 51, 0.1)'
        ))
        
        # Add moving averages
        if len(data) > 50:
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['MA20'],
                mode='lines',
                name='20-Day MA',
                line=dict(color='#138808', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['MA50'],
                mode='lines',
                name='50-Day MA',
                line=dict(color='#000080', width=1)
            ))
    
    else:  # Candlestick
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig = go.Figure(data=[go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                increasing_line_color='#138808',  # Green for up
                decreasing_line_color='#FF3333'   # Red for down
            )])
        else:
            st.warning("OHLC data not available. Showing line chart.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                mode='lines',
                name='Closing Price',
                line=dict(color='#FF9933', width=2)
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{stock_name} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode='x unified',
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart if available
    if 'Volume' in data.columns:
        st.subheader("Trading Volume")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker_color='#1f77b4'
        ))
        fig_volume.update_layout(
            height=300,
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white"
        )
        st.plotly_chart(fig_volume, use_container_width=True)

with tab2:
    # Forecasting
    st.subheader("📈 Price Forecast using Prophet")
    
    if len(data) < 100:
        st.warning(f"⚠️ For better forecasts, use at least 100 days of data. Currently: {len(data)} days")
    
    # Prepare data for Prophet
    df_prophet = data[['Date', 'Close']].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    # Train-test split info
    train_size = int(len(df_prophet) * 0.8)
    test_size = len(df_prophet) - train_size
    
    st.info(f"**Training Data:** {train_size} days | **Test Period:** {test_size} days")
    
    # Forecast button
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Training Prophet model..."):
            try:
                # Initialize Prophet model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    interval_width=confidence_level/100
                )
                
                # Add Indian market holidays
                model.add_country_holidays(country_name='IN')
                
                # Fit model
                model.fit(df_prophet)
                
                # Create future dataframe
                future = model.make_future_dataframe(periods=period)
                
                # Generate forecast
                forecast = model.predict(future)
                
                st.success("✅ Forecast generated successfully!")
                
                # Plot forecast
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(
                    title=f"{stock_name} {n_years}-Year Price Forecast",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    showlegend=True
                )
                
                # Add actual data
                fig_forecast.add_trace(go.Scatter(
                    x=df_prophet['ds'],
                    y=df_prophet['y'],
                    mode='markers',
                    name='Actual Data',
                    marker=dict(color='blue', size=4)
                ))
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast summary
                st.subheader("📊 Forecast Summary")
                
                forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
                
                with forecast_col1:
                    last_actual = df_prophet['y'].iloc[-1]
                    st.metric("Last Actual Price", f"₹{last_actual:,.2f}")
                
                with forecast_col2:
                    forecast_price = forecast['yhat'].iloc[-1]
                    forecast_change = ((forecast_price - last_actual) / last_actual) * 100
                    st.metric(
                        f"Forecast Price ({n_years} years)",
                        f"₹{forecast_price:,.2f}",
                        delta=f"{forecast_change:.1f}%"
                    )
                
                with forecast_col3:
                    lower_bound = forecast['yhat_lower'].iloc[-1]
                    upper_bound = forecast['yhat_upper'].iloc[-1]
                    st.metric(
                        "Forecast Range",
                        f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}",
                        help=f"{confidence_level}% confidence interval"
                    )
                
                # Show forecast components
                with st.expander("📈 View Forecast Components"):
                    fig_components = model.plot_components(forecast)
                    st.pyplot(fig_components)
                
                # Forecast data table
                with st.expander("📋 View Forecast Data"):
                    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
                    forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                    forecast_display['Forecast'] = forecast_display['Forecast'].apply(lambda x: f"₹{x:,.2f}")
                    forecast_display['Lower Bound'] = forecast_display['Lower Bound'].apply(lambda x: f"₹{x:,.2f}")
                    forecast_display['Upper Bound'] = forecast_display['Upper Bound'].apply(lambda x: f"₹{x:,.2f}")
                    
                    st.dataframe(
                        forecast_display,
                        use_container_width=True,
                        hide_index=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Forecast failed: {str(e)}")
                st.info("Try with more historical data or different settings.")
    else:
        st.info("Click the button above to generate forecast")

with tab3:
    # Technical Analysis
    st.subheader("📊 Technical Indicators")
    
    if len(data) > 20:
        # Calculate indicators
        data_tech = data.copy()
        
        # Moving Averages
        data_tech['MA20'] = data_tech['Close'].rolling(window=20).mean()
        data_tech['MA50'] = data_tech['Close'].rolling(window=50).mean()
        data_tech['MA200'] = data_tech['Close'].rolling(window=200).mean()
        
        # RSI
        delta = data_tech['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data_tech['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data_tech['BB_Middle'] = data_tech['Close'].rolling(window=20).mean()
        bb_std = data_tech['Close'].rolling(window=20).std()
        data_tech['BB_Upper'] = data_tech['BB_Middle'] + (bb_std * 2)
        data_tech['BB_Lower'] = data_tech['BB_Middle'] - (bb_std * 2)
        
        # Display current values
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            ma_signal = "Bullish" if data_tech['MA20'].iloc[-1] > data_tech['MA50'].iloc[-1] else "Bearish"
            st.metric("MA Signal", ma_signal)
        
        with tech_col2:
            current_rsi = data_tech['RSI'].iloc[-1]
            rsi_status = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
            st.metric("RSI", f"{current_rsi:.1f}", rsi_status)
        
        with tech_col3:
            bb_position = (data_tech['Close'].iloc[-1] - data_tech['BB_Lower'].iloc[-1]) / \
                         (data_tech['BB_Upper'].iloc[-1] - data_tech['BB_Lower'].iloc[-1]) * 100
            st.metric("BB Position", f"{bb_position:.1f}%")
        
        with tech_col4:
            volatility = data_tech['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Volatility", f"{volatility:.1f}%")
        
        # Plot technical indicators
        st.subheader("RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data_tech['Date'],
            y=data_tech['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=50, line_dash="dash", line_color="gray")
        fig_rsi.update_layout(height=300, xaxis_title="Date", yaxis_title="RSI")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Bollinger Bands
        st.subheader("Bollinger Bands")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=data_tech['Date'],
            y=data_tech['BB_Upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='gray', width=1)
        ))
        fig_bb.add_trace(go.Scatter(
            x=data_tech['Date'],
            y=data_tech['BB_Middle'],
            mode='lines',
            name='Middle Band',
            line=dict(color='blue', width=1)
        ))
        fig_bb.add_trace(go.Scatter(
            x=data_tech['Date'],
            y=data_tech['BB_Lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ))
        fig_bb.add_trace(go.Scatter(
            x=data_tech['Date'],
            y=data_tech['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#FF9933', width=2)
        ))
        fig_bb.update_layout(height=400, xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig_bb, use_container_width=True)
        
    else:
        st.warning("Need at least 20 days of data for technical analysis")

with tab4:
    # Data tab
    st.subheader("📋 Historical Data")
    
    # Data summary
    st.markdown(f"""
    <div class="metric-card">
        <strong>Data Summary:</strong><br>
        • Period: {data['Date'].min().strftime('%d %b %Y')} to {data['Date'].max().strftime('%d %b %Y')}<br>
        • Trading Days: {len(data):,}<br>
        • Price Range: ₹{data['Close'].min():,.2f} - ₹{data['Close'].max():,.2f}<br>
        • Average Price: ₹{data['Close'].mean():,.2f}
    </div>
    """, unsafe_allow_html=True)
    
    # Data preview
    st.subheader("Data Preview")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_rows = st.slider("Rows to show:", 5, 100, 20)
    with col2:
        sort_order = st.selectbox("Sort by:", ["Latest First", "Oldest First"])
    
    # Prepare data for display
    display_data = data.copy()
    display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
    
    # Format numeric columns
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in display_data.columns:
            display_data[col] = display_data[col].apply(lambda x: f"₹{x:,.2f}")
    
    if 'Volume' in display_data.columns:
        display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x:,}")
    
    # Sort data
    if sort_order == "Latest First":
        display_data = display_data.iloc[::-1]
    
    # Display table
    st.dataframe(
        display_data.head(show_rows),
        use_container_width=True,
        hide_index=True
    )
    
    # Download options
    st.subheader("Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_ticker.split('.')[0]}_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel download
        @st.cache_data
        def convert_to_excel(df):
            output = pd.ExcelWriter('temp.xlsx', engine='openpyxl')
            df.to_excel(output, index=False)
            output.close()
            with open('temp.xlsx', 'rb') as f:
                return f.read()
        
        excel_data = convert_to_excel(data)
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"{selected_ticker.split('.')[0]}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>🇮🇳 <strong>Indian Stock Forecast Pro</strong> • NSE/BSE Data via Yahoo Finance • Powered by Prophet</p>
    <p><em>📊 Data as of {datetime.now().strftime('%d %B %Y, %I:%M %p')} • For educational purposes only</em></p>
    <p><small>⚠️ <strong>Disclaimer:</strong> This is not investment advice. Always do your own research.</small></p>
</div>
""", unsafe_allow_html=True)
