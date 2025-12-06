import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import joblib
import os
import time
from datetime import timedelta, datetime 
from tensorflow.keras.models import load_model


# CONFIGURATION
LOG_FILE = "logs.csv"            # Prediction log file
MARKET_FILE = "XAU_1d_data.csv"  # Historical OHLC CSV

# ML Model Configuration
[cite_start]SEQ_HORIZON = 7                  # Number of predicted days [cite: 2]
[cite_start]SEQ_LEN = 60                     # Model lookback (input sequence length) [cite: 2]
[cite_start]MAX_MARKET_ROWS = 1500           # Limit total rows read for large historical files [cite: 2]
FEATURES = ['Open','High','Low','Close','Volume','Return','MA_7','MA_21','Vol_7']

# File paths
MODEL_CANDIDATES = ["xau_seq2seq_7d.h5"]
SCALER_X_CANDIDATES = ["scaler_X.pkl"]
SCALER_y_CANDIDATES = ["scaler_y.pkl"]

st.set_page_config(page_title="Gold Forecast • Premium Dashboard", layout="wide", page_icon="🟡")

# Gradient header and initial setup
st.markdown(
    """
    <style>
    .grad-header {
    background: linear-gradient(90deg,#0f172a,#D4AF37 60%);
    padding: 22px;
    border-radius: 8px;
    color: white;
    }
    .small {
    color: #e6e6e6;
    font-size:14px;
    margin-top:6px;
    }
    </style>
    <div class="grad-header">
    <h1 style="margin:0">💰 XAU/USD Forecast Dashboard</h1>
    <div class="small">Candlesticks • 7 day seq forecasts • Confidence gauge • Interactive filters</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

# ML Resource and Data Loading (Cached)

def _find_file(candidates):
    """Helper to find the first available file from a list of candidates."""
    for p in candidates:
        if os.path.exists(p):
            return p
    [cite_start]return None [cite: 5]

@st.cache_data(ttl=600)
def load_market(path=MARKET_FILE):
    """Loads and preprocesses market data."""
    [cite_start]if not os.path.exists(path): return None [cite: 5]
    try:
        df = pd.read_csv(path, sep=";")
        df.columns = [c.strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            [cite_start]if c.lower() in ("date", "time", "timestamp"): col_map[c] = "Date" [cite: 6]
            [cite_start]elif c.lower() == "open": col_map[c] = "Open" [cite: 6]
            [cite_start]elif c.lower() == "high": col_map[c] = "High" [cite: 6]
            [cite_start]elif c.lower() == "low": col_map[c] = "Low" [cite: 6]
            [cite_start]elif c.lower() == "close": col_map[c] = "Close" [cite: 6]
            [cite_start]elif c.lower() == "volume": col_map[c] = "Volume" [cite: 6]

        if "Date" not in col_map.values() or "Close" not in col_map.values(): return None

        [cite_start]df = df.rename(columns=col_map) [cite: 7]
        [cite_start]df["Date"] = pd.to_datetime(df["Date"], errors='coerce') [cite: 7]
        [cite_start]df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True) [cite: 7]
        
        if len(df) > MAX_MARKET_ROWS:
            df = df.tail(MAX_MARKET_ROWS).reset_index(drop=True)

        return df
    except Exception:
        return None

@st.cache_data(ttl=300)
def load_logs(path=LOG_FILE):
    """Loads log data and expands pipe-separated predictions."""
    [cite_start]if not os.path.exists(path): return None [cite: 8]
    try:
        logs = pd.read_csv(path)
        logs.columns = [c.strip() for c in logs.columns]
        
        if 'timestamp' in logs.columns:
            [cite_start]logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors='coerce') [cite: 8]
            [cite_start]logs = logs.dropna(subset=['timestamp']) [cite: 8]
        else:
            [cite_start]logs['timestamp'] = pd.NaT [cite: 9]

        if 'predictions' in logs.columns and not logs.empty:
            [cite_start]preds = logs['predictions'].astype(str).str.split('|', expand=True) [cite: 9]
            [cite_start]preds = preds.iloc[:, :SEQ_HORIZON] [cite: 9]
            [cite_start]preds.columns = [f'h+{i+1}' for i in range(preds.shape[1])] [cite: 9]
            [cite_start]preds = preds.apply(pd.to_numeric, errors='coerce') [cite: 9]
            
            [cite_start]logs = logs.reset_index(drop=True) [cite: 10]
            [cite_start]preds = preds.reset_index(drop=True) [cite: 10]
            
            [cite_start]logs = pd.concat([logs, preds], axis=1) [cite: 10]
            
        return logs
    except Exception:
        return None

@st.cache_resource
def load_ml_resources():
    """
    Find and load model and scalers.
    
    NOTE: The path assignments MUST be correctly indented inside this function.
    """
    # 🐛 FIX: Ensure these lines are present and correctly indented to define the variables
    [cite_start]model_path = _find_file(MODEL_CANDIDATES) [cite: 11]
    [cite_start]scaler_x_path = _find_file(SCALER_X_CANDIDATES) [cite: 11]
    [cite_start]scaler_y_path = _find_file(SCALER_y_CANDIDATES) [cite: 11]

    if not all([model_path, scaler_x_path, scaler_y_path]):
        # Log to deployment console if files are not found on disk
        print("LOG: ML resource files not found on disk.")
        return None, None, None, False

    try:
        model = load_model(model_path, compile=False)
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_X, scaler_y, True
    except Exception as e:
        # 💡 CRITICAL ADDITION: Log the actual loading failure reason (e.g., Keras/TF error)
        print(f"CRITICAL LOG: Failed to load ML resources. Exception: {e}")
        return None, None, None, False


# Load all resources globally
[cite_start]market_df = load_market() [cite: 12]
logs = load_logs()
MODEL, SCALER_X, SCALER_y, ML_READY = load_ml_resources()

# 📝 LOGGING UTILITY
def log_prediction(pred_list, latency):
    """Logs the prediction result to the CSV file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "predictions": "|".join([f"{p:.6f}" for p in pred_list]),
        "latency": float(latency)
    }
    row = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        [cite_start]row.to_csv(LOG_FILE, mode="a", header=False, index=False) [cite: 13]
    else:
        [cite_start]row.to_csv(LOG_FILE, mode="w", header=True, index=False) [cite: 13]

# 🔮 PREDICTION BUTTON AND LOGIC
if ML_READY and market_df is not None and not market_df.empty:
    st.subheader("Run Forecast")
    
    predict_col, info_col = st.columns([1, 4])
    
    with predict_col:
        if st.button("📈 Predict Next 7 Days", type="primary"):
            with st.spinner(f"Running Seq2Seq prediction for {SEQ_HORIZON} days..."):
                try:
                    [cite_start]t0 = time.time() [cite: 14]
                    
                    # 1. Feature Engineering (must match training pipeline exactly)
                    df2 = market_df.copy()
                    [cite_start]df2['Return'] = df2['Close'].pct_change() [cite: 15]
                    [cite_start]df2['MA_7'] = df2['Close'].rolling(7).mean() [cite: 15]
                    [cite_start]df2['MA_21'] = df2['Close'].rolling(21).mean() [cite: 15]
                    [cite_start]df2['Vol_7'] = df2['Return'].rolling(7).std() [cite: 15]
                    [cite_start]df2 = df2.dropna().reset_index(drop=True) [cite: 16]
        
                    
                    # 2. Check Data Sufficiency
                    if len(df2) < SEQ_LEN:
                        st.error(f"Need {SEQ_LEN} days of data, found {len(df2)}.")
                        [cite_start]st.stop() [cite: 17]
                        
                    # 3. Scale and Reshape
                    [cite_start]last_window = df2[FEATURES].tail(SEQ_LEN).values.astype(float) [cite: 17]
                    [cite_start]last_window_scaled = SCALER_X.transform(last_window) [cite: 18]
                    seq = last_window_scaled.reshape(1, SEQ_LEN, last_window_scaled.shape[1])
                    
                    # 4. Predict
                    [cite_start]pred_scaled = MODEL.predict(seq, verbose=0) [cite: 19]
                    [cite_start]pred_scaled = pred_scaled.reshape(SEQ_HORIZON, 1) [cite: 19]
                    [cite_start]pred_raw = SCALER_y.inverse_transform(pred_scaled).reshape(-1) [cite: 19]
                    latency = round(time.time() - t0, 4)

                    # 5. Log and Refresh
                    [cite_start]log_prediction(pred_raw.tolist(), latency) [cite: 20]
                    load_logs.clear() # Clear cache to force dashboard refresh
                    
                    [cite_start]st.success(f"Prediction successful! Latency: {latency}s. Dashboard is refreshed.") [cite: 21]
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    
    with info_col:
        [cite_start]st.info(f"Last market date used: **{market_df['Date'].max().strftime('%Y-%m-%d')}**. Prediction horizon: **{SEQ_HORIZON} days**.") [cite: 22]
    
    st.markdown("---")
else:
    st.warning("ML Model or Data not fully loaded. Prediction button is disabled.")
    if market_df is None:
        st.warning(f"Check if `{MARKET_FILE}` exists.")
    if not ML_READY:
        st.warning("Check if model and scaler files (`.h5`, `.pkl`) are present.")

# Sidebar filters (FIXED the logic to ensure variables are defined)
st.sidebar.header("Filters & Controls")
# Initialize with safe defaults
start_date, end_date = None, None 

if market_df is not None and not market_df.empty:
    [cite_start]min_date = market_df["Date"].min().date() [cite: 23]
    [cite_start]max_date = market_df["Date"].max().date() [cite: 23]
    [cite_start]date_range_default = (min_date, max_date) [cite: 23]
else:
    min_date = None
    max_date = None
    date_range_default = (None, None)

date_range = st.sidebar.date_input("Market Date Range", value=date_range_default)

# Check if date_range has 2 elements and neither is None
dates_are_valid = (len(date_range) == 2 and date_range[0] is not None and date_range[1] is not None)

if dates_are_valid:
    # Safely assign the variables only when valid
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    [cite_start]st.sidebar.error("Select a valid start and end date for the market chart.") [cite: 24]
    # start_date and end_date remain None if invalid, preventing downstream errors

st.sidebar.markdown("---")
# log filtering range (logs)
if logs is not None and not logs.empty:
    log_min = logs['timestamp'].min().date()
    log_max = logs['timestamp'].max().date()
    log_from, log_to = st.sidebar.date_input("Log Timestamp Range", value=(log_min, log_max)) 
else:
    log_from, log_to = None, None

st.sidebar.markdown("---")
st.sidebar.write("Display options")
show_candles = st.sidebar.checkbox("Show candlestick (market)", value=True)
show_predictions_overlay = st.sidebar.checkbox("Overlay latest predictions", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Confidence gauge uses prediction spread (narrower = higher confidence).")
st.sidebar.markdown("")

# Dashboard Sections
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("XAU/USD PRICE CHART")
    if market_df is None:
        [cite_start]st.info("Market CSV not found (XAU_1d_data.csv).") [cite: 25]
    elif not show_candles:
        st.info("Candlestick chart disabled by sidebar control.")
    elif start_date is None or end_date is None: # NEW CHECK
        st.warning("Cannot display chart: Invalid date range selected.")
    else:
        # Code now safely uses start_date and end_date
        mdf = market_df[(market_df["Date"] >= start_date) & (market_df["Date"] <= end_date)].copy()
        
        [cite_start]if len(mdf) > 800: [cite: 26]
            [cite_start]mdf = mdf.tail(800) [cite: 26]
            
        if mdf.empty:
            st.warning("No market rows in selected date range.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                [cite_start]x=mdf['Date'], [cite: 27]
                [cite_start]open=mdf['Open'], [cite: 27]
                [cite_start]high=mdf['High'], [cite: 27]
                [cite_start]low=mdf['Low'], [cite: 27]
                [cite_start]close=mdf['Close'], [cite: 27]
                [cite_start]name='OHLC' [cite: 28]
            ))
            current_logs = load_logs() 
            if show_predictions_overlay and current_logs is not None and not current_logs.empty:
                last_log = current_logs.sort_values('timestamp').iloc[-1]
                preds = [last_log.get(f'h+{i+1}', np.nan) for i in range(SEQ_HORIZON)] 
                [cite_start]last_market_date = market_df['Date'].max() [cite: 28]
                [cite_start]future_dates = [last_market_date + timedelta(days=i) for i in range(1, SEQ_HORIZON+1)] [cite: 29]
                fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name='Latest 7-day Prediction', line=dict(color='#D4AF37')))
            
            fig.update_layout(height=520, margin=dict(l=0,r=0,t=30,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Quick KPIs")
    current_logs = load_logs()
    
    [cite_start]if market_df is not None and not market_df.empty: [cite: 30]
        [cite_start]latest_close = market_df['Close'].iloc[-1] [cite: 30]
        [cite_start]st.metric("Latest Close", f"${latest_close:,.2f}") [cite: 30]
    else:
        st.metric("Latest Close", "N/A")
    
    if current_logs is not None and not current_logs.empty:
        recent = current_logs.sort_values('timestamp').iloc[-1]
        pred_cols = [f'h+{i+1}' for i in range(SEQ_HORIZON)]
        preds = recent[pred_cols].dropna().values
        
        [cite_start]if preds.size > 0: [cite: 31]
            [cite_start]mean_pred = preds.mean() [cite: 31]
            [cite_start]spread = preds.max() - preds.min() [cite: 31]
            [cite_start]st.metric("Latest Pred Mean", f"${mean_pred:,.2f}") [cite: 31]
            [cite_start]st.metric("Pred Spread (max-min)", f"${spread:,.2f}") [cite: 31]
        else:
            st.metric("Latest Pred Mean", "N/A")
            [cite_start]st.metric("Pred Spread (max-min)", "N/A") [cite: 32]
    else:
        st.metric("Latest Pred Mean", "N/A")
        st.metric("Pred Spread (max-min)", "N/A")

#Confidence Gauge, Latency, Snapshot
st.markdown("---")
g1, g2, g3 = st.columns([1,2,2])

with g1:
    st.subheader("Confidence Gauge")
    current_logs = load_logs()
    if current_logs is None or current_logs.empty:
        st.info("No prediction logs to compute confidence.")
    else:
        r = current_logs.sort_values('timestamp').iloc[-1]
        [cite_start]pred_cols = [f'h+{i+1}' for i in range(SEQ_HORIZON)] [cite: 33]
        [cite_start]preds = r[pred_cols].dropna().values [cite: 33]
        if len(preds) == 0:
            st.info("No numeric predictions found.")
        else:
            spread = preds.max() - preds.min()
            mean_p = preds.mean() if preds.mean() != 0 else 1.0
            [cite_start]conf = max(0.0, 1 - (spread / mean_p)) [cite: 34]
            conf_pct = round(conf * 100, 1)
            gauge = go.Figure(go.Indicator(mode="gauge+number",value=conf_pct,domain={'x': [0, 1], 'y': [0, 1]},title={'text': "Confidence (%)"},
                gauge={'axis': {'range': [0, 100]},'bar': {'color': "#D4AF37"},'steps': [{'range': [0, 33], 'color': "red"},{'range': [33, 66], 'color': "yellow"},{'range': [66, 100], 'color': "green"}]}))
            gauge.update_layout(height=240, margin=dict(t=10,b=10,l=10,r=10))
            [cite_start]st.plotly_chart(gauge, use_container_width=True) [cite: 35]

with g2:
    st.subheader("Recent Latencies")
    current_logs = load_logs()
    if current_logs is not None and not current_logs.empty:
        if log_from and log_to:
            filtered_logs = current_logs[(current_logs['timestamp'].dt.date >= log_from) & (current_logs['timestamp'].dt.date <= log_to)]
        else:
            filtered_logs = current_logs
        st.bar_chart(filtered_logs.sort_values('timestamp', ascending=False).head(20).set_index('timestamp')['latency'])
    else:
        st.info("No logs.")

with g3:
    [cite_start]st.subheader("Prediction Snapshot (last log)") [cite: 36]
    current_logs = load_logs()
    if current_logs is not None and not current_logs.empty:
        last = current_logs.sort_values('timestamp').iloc[-1]
        pred_cols = [f'h+{i+1}' for i in range(SEQ_HORIZON)]
        pred_vals = last[pred_cols].tolist()
        pred_df = pd.DataFrame({"horizon": [f"h+{i+1}" for i in range(SEQ_HORIZON)], "pred": pred_vals})
        st.table(pred_df.style.format({"pred":"${:.2f}"}))
    else:
        st.info("No logs to show.")

[cite_start]#Log History Button Section [cite: 37]
st.markdown("---")
st.subheader("Prediction History")

if st.button("Click to View All Prediction Logs"):
    current_logs = load_logs()
    if current_logs is not None and not current_logs.empty:
        display_cols = ['timestamp', 'latency'] + [f'h+{i+1}' for i in range(SEQ_HORIZON)]
        existing_cols = [col for col in display_cols if col in current_logs.columns]
        
        st.dataframe(
            current_logs[existing_cols]
            [cite_start].sort_values('timestamp', ascending=False) [cite: 38]
            .head(100) 
            .style.format({f'h+{i+1}': "${:.2f}" for i in range(SEQ_HORIZON)}),
            use_container_width=True
        )
    else:
        st.info("No prediction logs found in logs.csv.")

# Footer
st.markdown("---")
[cite_start]st.markdown("<div style='text-align:center;color:gray;'>Built with❤️ for data-driven gold insights.</div>", unsafe_allow_html=True) [cite: 39]
