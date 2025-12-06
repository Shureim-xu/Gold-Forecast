# resave_scalers.py
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration (Must match your app.py) ---
MARKET_FILE = "XAU_1d_data.csv"  # Ensure this file is in the same directory
FEATURES = ['Open','High','Low','Close','Volume','Return','MA_7','MA_21','Vol_7']
SCALER_X_PATH = "scaler_X.pkl"   # Output file for input features (X)
SCALER_y_PATH = "scaler_y.pkl"   # Output file for target variable (y)
# -----------------------------------------------

def create_features(df):
    """
    Replicates the exact feature engineering steps from your model training.
    """
    print("Creating features...")
    df['Return'] = df['Close'].pct_change()
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['MA_21'] = df['Close'].rolling(21).mean()
    df['Vol_7'] = df['Return'].rolling(7).std()
    
    # Drop initial NaNs created by rolling windows/returns
    df_processed = df.dropna().reset_index(drop=True)
    return df_processed

def resave_scalers():
    if not os.path.exists(MARKET_FILE):
        print(f"Error: Market file '{MARKET_FILE}' not found. Please ensure it is in the correct folder.")
        return

    try:
        # 1. Load and Process Data
        # Using the same loading logic as the app for robustness
        df = pd.read_csv(MARKET_FILE, sep=";")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={
            'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        }, errors='ignore')
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
        
        df_processed = create_features(df)
        
        # 2. Define X and y for Scaling
        X_data = df_processed[FEATURES].values
        # For sequence models, the target y is often the Close price itself
        # scaled separately. We assume 'Close' is the variable to be predicted.
        y_data = df_processed[['Close']].values 
        
        print(f"Data ready. X shape: {X_data.shape}, y shape: {y_data.shape}")

        # 3. Fit New Scalers
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        scaler_X.fit(X_data)
        scaler_y.fit(y_data)
        
        print("Scalers successfully fitted with the new scikit-learn version.")

        # 4. Save Scalers
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_y, SCALER_y_PATH)
        
        print("-" * 30)
        print(f"✅ Success! New scalers saved:")
        print(f"   - Input Scaler (X): {SCALER_X_PATH}")
        print(f"   - Target Scaler (y): {SCALER_y_PATH}")
        print("The InconsistentVersionWarning should now be resolved.")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred during scaler resaving: {e}")

if __name__ == "__main__":
    resave_scalers()