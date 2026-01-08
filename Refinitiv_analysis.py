"""
This script performs rolling forecasts on Realized Variance (RV) data using the TimesFM 2.5 model.
It uses the Refinitiv dataset that was constructed with the script `full_clean_combine_ver_2.py`.
Key Features:
1. **Configuration**: Allows customization of input data path, window size, forecast horizon, and output file name.
2. **Data Loading**: Reads and preprocesses the input CSV file containing Realized Variance data.
3. **Model Initialization**: Loads and configures the TimesFM 2.5 model for forecasting.
4. **Rolling Forecast Loop**: Iterates over each RV column, generates forecasts using a rolling window approach, 
    and stores the results in a structured format.
5. **Output Formatting**: Converts the forecast results into a pivoted DataFrame, optionally joins with actual values, 
    and saves the output to a CSV file called "TimesFM_Forecasts_Context_[WINDOW_SIZE].csv".
"""
import numpy as np
import pandas as pd
import timesfm
from tqdm import tqdm



# --- 1. CONFIGURATION ---
DATA_PATH = "/Users/santtunevalainen/Desktop/Koulu/Gradu/combined_data/Refinitiv_manual_RV_py.csv"
WINDOW_SIZE = 128  # Change this to 64, 128, 256 etc., for future runs
OUTPUT_FILE = f"TimesFM_Forecasts_Context_{WINDOW_SIZE}.csv"
HORIZON = 1

# --- 2. LOAD DATA ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Identify all Realized Variance columns
rv_columns = [col for col in df.columns if "RV_" in col]

# --- 3. INITIALIZE MODEL ---
print(f"Initializing TimesFM 2.5 with Context Length {WINDOW_SIZE}...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", 
    torch_compile=False
)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=HORIZON,
        normalize_inputs=True,
        infer_is_positive=True,
    )
)

# --- 4. ROLLING FORECAST LOOP ---
# Using a dictionary to store lists of forecasts for each index
forecast_results = []

for col in rv_columns:
    print(f"\n>>> Processing {col}")
    series_df = df[col].dropna()
    series_values = series_df.values
    series_dates = series_df.index
    
    if len(series_values) <= WINDOW_SIZE:
        continue

    for i in tqdm(range(WINDOW_SIZE, len(series_values)), desc=f"Forecasting {col}"):
        context = series_values[i - WINDOW_SIZE : i]
        
        point_forecast, _ = model.forecast(
            horizon=HORIZON,
            inputs=[context]
        )
        
        # We save the Date and a dynamically named column
        # Example: fc_timesfm_64_AEX_RV_day (fc indicating forecast)
        forecast_results.append({
            "Date": series_dates[i],
            "Column": f"fc_timesfm_{WINDOW_SIZE}_{col}",
            "Value": point_forecast[0, 0]
        })

# --- 5. FORMAT & SAVE ---
# Convert list of dicts to DataFrame
raw_df = pd.DataFrame(forecast_results)

# Pivot the data so Dates are rows and 'fc_timesfm_64_IndexName' are columns
final_df = raw_df.pivot(index="Date", columns="Column", values="Value")

# Optional: Join with original 'Actual' values for easy comparison
final_df = final_df.join(df[rv_columns], rsuffix='_actual')

final_df.to_csv(OUTPUT_FILE)

print("\n" + "="*30)
print(f"SUCCESS: Results saved to {OUTPUT_FILE}")
print(f"Format: Columns are named 'timesfm_{WINDOW_SIZE}_[Index]'")
print("="*30)