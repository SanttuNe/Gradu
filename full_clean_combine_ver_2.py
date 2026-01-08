import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

#This script reads and combines multiple excel files containing stock market index data from Refinitiv into
# a single cleaned CSV file and calculates the realized variance for each index and day.
#Note: Google Gemini 3.0 was used to help write this code (mostly with the read_index_file and main functions)

# CONFIGURATION 
load_dotenv()
RAW_BASE = Path(os.getenv("RAW_DATA_PATH", "./RAW_DATA"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR_PATH", "./combined_data"))
COMBINED_NAME = "data_combined.csv"
WIDE_NAME = "refinitiv_wide.csv"
RV_NAME = "PYTHON_Refinitiv_manual_RV.csv"
LATEX_OUT = "Index_Summary.tex"

# Select these indexes
INDEXES = [".AEX", ".MXX", ".RUT", ".SSMI", ".BVSP", ".FTSE", ".IXIC", ".HSI", ".GDAXI", ".FCHI"]

#Folder and filename mapping
#Format: (Folder_Path, Filename_Template)
SOURCE_CONFIGS = [
    (RAW_BASE / "2024", "{sym}_2024.xlsx"),
    (RAW_BASE / "2025 PARTIAL", "{sym}_2025.xlsx"),
    (RAW_BASE / "2025 END", "{sym}_2025_END.xlsx"),
]
#Choose columns to keep
KEEP_COLS = ["Exchange Date", "Exchange Time", "Local Time", "Close", "Open", "Low", "High"]

# FUNCTIONS

def read_index_file(path: Path, ticker: str) -> pd.DataFrame:
    """Detects header, reads the file, and cleans columns."""
    # Detect header row (checking first 60 rows)
    preview = pd.read_excel(path, header=None, nrows=60, engine="openpyxl")
    keywords = {"exchange date", "exchange time", "close", "open", "high", "low"}
    
    header_idx = 0
    for i, row in preview.iterrows():
        vals = [str(v).lower() for v in row if pd.notna(v)]
        if sum(any(kw in val for kw in keywords) for val in vals) >= 4:
            header_idx = i
            break
    # Read full file starting from the detected header
    df = pd.read_excel(path, header=header_idx, engine="openpyxl")
    
    # Filter columns and clean ticker (Remove dots)
    present_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[present_cols].copy()
    df.insert(0, "Symbol", ticker.lstrip("."))
    
    # Date conversions
    if "Exchange Date" in df.columns:
        df["Exchange Date"] = pd.to_datetime(df["Exchange Date"], errors='coerce')
    if "Local Time" in df.columns:
        df["Local Time"] = pd.to_datetime(df["Local Time"], errors='coerce')
        
    return df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    # READ AND COMBINE 
    for ticker in INDEXES:
        sym = ticker.lstrip(".")
        for folder, pattern in SOURCE_CONFIGS:
            file_path = folder / pattern.format(sym=sym)
            if file_path.exists():
                print(f"[OK] Processing {ticker} from {file_path.name}")
                try:
                    all_dfs.append(read_index_file(file_path, ticker))
                except Exception as e:
                    print(f"[ERROR] Could not read {file_path}: {e}")

    if not all_dfs:
        print("No files found.")
        return

    refinitiv_data = pd.concat(all_dfs, ignore_index=True)
    refinitiv_data = refinitiv_data.sort_values(["Symbol", "Exchange Date"], ascending=[True, False])
    refinitiv_data.to_csv(OUTPUT_DIR / COMBINED_NAME, index=False)

    # PIVOT WIDER
    # Using 'Local Time' as the primary time index
    refinitiv_wide = refinitiv_data.pivot(
        index="Local Time", 
        columns="Symbol", 
        values=["Open", "Close", "High", "Low"]
    )
    # Flatten multi-index columns: Symbol_Value (e.g., AEX_Close)
    refinitiv_wide.columns = [f"{col[1]}_{col[0]}" for col in refinitiv_wide.columns]
    refinitiv_wide = refinitiv_wide.sort_index()
    refinitiv_wide.to_csv(OUTPUT_DIR / WIDE_NAME)

    # CALCULATE REALIZED VARIANCE 
    print("Calculating Realized Variance...")
    # Work with long format for RV calculation (more efficient in pandas)
    rv_data = refinitiv_data.copy()
    rv_data['Date'] = rv_data['Local Time'].dt.date
    rv_data = rv_data.sort_values(['Symbol', 'Local Time'])
    
    # Calculate log returns per Symbol per Day
    rv_data['log_ret'] = rv_data.groupby(['Symbol', 'Date'])['Close'].transform(lambda x: np.log(x) - np.log(x.shift(1)))
    
    # Sum squared log returns (Realized Variance)
    manual_rv_results = rv_data.groupby(['Symbol', 'Date'])['log_ret'].apply(lambda x: (x**2).sum()).reset_index()
    manual_rv_results.rename(columns={'log_ret': 'Manual_RV'}, inplace=True)
    
    # Pivot to wide RV format
    rv_wide = manual_rv_results.pivot(index='Date', columns='Symbol', values='Manual_RV')
    rv_wide.columns = [f"RV_{col}" for col in rv_wide.columns]
    rv_wide.to_csv(OUTPUT_DIR / RV_NAME)

    # SUMMARY STATISTICS 
    # Raw observation counts
    obs_counts = refinitiv_data.groupby("Symbol").size().reset_index(name="Raw_Obs")
    
    # RV day counts (exclude days with 0 RV if they represent missing data)
    rv_counts = manual_rv_results[manual_rv_results['Manual_RV'] > 0].groupby("Symbol").size().reset_index(name="RV_Obs")
    
    combined_summary = pd.merge(obs_counts, rv_counts, on="Symbol", how="outer").sort_values("Raw_Obs", ascending=False)
    
    print("\n--- Data Availability Summary ---")
    print(combined_summary.to_string(index=False))

    #EXPORT TO LATEX
    combined_summary.to_latex(
        OUTPUT_DIR / LATEX_OUT, 
        index=False, 
        caption="Summary of Data Observations and Realized Variance Days",
        label="tab:obs_summary"
    )
    print(f"\n[DONE] Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()