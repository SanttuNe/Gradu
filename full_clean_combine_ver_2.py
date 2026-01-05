import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

#This script reads and combines multiple excel files containing stock market index data from Refinitiv into
# a single cleaned CSV file.
#Note: Google Gemini 3.0 was used to help write this code (mostly with the read_index_file and main functions)

# Configuration

load_dotenv()
RAW_BASE = Path(os.getenv("RAW_DATA_PATH", "./RAW_DATA"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR_PATH", "./combined_data"))
OUTPUT_NAME = "data_combined.csv"

#Select these indexes
INDEXES = [".AEX", ".MXX", ".RUT", ".SSMI", ".BVSP", ".FTSE", ".IXIC", ".HSI", ".GDAXI", ".FCHI"]

# Folder and Filename mapping
# Format: (Folder_Path, Filename_Template)
SOURCE_CONFIGS = [
    (RAW_BASE / "2024", "{sym}_2024.xlsx"),
    (RAW_BASE / "2025 PARTIAL", "{sym}_2025.xlsx"),
    (RAW_BASE / "2025 END", "{sym}_2025_END.xlsx"),
]

#Choose columns to keep
KEEP_COLS = ["Exchange Date", "Exchange Time", "Local Time", "Close", "Net", "%Chg", "Open", "Low", "High"]

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
    
    # filter columns, add Symbol, and convert dates
    present_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[present_cols].copy()
    df.insert(0, "Symbol", ticker)
    
    if "Exchange Date" in df.columns:
        df["Exchange Date"] = pd.to_datetime(df["Exchange Date"], errors='coerce')
        
    return df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

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
            else:
                print(f"[SKIP] Missing: {file_path}")

    if not all_dfs:
        print("No files were found. Check your RAW_BASE path.")
        return

    # combine and sort
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Sort: Symbol ascending, Date descending
    combined = combined.sort_values(["Symbol", "Exchange Date"], ascending=[True, False])

    # Save
    out_path = OUTPUT_DIR / OUTPUT_NAME
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined):,} rows to: {out_path}")

if __name__ == "__main__":
    main()