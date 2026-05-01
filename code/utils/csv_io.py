import pandas as pd
import os

def read_tickets(file_path):
    """
    Reads the input CSV. Handles both sample and real formats.
    Expects columns like issue, subject, company.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    df = pd.read_csv(file_path, engine="python", encoding="utf-8", on_bad_lines="warn")
    # Lowercase column names to avoid case sensitivity issues
    df.columns = [col.strip().lower() for col in df.columns]
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "issue": str(row.get("issue", "")),
            "subject": str(row.get("subject", "")),
            "company": row.get("company", None) if pd.notna(row.get("company")) else None
        })
    return records

def write_output(file_path, results):
    """
    Writes the final output.csv.
    """
    df_out = pd.DataFrame(results)
    df_out.to_csv(file_path, index=False)
