import pandas as pd

# Path to your dataset inside out_2/
INPUT_CSV = "cleaned_dataset.csv"

# Your official ticker map (same as in run_merton_pipeline.py)
TICKER_MAP = {
    "Cisco": "CSCO",
    "Micron Technology": "MU",
    "Apple": "AAPL",
    "QualComm": "QCOM",
    "IBM": "IBM",
    "Adobe": "ADBE",
    "AMD": "AMD",
    "Roku": "ROKU",
    "Twilio": "TWLO",
    "Uber": "UBER",
    "Zillow": "Z",
    "Snapchat": "SNAP",
    "Microsoft": "MSFT",
    "Block": "SQ",
    "Texas Instruments": "TXN",
    "Amazon": "AMZN",
    "Meta": "META",
    "Pinterest": "PINS",
    "PayPal": "PYPL",
    "Nvidia": "NVDA",
    "DocuSign": "DOCU",
    "MongoDB": "MDB",
    "Salesforce": "CRM",
    "Oracle": "ORCL",
    "Peloton": "PTON",
    "KLA Corp": "KLAC",
    "Cloudflare": "NET",
    "Datadog": "DDOG",
    "Zoom": "ZM",
    "Etsy": "ETSY"
}

# Normalize issuer names for comparison
TICKER_MAP_LOWER = {k.strip().lower(): v for k, v in TICKER_MAP.items()}

def main():
    df = pd.read_csv(INPUT_CSV, usecols=["issuer"])
    
    # Normalize issuer names
    issuers_lower = df["issuer"].astype(str).str.strip().str.lower()
    
    # Identify unmatched issuers
    unmatched = sorted(set(issuers_lower) - set(TICKER_MAP_LOWER.keys()))
    
    print("\n=== Unmatched Issuers (not in TICKER_MAP) ===")
    if unmatched:
        for u in unmatched:
            print(" -", u)
    else:
        print("All issuers in cleaned_dataset.csv are mapped!")
    
    print("\nTotal unique issuers in dataset:", df["issuer"].nunique())
    print("Total unmatched:", len(unmatched))

if __name__ == "__main__":
    main()
