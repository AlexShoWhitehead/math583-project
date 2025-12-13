import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# =========================
# CONFIG
# =========================
ROOT_DIR = "."
EXCEL_NAME = "list.xlsx"

ALLOWED_FORMS_BASE = {"10-K", "10-Q", "8-K"}
ALLOW_AMENDMENTS = True

# --- CONFIGURATION FOR STRUCTURED DATA ---
# These are the extensions for the full machine-readable XBRL submission files
XBRL_EXTENSIONS = {
    ".xml", ".xsd", ".ins", ".cal", ".lab", ".pre", ".def", ".htm", ".txt"
}
# ---------------------------------------------

CONVERT_TO_PDF = False

REQUEST_DELAY = 0.25
HEADERS = {
    "User-Agent": "RiskClassProject/1.0 (alexander.s.whitehead@gmail.com) - Data Acquisition",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

# =========================
# HELPERS
# =========================

def safe_filename(name: str) -> str:
    """Sanitizes a string for use as a filename."""
    return re.sub(r"[^\w\-. ]+", "_", name).strip()

def ensure_dir(path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def fetch(url: str) -> requests.Response:
    """Fetches a URL with a delay and error checking."""
    time.sleep(REQUEST_DELAY)
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r

def normalize_to_index(url: str) -> str:
    """Ensures the URL ends with -index.htm for consistency."""
    if url.endswith("-index.htm") or url.endswith("-index.html"):
        return url
    if url.endswith(".htm") or url.endswith(".html"):
        return re.sub(r"\.htm(l)?$", "-index.htm", url)
    return url

def allowed_form(form_type: str) -> bool:
    """Checks if the form type is in the allowed list."""
    ft = form_type.strip().upper()
    if ft in ALLOWED_FORMS_BASE:
        return True
    if ALLOW_AMENDMENTS:
        base = ft.split("/")[0].strip()
        return base in ALLOWED_FORMS_BASE
    return False

def read_company_excel(excel_path: str):
    """Reads the company excel file."""
    df = pd.read_excel(excel_path, header=2)
    cols = {c.lower().strip(): c for c in df.columns}
    form_col = cols.get("form type")
    url_col = cols.get("filings url")
    if form_col is None or url_col is None:
        raise ValueError(f"Missing columns in {excel_path}. Found: {list(df.columns)}")
    df = df[[form_col, url_col]].dropna()
    df[form_col] = df[form_col].astype(str)
    df[url_col] = df[url_col].astype(str)
    df = df[df[form_col].apply(allowed_form)]
    return df, form_col, url_col

def download_file(file_url: str, out_dir: str):
    """Downloads a single file and saves it, checking for existence first."""
    ensure_dir(out_dir)
    filename = safe_filename(os.path.basename(urlparse(file_url).path))
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        return out_path

    try:
        r = fetch(file_url)
        with open(out_path, "wb") as f:
            f.write(r.content)
        
        # Simple check for file type for printing
        file_ext = os.path.splitext(filename)[1].lower()
        file_type = "XBRL" if file_ext in XBRL_EXTENSIONS and file_ext != ".htm" else "Document"
        print(f"      Downloaded {file_type}: {filename}")
        return out_path
    except requests.exceptions.RequestException as e:
        print(f"      ❌ ERROR downloading {filename}: {e}")
        return None


def get_structured_doc_links(index_url: str, target_form: str):
    """
    Scans the index page for the primary HTML document (which is iXBRL
    for modern filings) and all associated XBRL/XML supporting files.

    Returns: A list of unique document links to download.
    """
    r = fetch(index_url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Find the table containing the document links
    table = soup.find("table", class_="tableFile", summary=re.compile("Document Format Files", re.I))
    if not table:
        return []

    target_form = target_form.upper()
    links_to_download = set()

    rows = table.find_all("tr")[1:]  # skip header

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        doc_href_tag = cols[2].find("a")
        doc_type = cols[3].get_text(strip=True).upper()

        if doc_href_tag:
            href = doc_href_tag.get("href", "")
            if not href: continue

            full_link = urljoin(index_url, href)
            path = urlparse(full_link).path.lower()
            file_ext = os.path.splitext(path)[1]
            description = cols[1].get_text(strip=True).lower()

            # 1. Primary Document: The one whose Type matches the form (e.g., '10-Q').
            # This is the modern iXBRL-enabled HTML file.
            if doc_type == target_form:
                links_to_download.add(full_link)

            # 2. Structured XBRL/Supporting Files: Check for common XBRL types/extensions.
            if doc_type.startswith("EX-101") or file_ext in XBRL_EXTENSIONS:
                 # Exclude the full text file if it's the large submission text (legacy file)
                 if "complete submission text file" not in description:
                     links_to_download.add(full_link)
            
            # 3. Legacy Full Submission Text File: The all-in-one file.
            # We download this as a valuable fallback or archive copy.
            if "complete submission text file" in description and doc_type in ("TXT", target_form):
                links_to_download.add(full_link)

    return list(links_to_download)

def process_filing(index_url: str, form_type: str, out_dir: str, relational_data: list):
    """
    Downloads all relevant documents for a filing and records their metadata.
    """
    index_url = normalize_to_index(index_url)

    # Get links for structured data (iXBRL/XBRL files) and the old full text file.
    links_to_download = get_structured_doc_links(index_url, form_type)

    if not links_to_download:
        print("      ⚠️ Could not find any structured documents or primary HTML/text file.")
        return

    print(f"      Found {len(links_to_download)} documents to download (iXBRL/XBRL/Legacy).")
    
    # Extract company names for relational table
    company_name = os.path.basename(out_dir)
    company_bucket = os.path.basename(os.path.dirname(out_dir))

    for link in links_to_download:
        local_path = download_file(link, out_dir) # returns local path
        if local_path:
            # Collect data for the relational table
            relational_data.append({
                "Form_Type": form_type,
                "Index_URL": index_url,
                "Downloaded_Filename": os.path.basename(local_path),
                "Local_Path": local_path,
                "Company_Bucket": company_bucket,
                "Company_Name": company_name
            })


# =========================
# MAIN WALKER
# =========================

def main():
    buckets = ["healthy", "risky", "failed"]
    relational_data = [] # NEW: List to store our relational map

    for bucket in buckets:
        bucket_path = os.path.join(ROOT_DIR, bucket)
        if not os.path.isdir(bucket_path):
            continue

        for company in os.listdir(bucket_path):
            company_dir = os.path.join(bucket_path, company)
            if not os.path.isdir(company_dir):
                continue

            excel_path = os.path.join(company_dir, EXCEL_NAME)
            if not os.path.exists(excel_path):
                continue

            print(f"\n=== {bucket} / {company} ===")

            try:
                df, form_col, url_col = read_company_excel(excel_path)
            except Exception as e:
                print(f"Excel read error: {e}")
                continue

            for _, row in df.iterrows():
                form_type = row[form_col].strip().upper()
                filing_url = row[url_col].strip()

                if not filing_url or filing_url.lower() == "nan":
                    continue

                print(f"  Processing {form_type} filing: {filing_url}")
                # NEW: Pass the relational_data list
                process_filing(filing_url, form_type, company_dir, relational_data) 

    # NEW: Output the final relational table
    if relational_data:
        df_relational = pd.DataFrame(relational_data)
        
        # Display a clean subset of the data
        print("\n====================================================================================================")
        print("RELATIONAL FILING DOCUMENT MAP (Summary)")
        print("====================================================================================================")
        
        # Format the DataFrame for clean display
        df_display = df_relational[['Company_Bucket', 'Company_Name', 'Form_Type', 'Downloaded_Filename', 'Index_URL']]
        df_display = df_display.rename(columns={
            'Company_Bucket': 'Bucket', 
            'Company_Name': 'Company', 
            'Form_Type': 'Form', 
            'Downloaded_Filename': 'File_Name', 
            'Index_URL': 'Filing_Index_URL'
        })
        
        # Print using Markdown format for readability
        print(df_display.to_markdown(index=False))

        # Save the full relational data (including Local_Path) to CSV
        output_csv_path = os.path.join(ROOT_DIR, "filing_document_map.csv")
        df_relational.to_csv(output_csv_path, index=False)
        print(f"\nFull relational map saved to: {output_csv_path}")

    else:
         print("\nNo filings were processed to generate a relational map.")


if __name__ == "__main__":
    main()