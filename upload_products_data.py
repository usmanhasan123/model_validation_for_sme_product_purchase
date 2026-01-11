import pandas as pd
import numpy as np
import os
import tempfile
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def connect_to_gsheet(creds_json,spreadsheet_name):
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    
    credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(credentials)
    spreadsheet = client.open(spreadsheet_name)  # Access the first sheet
    return spreadsheet

def generate_products():
    products = pd.DataFrame({
        "product_id": ["overdraft", "invoice_financing", "term_loan"],
        "max_limit": [100_000, 500_000, 1_000_000],
        "base_rate": [0.14, 0.02, 0.11],
        "tenure_days": [0, 90, 365],
        "risk_weight": [0.4, 0.3, 0.6]
    })
    return products

# day=datetime.now().hour
private_key_json=os.getenv('private_key_json')

with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    tmp.write(private_key_json.encode())  # write bytes
    tmp.flush()
    creds_path = tmp.name

SPREADSHEET_NAME = 'Synthetic Data'
# SHEET_NAME = 'Sheet1'
CREDENTIALS_FILE = creds_path # './crendentials.json'

sheet_by_name= connect_to_gsheet(CREDENTIALS_FILE, SPREADSHEET_NAME)
products_df=generate_products()

data_to_upload = products_df.values.tolist()
# data_to_upload
# sheet_by_name.clear()
ws = sheet_by_name.worksheet("products_data")
# ws.resize(rows=1, cols=1)  # Shrink sheet completely
# ws.clear()
ws.append_rows(data_to_upload)
print('uploaded')

            
