import pandas as pd
import numpy as np
import os
import tempfile
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import streamlit as st
import joblib
import json

def connect_to_gsheet(self, creds_json,spreadsheet_name):
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    
    credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(credentials)
    spreadsheet = client.open(spreadsheet_name)  # Access the first sheet
    return spreadsheet


private_key_json=st.secrets['private_key_json']
json_str = json.dumps(dict(private_key_json))

with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    tmp.write(json_str.encode())  # write bytes
    tmp.flush()
    creds_path = tmp.name

SPREADSHEET_NAME = 'Synthetic Data'
# SHEET_NAME = 'Sheet1'
CREDENTIALS_FILE = creds_path # './crendentials.json'

sheet_by_name= connect_to_gsheet(CREDENTIALS_FILE, SPREADSHEET_NAME)

ws = sheet_by_name.worksheet("sme_raw_data")
x=ws.get_all_records()
df=pd.DataFrame(x)

ws = sheet_by_name.worksheet("products_data")
x=ws.get_all_records()
products_df=pd.DataFrame(x)

products=products_df['product_id'].to_list()
df['products']=[products]*len(df)
df_2=df.explode('products')
df_3=df_2.merge(products_df, how='left', left_on='products', right_on='product_id')
df_3=df_3.drop(columns='products')
df_3['is_purchase']=df_3.apply(lambda x: 1 if x['product_purchased']==x['product_id'] else 0, axis=1)
df_bin=df_3.drop(columns='product_purchased')

model=joblib.load('model.pkl')
df_bin['prob']=model.predict_proba(df_bin)

st.write(df_bin)






