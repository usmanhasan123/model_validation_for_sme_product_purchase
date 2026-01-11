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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.graph_objects as go

def connect_to_gsheet(creds_json,spreadsheet_name):
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
df_bin['prob']=model.predict_proba(df_bin)[:, 1]

# result = df_bin.loc[df_bin.groupby(["sme_id", "day"])["prob"].idxmax()]
df_bin["is_recommended"] = (
    df_bin["prob"]
    == df_bin.groupby(["sme_id", "day"])["prob"].transform("max")
).astype(int)
# st.write(result)
# st.write(df_bin)
df_purch=df_bin[df_bin['is_purchase']==1]
accuracy_list=[]
precision_list=[]
recall_list=[]
f1_list=[]
auc_list=[]
day=[]
for i in df_purch['day'].unique():
    df_day=df_purch[df_purch['days']==i]
    accuracy=accuracy_score(df_day['is_purchase'], df_day['is_recommended'])
    precision=precision_score(df_day['is_purchase'], df_day['is_recommended'])
    recall=recall_score(df_day['is_purchase'], df_day['is_recommended'])
    f1=f1_score(df_day['is_purchase'], df_day['is_recommended'])
    auc=roc_auc_score(df_day['is_purchase'], df_day['is_recommended'])

    day.append(i)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    auc_list.append(auc)

dff=pd.DataFrame()
dff['day']=day
dff['accuracy']=accuracy_list
dff['precision']=precision_list
dff['recall']=recall_list
dff['f1']=f1_list
dff['auc']=auc_list

fig = go.Figure([
go.Scatter(x=dff['day'], y=dff['accuracy'], mode='lines+markers', name='Accuracy', yaxis='y1'),
go.Scatter(x=dff['day'], y=dff['precision'], mode='lines+markers', name='Precision', yaxis='y1'),
go.Scatter(x=dff['day'], y=dff['recall'], mode='lines+markers', name='Recall', yaxis='y1'),
go.Scatter(x=dff['day'], y=dff['f1'], mode='lines+markers', name='F1 score', yaxis='y1'),
go.Scatter(x=dff['day'], y=dff['auc'], mode='lines+markers', name='AUC score', yaxis='y1'),
])

fig.update_layout(
    title="Day on Day model performance",
    xaxis_title="Day",
    yaxis_title="Score",
    template="plotly_white",
    legend=dict(x=1.1, y=1.1),
    height=500
)

st.plotly_chart(fig, use_container_width=True)




