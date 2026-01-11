import pandas as pd
import numpy as np
import os
import tempfile
import gspread
from oauth2client.service_account import ServiceAccountCredentials

class generate_sme:
    def __init__(self, day):
        self.day=day

    def connect_to_gsheet(self, creds_json,spreadsheet_name):
        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
        
        credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
        client = gspread.authorize(credentials)
        spreadsheet = client.open(spreadsheet_name)  # Access the first sheet
        return spreadsheet

    @staticmethod
    def generate_products():
        products = pd.DataFrame({
            "product_id": ["overdraft", "invoice_financing", "term_loan"],
            "max_limit": [100_000, 500_000, 1_000_000],
            "base_rate": [0.14, 0.02, 0.11],
            "tenure_days": [0, 90, 365],
            "risk_weight": [0.4, 0.3, 0.6]
        })
        return products

    def generate_events(self, sme_ids, avg_events=6):
        events = []
        # sme_df=generate_sme_snapshot(self)
        # sme_ids=list(self.sme_df['sme_id'].unique())
        for sme_id in sme_ids:
            n_events = np.random.poisson(avg_events)
    
            for _ in range(n_events):
                event_type = np.random.choice(
                    ["invoice_submitted", "invoice_financed", "od_draw", "repayment"],
                    p=[0.35, 0.25, 0.25, 0.15]
                )
    
                amount = np.random.uniform(5_000, 60_000)
    
                events.append({
                    "sme_id": sme_id,
                    "event_type": event_type,
                    "amount": round(amount, 2),
                    "day": self.day
                })
    
        return pd.DataFrame(events)


    def generate_sme_snapshot(self, n_smes=500, drift_strength=0.002):
        np.random.seed(self.day)
    
        base_cash_in = np.random.lognormal(mean=11, sigma=0.5, size=n_smes)
        base_cash_out = base_cash_in * np.random.uniform(0.85, 1.15, size=n_smes)
    
        # Concept drift: expenses rise slowly over time (inflation / slowdown)
        cash_out = base_cash_out * (1 + drift_strength * self.day)
    
        invoice_vol = base_cash_in * np.random.uniform(0.4, 1.2, size=n_smes)
    
        od_utilization = np.clip(
            np.random.beta(2, 5, size=n_smes) + drift_strength * self.day,
            0, 1
        )
    
        sme_df = pd.DataFrame({
            "sme_id": [f"SME_{i}" for i in range(n_smes)],
            "cash_in_30d": base_cash_in,
            "cash_out_30d": cash_out,
            "invoice_vol_30d": invoice_vol,
            "od_utilization": od_utilization,
            "relationship_age_days": np.random.randint(30, 900, size=n_smes),
            "day": self.day
        })
    
        sme_df["cash_flow_gap"] = sme_df["cash_in_30d"] - sme_df["cash_out_30d"]
    
        return sme_df

    @staticmethod
    def aggregate_event_features(events_df):
        # events_df=generate_events(self)
        agg = events_df.groupby("sme_id").agg(
            txn_count_30d=("event_type", "count"),
            avg_txn_value_30d=("amount", "mean"),
            invoice_submit_count_30d=("event_type", lambda x: (x == "invoice_submitted").sum()),
            od_draw_count_30d=("event_type", lambda x: (x == "od_draw").sum()),
            repayment_count_30d=("event_type", lambda x: (x == "repayment").sum()),
            invoice_financed_amt=("amount", lambda x: x[events_df.loc[x.index, "event_type"] == "invoice_financed"].sum()),
            repayment_amt=("amount", lambda x: x[events_df.loc[x.index, "event_type"] == "repayment"].sum()),
            od_draw_amt=("amount", lambda x: x[events_df.loc[x.index, "event_type"] == "od_draw"].sum())
        ).reset_index()
    
        return agg

    @staticmethod
    def calculate_exposure(sme_df, event_agg_df):
        df = sme_df.merge(event_agg_df, on="sme_id", how="left").fillna(0)
    
        # Outstanding exposures
        df["open_invoice_exposure"] = df["invoice_financed_amt"] - df["repayment_amt"]
        df["od_exposure"] = df["od_draw_amt"] - df["repayment_amt"]
    
        df["open_invoice_exposure"] = df["open_invoice_exposure"].clip(lower=0)
        df["od_exposure"] = df["od_exposure"].clip(lower=0)
    
        df["total_exposure"] = df["open_invoice_exposure"] + df["od_exposure"]
    
        # Utilization relative to cash inflow
        df["utilization_ratio"] = df["total_exposure"] / df["cash_in_30d"]
        df["utilization_ratio"] = df["utilization_ratio"].clip(0, 2)
    
        return df

    @staticmethod
    def assign_product_label(sme_df):
        labels = []
        # sme_with_exp=self.calculate_exposure(self, event_agg_df)
    
        for _, r in sme_df.iterrows():
    
            # 1️⃣ Probability SME takes ANY product
            activation_score = (
                0.4 * (r["utilization_ratio"]) +
                0.3 * (r["invoice_vol_30d"] / 500_000) +
                0.3 * (r["txn_count_30d"] / 50)
            )
    
            p_active = min(0.8, max(0.05, activation_score))
    
            if np.random.rand() > p_active:
                labels.append("none")
                continue
    
            # 2️⃣ Conditional probabilities given activation
            p_od = min(0.6, r["od_exposure"] / 100_000 + 0.2)
            p_invoice = min(0.6, r["invoice_vol_30d"] / 500_000)
            p_term = max(0.05, -r["cash_flow_gap"] / 300_000)
    
            total = p_od + p_invoice + p_term
            probs = [p_od / total, p_invoice / total, p_term / total]
    
            labels.append(
                np.random.choice(
                    ["overdraft", "invoice_financing", "term_loan"],
                    p=probs
                )
            )
    
        sme_df["product_purchased"] = labels
        return sme_df

    def final_data(self):
        sme_data=self.generate_sme_snapshot()
        events_data=self.generate_events(list(sme_data['sme_id'].unique()))
        agg_df=self.aggregate_event_features(events_data)
        df_sme=self.calculate_exposure(sme_data, agg_df)
        df=self.assign_product_label(df_sme)

        products_df=self.generate_products()

        #binary data
        products=products_df['product_id'].to_list()
        df['products']=[products]*len(df)
        df_2=df.explode('products')
        df_3=df_2.merge(products_df, how='left', left_on='products', right_on='product_id')
        df_3=df_3.drop(columns='products')
        df_3['is_purchase']=df_3.apply(lambda x: 1 if x['product_purchased']==x['product_id'] else 0, axis=1)
        df_bin=df_3.drop(columns='product_purchased')

        # multiclass data
        df_2=df.merge(products_df, how='left', left_on='product_purchased', right_on='product_id')
        df_2=df_2.drop(columns='product_id')
        df_multi=df_2.rename(columns={'product_purchased': 'product_id'})

        return df, products_df, df_bin, df_multi

generate=generate_sme()
private_key_json=os.getenv('private_key_json')

with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    tmp.write(private_key_json.encode())  # write bytes
    tmp.flush()
    creds_path = tmp.name

SPREADSHEET_NAME = 'Synthetic Data'
# SHEET_NAME = 'Sheet1'
CREDENTIALS_FILE = creds_path # './crendentials.json'

sheet_by_name= generate.connect_to_gsheet(CREDENTIALS_FILE, SPREADSHEET_NAME)
data, products_df, df_bin, df_multi=generate.final_data()

data_to_upload = [data.columns.values.tolist()] + data.values.tolist()
# data_to_upload
# sheet_by_name.clear()
ws = sheet_by_name.worksheet("sme_raw_data")
ws.resize(rows=1, cols=1)  # Shrink sheet completely
ws.clear()
ws.update("A1", data_to_upload)
print('uploaded')

            
