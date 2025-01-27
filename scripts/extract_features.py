import pandas as pd

def extract_features(df):
    df['transaction_hour'] = pd.to_datetime(df['TransactionStartTime']).dt.hour
    df['transaction_day'] = pd.to_datetime(df['TransactionStartTime']).dt.day
    df['transaction_month'] = pd.to_datetime(df['TransactionStartTime']).dt.month
    df['transaction_year'] = pd.to_datetime(df['TransactionStartTime']).dt.year
    
    return df