def create_aggregate_features(df):
    # Total Transaction Amount
    df['total_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    
    # Average Transaction Amount
    df['average_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    
    # Transaction Count
    df['transaction_count'] = df.groupby('CustomerId')['Amount'].transform('count')
    
    # Standard Deviation of Transaction Amounts
    df['std_transaction_amount'] = df.groupby('CustomerId')['Amount'].transform('std')
    
    return df