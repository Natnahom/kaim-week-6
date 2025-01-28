def handle_missing_values(df):
    # Fill numerical NaNs with mean
    numeric_cols = df.select_dtypes(include=['number']).columns  # Select only numeric columns
    df[numeric_cols].fillna(df[numeric_cols].mean(), inplace=True)
    
    # Removal (optional)
    df.dropna(inplace=True)  # Remove rows with NaNs if few
    
    return df