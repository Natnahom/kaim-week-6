from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_standardize(df, numerical_cols, method='normalize'):
    if method == 'normalize':
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    elif method == 'standardize':
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df