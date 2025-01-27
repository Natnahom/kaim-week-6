from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encode_categorical_variables(df, categorical_cols):
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df