import pandas as pd
from src.schemas import TransactionData
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define the preprocessor (should match the one used in training)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['CountryCode', 'Amount', 'Value', 'PricingStrategy']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId'])
    ])

def preprocess_input(data: TransactionData) -> pd.DataFrame:
    """Preprocess the input data for prediction."""
    input_data = pd.DataFrame([data.dict()])
    return preprocessor.transform(input_data)