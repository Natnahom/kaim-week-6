import pandas as pd

def load_data(data_url):
    """Load dataset from a given URL or file path."""
    return pd.read_csv(data_url)

def overview_data(df):
    """Display basic information and first few rows of the dataset."""
    print("Data Overview:")
    print(df.info())
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

def summary_statistics(df):
    """Print summary statistics of the dataset."""
    print("\nSummary Statistics:")
    print(df.describe(include='all'))
