import pytest
import pandas as pd
from scripts.load_and_overview import *
from scripts.EDA import *

def test_load_data():
    """Test if data is loaded correctly."""
    df = load_data('path_to_test_data.csv')  # Use a sample data file
    assert isinstance(df, pd.DataFrame)
    assert not df.empty  # Ensure the DataFrame is not empty

def test_plot_numerical_distribution():
    """Test numerical distribution plotting."""
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    # This is a demonstration; you may need to check the output visually or by mocking plt.show()
    plot_numerical_distribution(df)

def test_plot_categorical_distribution():
    """Test categorical distribution plotting."""
    df = pd.DataFrame({'B': ['cat', 'dog', 'cat', 'bird']})
    plot_categorical_distribution(df)

# More tests can be added here for other functions