import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_distribution(df, sample_size=20000):
    """Plot distributions of numerical features with sampling."""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    df_sample = df.sample(n=min(sample_size, len(df)))  # Sample the data
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(df_sample[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()
        plt.clf()

def plot_categorical_distribution(df, max_features=5, sample_size=50):
    """Plot distributions of a limited number of categorical features with sampling."""
    categorical_features = df.select_dtypes(include=[object]).columns.tolist()[:max_features]  # Limit to max_features
    df_sample = df.sample(n=min(sample_size, len(df)))  # Sample the data

    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df_sample, x=feature)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
        plt.clf()

def correlation_analysis(df):
    """Generate and display the correlation matrix."""
    # Select only numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def missing_values_analysis(df):
    """Identify and visualize missing values in the dataset."""
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_outliers(df):
    """Detect outliers using box plots for numerical features."""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.xlabel(feature)
        plt.show()

