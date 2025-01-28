import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load Data Function
def load_data2(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Strip spaces from column names
    print("Columns in DataFrame after loading:", df.columns.tolist())  # Print for verification
    return df

# Preprocess Data Function
def preprocess_data(df, target_column):
    """Preprocess the data by dropping unnecessary columns and encoding categorical variables."""
    # Check available columns
    print("Available columns before dropping:", df.columns.tolist())
    
    # Drop unnecessary columns (make sure not to drop the target column)
    columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Check if the target column is still in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    
    # # Convert TransactionStartTime to datetime and extract useful features
    # if 'TransactionStartTime' in df.columns:
    #     df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    #     df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    #     df['TransactionDay'] = df['TransactionStartTime'].dt.day
    #     df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    #     df['TransactionYear'] = df['TransactionStartTime'].dt.year
    #     df = df.drop(columns=['TransactionStartTime'])  # Drop the original column
    
    # Check columns after creating new features
    print("Columns after feature extraction:", df.columns.tolist())
    
    # Identify categorical and numerical columns (excluding the target column)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    
    # Remove the target column from numerical_cols
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),  # Keep numerical columns as is
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical columns
        ])
    
    # Split the data into features and target
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]                  # Target
    
    print(f"Features (X) columns after dropping target: {X.columns.tolist()}")
    print(f"Target (y) has {y.shape[0]} records.")
    
    return preprocessor, X, y

# Split Data Function
def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train Logistic Regression Model
def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Hyperparameter Tuning for Random Forest
def tune_random_forest(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest using Grid Search."""
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_rf, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Evaluate Models
def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }