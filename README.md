## This is week-6 of 10 academy.

# Task1: Understanding credit score

## Key Concepts of Credit Risk

- Definition: Credit risk is the potential that a borrower will fail to meet obligations in accordance with agreed terms. It's crucial for lenders to assess creditworthiness, especially for micro-, small, and medium-sized enterprises (MSMEs) that may lack traditional credit histories.
- Importance of Credit Scoring: Effective credit scoring models help predict the likelihood of default, influencing lending decisions and risk management.
- Alternative Credit Scoring: This modern approach uses alternative data sources (like transaction records) to evaluate creditworthiness, improving access to finance for MSMEs.
References to Explore
- Statistical Techniques: Review statistical methods for credit scoring, including logistic regression and machine learning algorithms.
HKMA Guidelines: Understand the Hong Kong Monetary Authority's perspectives on alternative credit scoring.
- World Bank Guidelines: Familiarize yourself with global standards and practices in credit scoring.

# Exploratory Data Analysis (EDA) Project

## Overview
This project involves performing Exploratory Data Analysis (EDA) on a dataset to extract insights and visualize the data using Python. The EDA process is modularized into functions for better maintainability and reusability. The project also includes unit tests to ensure the correctness of the functions.

## Installation
To run this project, you need to have Python and pip installed on your machine. Follow these steps to set up your environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment (optional but recommended):
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
    pip install -r requirements.txt
## Usage
To perform EDA on your dataset, update the data_url in the main function of your_script.py with the path to your CSV file. Then run the script:
- Run it as you see in the analysis.ipynb

# Task 3: WOE Transformation and Information Value Calculation

## Overview

This task involves performing Weight of Evidence (WOE) transformation on a specified feature of a dataset and calculating its Information Value (IV). The WOE transformation is commonly used in binary classification to improve the predictive power of continuous variables.

## Objectives

- Transform the continuous variable `total_transaction_amount` using WOE.
- Calculate the Information Value (IV) of the transformed variable.
- Evaluate the transformation's effectiveness.

## Requirements

- Python 3.x
- Pandas
- NumPy

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
    ```

## Install the required packages:
pip install pandas numpy
Ensure that you have the dataset nor_standa_data available for processing.
Functions
calulate_iv_wrapper(df, var, target, global_bt=1, global_gt=1)
Description: Wrapper function to calculate IV and handle target column renaming.
Parameters:
df: DataFrame containing the data.
var: Feature variable for which to calculate IV.
target: Target variable (binary).
global_bt: Global binning threshold (optional).
global_gt: Global good threshold (optional).
Returns: Tuple containing Information Value (IV) and the renamed DataFrame.
woe_transformation(df, feature, target, global_bt=1, global_gt=1, min_sample=1)
Description: Performs WOE transformation on a specified feature.
Parameters:
df: DataFrame containing the data.
feature: The name of the feature to transform.
target: The name of the target variable (binary).
global_bt: Global binning threshold (optional).
global_gt: Global good threshold (optional).
min_sample: Minimum sample size for the transformation.
Returns: Tuple containing the transformed DataFrame, Information Value (IV), and evaluation summary.

## Usage
Import the necessary functions in your script:
from your_module import calulate_iv_wrapper, woe_transformation
Load your dataset:
- import pandas as pd

- nor_standa_data = pd.read_csv('path/to/your/data.csv')
- Calculate IV and perform WOE transformation:
- iv_value, df_renamed = calulate_iv_wrapper(nor_standa_data, 'total_transaction_amount', 'FraudResult')
- df_transformed, evaluation, eval_summary = woe_transformation(df_renamed, 'total_transaction_amount', 'target')

# Task 4: Fraud Detection Model Development

## Overview

This project focuses on developing a machine learning model to detect fraudulent transactions using a dataset that includes various features related to transactions. The primary goal is to preprocess the data, train a model using a Random Forest classifier, and tune the hyperparameters for optimal performance.

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Development](#model-development)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, clone the repository and install the required packages using pip:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Usage
- You can see it in the analysis3.ipynb
- This will preprocess the data, train the model, and output the results.

## Data Description
The dataset contains the following columns:

- TransactionId: Unique identifier for each transaction.
- BatchId: Identifier for the batch of transactions.
- AccountId: Unique identifier for the account.
- SubscriptionId: Identifier for the subscription.
- CustomerId: Unique identifier for the customer.
- CurrencyCode: Currency of the transaction.
- CountryCode: Country associated with the transaction.
- ProviderId: Identifier for the service provider.
- ProductId: Identifier for the product.
- ProductCategory: Category of the product.
- ChannelId: Channel through which the transaction was made.
- Amount: Transaction amount.
- Value: Transaction value.
- TransactionStartTime: Timestamp of the transaction.
- PricingStrategy: Pricing strategy applied.
- FraudResult: Target variable indicating whether the transaction is fraudulent (1) or not (0).
Model Development
1. Data Preprocessing:
- Unnecessary columns are dropped.
- The TransactionStartTime is converted to datetime format, and useful features (hour, day, month, year) are extracted.
- Categorical variables are one-hot encoded, and numerical variables are kept as is.
2. Model Training:
- A Random Forest classifier is trained on the processed feature set (X_train_processed) with the target variable (y_train).
## Hyperparameter Tuning
- Hyperparameter tuning is performed using RandomizedSearchCV to optimize the following parameters:

- n_estimators: Number of trees in the forest.
- max_depth: Maximum depth of the tree.
- min_samples_split: Minimum number of samples required to split an internal node.
- The best model is selected based on accuracy.

## Results
The model's performance is evaluated based on accuracy and other relevant metrics. Results are printed to the console and can be further analyzed.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Author
    - Name: Natnahom Asfaw
    - Date: 22/01/2025