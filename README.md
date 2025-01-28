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

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Author
    - Name: Natnahom Asfaw
    - Date: 22/01/2025