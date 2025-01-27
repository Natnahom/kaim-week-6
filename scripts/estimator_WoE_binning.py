import pandas as pd
import matplotlib.pyplot as plt
from woe.feature_process import proc_woe_continuous
from woe.feature_process import calulate_iv
from woe.eval import eval_data_summary

def construct_default_estimator(df):
    # Check if RFMS_score exists, and create it if necessary
    if 'RFMS_score' not in df.columns:
        df['RFMS_score'] = (df['total_transaction_amount'] * df['transaction_count']) / df['average_transaction_amount']
    
    # Define cutoff_value appropriately
    cutoff_value = 0.5  #adjust as needed
    
    # Initialize default_label to avoid KeyError
    df['default_label'] = 0  # (0 for all)

    # Visualize transactions in RFMS space
    plt.scatter(df['total_transaction_amount'], df['transaction_count'], c=df['default_label'], cmap='coolwarm')
    plt.xlabel('Total Transaction Amount')  # Adjusted label
    plt.ylabel('Transaction Count')  # Adjusted label
    plt.title('RFMS Visualization')
    plt.show()
    
    # Establish boundary and classify users
    df['default_label'] = (df['RFMS_score'] < cutoff_value).astype(int)
    return df

def calulate_iv_wrapper(df, var, target, global_bt=1, global_gt=1):
    '''
    Wrapper for calculate_iv to handle target column renaming.
    
    :param df: pandas DataFrame
    :param var: the feature variable for which to calculate IV
    :param target: the target variable (binary)
    :param global_bt: global binning threshold (if applicable)
    :param global_gt: global good threshold (if applicable)
    :return: Information Value (IV)
    '''
    # Check if the target column is 'FraudResult' and rename it to 'target'
    if target == 'FraudResult':
        df_renamed = df.rename(columns={target: 'target'})
    else:
        df_renamed = df.copy()  # Make a copy without renaming if the target is different

    # Call the original calculate_iv function
    return calulate_iv(df_renamed, var, global_bt, global_gt), df_renamed

def woe_transformation(df, feature, target, global_bt=1, global_gt=1, min_sample=1):
    """
    Perform WOE transformation on a specified feature and calculate its Information Value (IV).
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - feature: the name of the feature to transform.
    - target: the name of the target variable (binary).
    - global_bt: global binning threshold for WOE calculation (if applicable).
    - global_gt: global good threshold for WOE calculation (if applicable).
    - min_sample: minimum sample size for the transformation.
    
    Returns:
    - df: DataFrame with the WOE transformed feature.
    - iv_value: Information Value of the feature.
    - eval_summary: Evaluation summary of the transformed feature.
    """
    if df[feature].dtype in ['int64', 'float64']:
        # Use .loc to avoid SettingWithCopyWarning
        df_reduced = df.loc[:, [feature, target]].copy()  # Make a copy to avoid the warning
        df_reduced.loc[:, 'woe_' + feature] = proc_woe_continuous(
            df_reduced, feature, global_bt, global_gt, min_sample
        )
        
        # Calculate Information Value (IV)
        iv_value = calulate_iv(df_reduced, 'woe_' + feature, global_bt, global_gt)
        eval_summary = "Evaluation summary here..."  # Replace with actual summary logic if needed

        return df_reduced, iv_value, eval_summary
    else:
        raise ValueError("Feature must be numeric (continuous) for this function.")
