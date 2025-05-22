import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Return full file paths for all datasets
def get_paths(data_path):
    cust_path = f"{data_path}/customer_data.csv"
    fin_path = f"{data_path}/customer_financials.csv"
    loan_path = f"{data_path}/loan_data.csv"
    delinq_path = f"{data_path}/loan_deliquencies.csv"
    return cust_path, fin_path, loan_path, delinq_path

# Load all CSV files and return as DataFrames
def read_data(data_path, do_inspect_data):
    # Get file paths for each dataset
    cust_path, fin_path, loan_path, delinq_path = get_paths(data_path)

    # Read CSV files into DataFrames
    cust = pd.read_csv(cust_path)
    fin = pd.read_csv(fin_path, sep=';')
    loan = pd.read_csv(loan_path)
    delinq = pd.read_csv(delinq_path)

    # Print duplicate and NaN info (optional)
    if do_inspect_data:
        inspect_data(cust, fin, loan, delinq)

    return cust, fin, loan, delinq

# Inspect lengths, duplicates and NaNs
def inspect_data(cust, fin, loan, delinq):
    # Print lengths
    print("\n")
    print("customer_data length:", len(cust))
    print("customer_financials length:", len(fin))
    print("loan_data length:", len(loan))
    print("loan_delinquencies length:", len(delinq))
    print("\n")

    # Print duplicates
    print("customer_data duplicates:", cust.duplicated().sum())
    print("customer_financials duplicates:", fin.duplicated().sum())
    print("loan_data duplicates:", loan.duplicated().sum())
    print("loan_delinquencies duplicates:", delinq.duplicated().sum())
    print("\n")

    # Helper to print NaNs with their percentages
    def print_nans(df, name):
        print(f"{name} NaNs:")
        nans = df.isnull().sum()
        total = len(df)
        for col, count in nans.items():
            pct = (count / total) * 100
            print(f"{col}: {count} ({pct:.1f}%)")
        print("\n")

    # Print NaNs
    print_nans(cust, "customer_data")
    print_nans(fin, "customer_financials")
    print_nans(loan, "loan_data")
    print_nans(delinq, "loan_delinquencies")

# Create binary default labels from delinquency data
def create_target(delinq):
    # Convert delinquency dates to datetime
    delinq['start_date'] = pd.to_datetime(delinq['start_date'])
    delinq['end_date'] = pd.to_datetime(delinq['end_date'])

    # Calculate delay duration in months
    delinq['delay_months'] = (delinq['end_date'] - delinq['start_date']) / np.timedelta64(1, 'M')
    delinq['delay_months'] = delinq['delay_months'].round().astype(int)

    # Label as default if delay >= 3 months
    delinq['default'] = (delinq['delay_months'] >= 3).astype(int)

    # Get one label per loan (max value = 1 if any default)
    target = delinq.groupby('loan_id')['default'].max().reset_index().sort_values(by='loan_id')

    return target

# Aggregate financial features per customer
def aggregate_financials(fin, fin_cols):
    # Convert financial dates to datetime
    fin['date'] = pd.to_datetime(fin['date'])

    # Calculate median and std for selected columns
    agg_dict = {col: ['median', 'std'] for col in fin_cols}
    agg_fin = fin.groupby('cust_id').agg(agg_dict)

    # Flatten multi-level columns
    agg_fin.columns = ['_'.join(col) for col in agg_fin.columns]
    agg_fin = agg_fin.reset_index().sort_values(by='cust_id')

    return agg_fin

# Create final dataset by merging, transforming, and encoding features
def feature_engineering(loan, cust, fin, target,
                        no_outlier_removal=False,
                        outlier_percentile=0.005):
    # Sort and merge loan and customer data
    loan = loan.sort_values(by='cust_id')
    cust = cust.sort_values(by='cust_id')
    df = loan.merge(cust, on='cust_id', how='left')
    
    # Add financial features
    fin_cols = ['salary', 'current_acc_balance', 'saving_acc_balance']
    agg_fin = aggregate_financials(fin, fin_cols)
    df = df.merge(agg_fin, on='cust_id', how='left')

    # Calculate age
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['age'] = 2020 - df['birth_date'].dt.year  # may raise ethical concerns

    # Calculate tenure
    df['joined_bank_date'] = pd.to_datetime(df['joined_bank_date'])
    df['customer_tenure_at_loan'] = (
        pd.to_datetime(df['date']) - pd.to_datetime(df['joined_bank_date'])
    ).dt.days

    # Create salary_missing flag
    df['salary_missing'] = df['salary_median'].isnull().astype(int)

    # Fill NaNs in financial features with column-wise medians
    for col in fin_cols:
        matching_cols = [c for c in df.columns if col in c]
        for mcol in matching_cols:
            fill_value = df[mcol].median()
            df[mcol].fillna(fill_value, inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(
        df,
        columns=['loan_reason', 'employment'],
        drop_first=True
    )

    # Merge default target labels
    df = df.merge(target, on='loan_id', how='left')

    # Fill NaN's with 0 (non-default) since they are not in the delinq data
    df['default'] = df['default'].fillna(0).astype(int)

    # Remove outliers (optional)
    if not no_outlier_removal:
        df = remove_outliers_percentile(df, outlier_percentile)

    return df

# Save preprocessed dataset to CSV
def save_dataset(df, data_path, save_name="final_dataset.csv"):
    df.to_csv(f"{data_path}/{save_name}", index=False)

# Load preprocessed dataset from CSV
def load_dataset(data_path, file_name="final_dataset.csv"):
    df = pd.read_csv(f"{data_path}/{file_name}")
    return df

# Remove outliers using percentile thresholds
def remove_outliers_percentile(df, percentile):
    # Identify numeric columns and initialize mask
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mask = pd.Series(True, index=df.index)

    # Build mask for all numeric columns
    for col in numeric_cols:
        lower = df[col].quantile(percentile)
        upper = df[col].quantile(1 - percentile)
        mask &= df[col].between(lower, upper)

    return df[mask]

# Split dataset into X and y, drop irrelevant or risky columns
def prepare_Xy(df):
    drop_cols = [
        'default',  # output (y) values
        'loan_id', 'cust_id', 'date',  # not related to the task
        'joined_bank_date',  # tenure is used alternatively
        'birth_date',  # age is used alternatively
        'number_client_calls_from_bank',  # causes data leakage
        'number_client_calls_to_bank',  # may cause data leakage
        'religion',  # raises ethical concerns
        'gender',  # raises ethical concerns
        'postal_code',  # may raise ethical concerns
    ]

    X = df.drop(drop_cols, axis=1)
    y = df['default']

    return X, y

# Split data into train and test sets with stratification
def split_data(X, y, test_size, random_state):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True
    )
