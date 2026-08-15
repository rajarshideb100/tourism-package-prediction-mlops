import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'tourism.csv'
EXPECTED_COLUMNS = [
    'Unnamed: 0','CustomerID','ProdTaken','Age','TypeofContact','CityTier',
    'DurationOfPitch','Occupation','Gender','NumberOfPersonVisiting',
    'NumberOfFollowups','ProductPitched','PreferredPropertyStar','MaritalStatus',
    'NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar',
    'NumberOfChildrenVisiting','Designation','MonthlyIncome'
]

def validate_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing_columns:
        raise ValueError(f'Missing expected columns: {missing_columns}')
    if df.empty:
        raise ValueError('Dataset is empty.')
    if df['ProdTaken'].isna().any():
        raise ValueError('Target column ProdTaken contains missing values.')
    if not set(df['ProdTaken'].dropna().unique()).issubset({0,1}):
        raise ValueError('ProdTaken must contain only 0 and 1.')
    print(f'Dataset validated successfully: {df.shape[0]:,} rows x {df.shape[1]} columns')
    print(f'Missing values: {int(df.isna().sum().sum())}')
    print(f'Duplicate rows: {int(df.duplicated().sum())}')
    print(f'Extra columns: {extra_columns if extra_columns else "None"}')
    print('\nTarget distribution:')
    print(df['ProdTaken'].value_counts().rename_axis('ProdTaken').to_frame('Count'))
    print('\nData types:')
    print(df.dtypes.to_string())
    return df

if __name__ == '__main__':
    validate_dataset()
