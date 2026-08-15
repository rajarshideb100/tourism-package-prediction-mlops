import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'tourism.csv'

# These columns are identifiers/indexes and do not contain generalizable predictive information.
DROP_COLUMNS = ['Unnamed: 0', 'CustomerID']
TARGET = 'ProdTaken'


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    before_shape = df.shape
    # Validate duplicates using the original customer-level records before removing identifiers.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=DROP_COLUMNS, errors='ignore')

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Workflow artifacts are written to the repository root so GitHub Actions can upload them directly.
    out_dir = Path(__file__).resolve().parents[2]
    Xtrain.to_csv(out_dir / 'Xtrain.csv', index=False)
    Xtest.to_csv(out_dir / 'Xtest.csv', index=False)
    ytrain.to_csv(out_dir / 'ytrain.csv', index=False, header=True)
    ytest.to_csv(out_dir / 'ytest.csv', index=False, header=True)

    print(f'Original shape: {before_shape}')
    print(f'Duplicate rows removed: {duplicate_count}')
    print(f'Cleaned shape: {df.shape}')
    print(f'Train shape: {Xtrain.shape}; Test shape: {Xtest.shape}')
    print(f'Train target rate: {ytrain.mean():.2%}; Test target rate: {ytest.mean():.2%}')
    print('Saved Xtrain.csv, Xtest.csv, ytrain.csv and ytest.csv')

if __name__ == '__main__':
    prepare_data()
