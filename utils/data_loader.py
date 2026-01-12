import pandas as pd
from sklearn.model_selection import train_test_split

def load_kaggle_mitbih(train_path, test_path, val_ratio=0.1, seed=42):
    train_df = pd.read_csv(train_path, header=None)
    test_df  = pd.read_csv(test_path, header=None)

    X_train_full = train_df.iloc[:, :186].values
    y_train_full = train_df.iloc[:, 187].values

    X_test = test_df.iloc[:, :186].values
    y_test = test_df.iloc[:, 187].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_ratio,
        stratify=y_train_full,
        random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
