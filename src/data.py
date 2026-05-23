import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"]   = scaler.fit_transform(df[["Time"]])
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X,y,scaler

def split_data(X, y):
    # Séparer le test en premier
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,           # garde le ratio de fraudes dans chaque split
        random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test