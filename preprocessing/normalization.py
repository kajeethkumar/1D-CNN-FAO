import numpy as np

def normalize_ecg(X):
    X = np.asarray(X)
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)
