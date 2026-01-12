import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

from utils.seed import set_seed
from utils.data_loader import load_kaggle_mitbih
from preprocessing.dwt_denoising import dwt_denoise
from preprocessing.normalization import normalize_ecg
from model.cnn_model import build_cnn
from model.fractional_adam import FractionalAdam
from evaluation.metrics import compute_metrics

set_seed()

X_train, X_val, X_test, y_train, y_val, y_test = load_kaggle_mitbih(
    "data/mitbih_train.csv",
    "data/mitbih_test.csv"
)

X_train = normalize_ecg([dwt_denoise(x) for x in X_train])
X_val   = normalize_ecg([dwt_denoise(x) for x in X_val])
X_test  = normalize_ecg([dwt_denoise(x) for x in X_test])

X_train = np.array(X_train).reshape(-1, 186, 1)
X_val   = np.array(X_val).reshape(-1, 186, 1)
X_test  = np.array(X_test).reshape(-1, 186, 1)

y_train = to_categorical(y_train, 5)
y_val   = to_categorical(y_val, 5)
y_test  = to_categorical(y_test, 5)
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
print('Model Loading and Training...')
model = build_cnn()
model.compile(
    optimizer=FractionalAdam(fraction=0.7),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print(model.summary())
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val)
)

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

metrics = compute_metrics(y_true, y_pred)
pd.DataFrame([metrics]).to_csv("results/fao_metrics.csv", index=False)
