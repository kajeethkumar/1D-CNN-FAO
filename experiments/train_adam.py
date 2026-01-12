from tensorflow.keras.optimizers import Adam
from experiments.train_fao import model, X_train, y_train, X_val, y_val

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val)
)
