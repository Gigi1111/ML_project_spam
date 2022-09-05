## 1. Try perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron(shuffle=True, random_state=0)
perceptron.fit(X_train, y_train)
# model_summary(perceptron, "perceptron", X_train, y_train, X_test, y_test)


model = keras.models.Sequential()
model.add(keras.layers.Dense(8, activation="relu", input_shape=([X_train.shape[-1]])))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(8, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(
    optimizer=optimizers.Adam(1e-2),
    loss= "mean_squared_error",
    metrics=[
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.Accuracy()
    ],
)

# X_train_train, X_train_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.10, random_state=64)
callback_early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto')

model.fit(
    X_train,
    y_train,
    batch_size=8,
    epochs=10,
    verbose=1,
    shuffle = True,
    callbacks=[callback_early_stopping]
#     validation_data=(X_train_val, y_test_val),
#     class_weight=class_weight,
)