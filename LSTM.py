#Keras LSTM for Time Series Forecasting
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


X = np.array([[[0.1], [0.2], [0.3]], [[0.2], [0.3], [0.4]]])
y = np.array([[0.4], [0.5]])

model = Sequential([
    LSTM(10, input_shape=(3, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

test_input = np.array([[[0.3], [0.4], [0.5]]])
print("Prediction:", model.predict(test_input))