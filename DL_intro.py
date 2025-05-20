from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model= Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(300, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
model.evaluate(X_test, y_test)

#Visualizing predictions
import matplotlib.pyplot as plt
import numpy as np

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

for i in range(5):
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'predicted: {predicted_labels[i]}, true: {true_labels[i]}')
    plt.axis('off')
    plt.show()