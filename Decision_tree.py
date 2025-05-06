'''from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits= load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")'''

# Regression with DiscisionTreeRegressor

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Generate training data
x = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(x).ravel()
y[::5] += 1 + 0.2 * np.random.randn(16)  # add noise

# Train model
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(x, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = regressor.predict(X_test)

# Evaluate
accuracy = r2_score(np.sin(X_test).ravel(), y_pred)
print(f"R² Score: {accuracy:.2f}")

# Plot
plt.figure()
plt.scatter(x, y, label="Training data")
plt.plot(X_test, y_pred, color="red", label="Prediction")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
