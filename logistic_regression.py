"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

X=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y=np.array([0,0,0,0,0,1,1,1,1,1])

model=LogisticRegression()
model.fit(X,y)

X_test = np.linspace(0,12,100).reshape(-1,1)
y_prob=model.predict_proba(X_test)[:,1]

plt.scatter(X,y,color='red',label='Data')
plt.plot(X_test,y_prob,color='blue',label='Logistic Regression')
plt.xlabel('Hours studied')
plt.ylabel('Probability')
plt.title('Logistic Regression Example')
plt.legend()
plt.grid()
plt.show()
w=model.coef_[0][0]
b=model.intercept_[0]
print(f'MOdel equation: Pass=1/(1+exp(-({w:.2f}*x+{b:.2f})))')
y_pred_rounded = np.round(model.predict(X))  # Round predicted values to integers
print(f"Accuracy score (rounded predictions): {accuracy_score(y, y_pred_rounded)}")"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

# Load dataset (Iris as example)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]  # Use only first two features for visualization
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy and Confusion Matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)

# Plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'blue', 'green']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['red', 'blue', 'green']))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
