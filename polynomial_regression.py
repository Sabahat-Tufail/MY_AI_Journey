'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

X= np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y=np.array([2,8,18,32,50,72,98,128,162,200])

poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X)

model=LinearRegression()
model.fit(X_poly,y)
X_test=np.linspace(1,10,100).reshape(-1,1)
X_test_poly=poly.transform(X_test)
y_pred=model.predict(X_test_poly)

plt.scatter(X,y,color='red',label='Actual Data')
plt.plot(X_test,y_pred,color='blue',label='Polynomial Fit')
plt.xlabel("Hours studied")
plt.ylabel("Marks obtained")
plt.title("Polynomial Regression")
plt.legend()
plt.show()


print(f"R² score: {r2_score(y, model.predict(X_poly))}")'''

'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Data
X = np.array([1, 2, 3]).reshape(-1, 1)
y = np.array([2, 4, 6])  # This is actually a linear pattern

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit model
model = LinearRegression()
model.fit(X_poly, y)

# Predict on same test data
X_test = np.array([1,4,58, 5, 3]).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

# For smooth curve
X_curve = np.linspace(1, 3, 100)
X_curve_poly = poly.transform(X_curve.reshape(-1, 1))
y_curve = model.predict(X_curve_poly)

# Plot
plt.scatter(X, y, color='red', label='True')
plt.plot(X_curve, y_curve, color='blue', label='Polynomial Fit')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

# R² Score
print(f"R² score: {r2_score(y, model.predict(X_poly))}")'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score ,classification_report,confusion_matrix

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\manufacturing.csv")

# Features and target
X = df[["Temperature (°C)", "Pressure (kPa)"]]
y = df["Quality Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred = model.predict(X_test_poly)

print("R² Score (training):", r2_score(y_train, model.predict(X_train_poly)))
print("R² Score (test):", r2_score(y_test, y_pred))
# Check for missing values in the dataset
print(df.isnull().sum())
df = df.fillna(df.mean())
print(df.duplicated().sum())
df = df.drop_duplicates()
df["Temperature (°C)"] = df["Temperature (°C)"].astype(float)
# Check unique values for categorical columns
print(df["Pressure (kPa)"].value_counts())

# Handle outliers (example: cap temperature values)
df["Temperature (°C)"] = df["Temperature (°C)"].clip(lower=0, upper=100)



# Ensure no negative values in Pressure (kPa)
df = df[df["Pressure (kPa)"] >= 0]

# Print cleaned data
print(df.head())
print("R² Score (test):", r2_score(y_test, y_pred))
