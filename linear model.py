'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score  # Keep accuracy_score import

# Data preparation
X = np.array([150, 160, 170, 180, 190, 200]).reshape(-1, 1)
Y = np.array([50, 60, 70, 80, 90, 100])

# Train the model
model = LinearRegression()
model.fit(X, Y)

# Get the model parameters
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope: {slope}, Intercept: {intercept}")

# Predict the weight for height 175 cm
height = 175
predicted_weight = model.predict([[height]])
print(f"Predicted weight for height {height} cm: {predicted_weight[0]} kg")

# Plotting the data points and regression line
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight')
plt.legend()
plt.grid(True)
plt.show()

# Use accuracy_score, but rounding the predicted values to integers
y_pred_rounded = np.round(model.predict(X))  # Round predicted values to integers
print(f"Accuracy score (rounded predictions): {accuracy_score(Y, y_pred_rounded)}")'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\archive\Salary_dataset.csv")

# Drop the 'Unnamed: 0' column as it's not needed
df = df.drop(columns=['Unnamed: 0'])

# Split features and target
X = df[['YearsExperience']]  # Feature: Years of experience
y = df['Salary']  # Target: Salary

# Train the model
model = LinearRegression()
model.fit(X, y)

# Model parameters
print(f"Slope: {model.coef_[0]}, Intercept: {model.intercept_}")

# Predict salary for a given years of experience
experience = 5  # Example input for years of experience
predicted_salary = model.predict(pd.DataFrame({'YearsExperience': [experience]}))
print(f"Predicted salary for {experience} years of experience: {predicted_salary[0]}")

# Plot data and regression line
plt.scatter(X, y, color='blue', alpha=0.3, label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs Salary')
plt.legend()
plt.grid(True)
plt.show()

# Calculate R² score
df = df.drop_duplicates(subset=['YearsExperience'], keep='first')
df = df.reset_index(drop=True)



r2 = r2_score(y, model.predict(X))
print(f"R² score: {r2}")





