'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score 

data={
    "Area": [1200,1500,1700,1800,2000,2200,2500,2700,3000,3200],
    "Bedrooms": [2,3,3,4,4,5,5,6,6,7],
    "Age" : [5,10,15,20,25,30,35,40,45,50],
    "Price": [300000,350000,400000,450000,500000,550000,600000,650000,700000,750000]
}

df=pd.DataFrame(data)
X=df[["Area","Bedrooms","Age"]]
y=df["Price"]

X_train ,X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared error : {mse}")

print("Intercept:",model.intercept_)
print("Coefficients: ",model.coef_)

new_house=np.array([[2000,3,5]])
predicted_price=model.predict(new_house)

print(f"Predicted price for one house : ${predicted_price[0]:,.2f}")
y_pred_rounded = np.round(model.predict(X))  # Round predicted values to integers
print(f"Accuracy score (rounded predictions): {accuracy_score(y, y_pred_rounded)}")'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Multivariate_Linear_Regression.csv")

# Feature and target selection
X = df[["X1", "X2", "X3", "X4"]]
y = df["y"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict on new data
new = np.array([[20, 4, 3.0047727, 27]])
predicted = model.predict(new)
print(f"Predicted Y : {predicted[0]:,.2f}")
