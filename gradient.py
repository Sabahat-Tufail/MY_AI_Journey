import numpy as np
import matplotlib.pyplot as plt
x=np.array([500,1000,1500,2000,2500])
y=np.array([100,150,200,250,300])

x=x/1000

def hypothesis(x,theta0,theta1):
    return theta0 + theta1 * x

def compute_cost(x,y,theta0,theta1):
    m=len(x)
    predictions=hypothesis(x,theta0,theta1)
    return (1/(2*m)) * np.sum((predictions-y)**2)

def gradient_descent(x,y,theta0,theta1,alpha,iterations):
    m=len(x)
    cost_history = []
    for i in range(iterations):
        predictions = hypothesis(x, theta0, theta1)
        error = predictions - y

        d_theta0 = (1/m) * np.sum(error)
        d_theta1 = (1/m) * np.sum(error * x)
        theta0 -= alpha * d_theta0

        theta1 -= alpha * d_theta1
        cost = compute_cost(x, y, theta0, theta1)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}, theta0 {theta0}, theta1 {theta1}")

    return theta0, theta1, cost_history
theta0 = 0
theta1 = 0
alpha = 0.1
iterations = 1000
theta0, theta1, cost_history = gradient_descent(x, y, theta0, theta1, alpha, iterations)
print(f"Final theta0: {theta0}, Final theta1: {theta1}")