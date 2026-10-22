##Find the number of iterations for convergence
## find the slope and the intercept

import math

import numpy as np
import pandas as pd
from sklearn import linear_model


def gradient_descent(x, y):
    m_curr = b_curr = 0
    n = len(x)
    iterations = 10000
    learning_rate = 0.001
    previous_cost = float("inf")
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * np.sum((y - y_predicted) ** 2)
        ##Stop when prev cost and current cost is the same
        if math.isclose(float(cost), float(previous_cost), rel_tol=1e-9, abs_tol=1e-9):
            print(f"Converged at iteration {i}")
            print(f"Previous cost: {previous_cost}")
            print(f"Current cost:  {cost}")
            break

        md = (-2 / n) * np.sum(x * (y - y_predicted))
        bd = (-2 / n) * np.sum((y - y_predicted))
        m_curr = m_curr - (learning_rate * md)
        b_curr = b_curr - (learning_rate * bd)
        previous_cost = cost
        print("M {}, B {}, cost {}, iterations{}".format(m_curr, b_curr, cost, i))
    return m_curr, b_curr


df = pd.read_csv("test_scores.csv")

x = df["math"].to_numpy().reshape(-1, 1)
y = df["cs"].to_numpy().reshape(-1, 1)
reg = linear_model.LinearRegression()
reg.fit(x, y)


## Standardization
x_mean = np.mean(x)
x_std = np.std(x)
x_scaled = (x - x_mean) / x_std


##Running Algorithm
m_scaled, b_scaled = gradient_descent(x_scaled, y)

##Converting the data back to the original
m_original = m_scaled / x_std
b_original = b_scaled - m_original * x_mean
print("Slope:", reg.coef_[0])
print("Intercept:", reg.intercept_)
print("M_original ", m_original)
print("B_original: ", b_original)
# print("M_scaled ", m_scaled)
# print("B_scaled: ", b_scaled)

## With original data
# m,b=gradient_descent(x,y)
# print("M: ", m)
# print("B: ", b)
