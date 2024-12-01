from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt

# Exponential Function
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Residuals for Least Squares
def residuals(params, x, y):
    a, b, c = params
    return exponential_func(x, a, b, c) - y

file = '1'

# Load data
data = np.loadtxt(f"./data/{file}.txt")
x = data[:, 0]  # Time (t)
y = data[:, 1]  # Amplitude (T)

# Fit Range
start_idx, end_idx = 400, 800
x_fit = x[start_idx:end_idx]
y_fit = y[start_idx:end_idx]

# Initial Guess and Bounds
initial_guess = [max(np.max(y_fit), 1e-3), -0.5, max(np.min(y_fit), 1e-3)]
bounds = ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf])  # Allow negative decay rates, but other parameters are positive

# Curve Fit
result = least_squares(residuals, x0=initial_guess, args=(x_fit, y_fit), bounds=bounds)
popt = result.x  # Fitted parameters

# Predicted values
y_pred = exponential_func(x_fit, *popt)

# Calculate R^2
ss_res = np.sum((y_fit - y_pred) ** 2)
ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Display the R^2 value on the plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x_fit, exponential_func(x_fit, *popt), color='red', 
         label=f'Fit: y = {popt[0]:.3f} * exp({popt[1]:.3f} * x) + {popt[2]:.3f}\nR² = {r_squared:.3f}')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Exponential Fit')
plt.show()
