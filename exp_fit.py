import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define an exponential function for fitting
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

file = '7'

# Load data
data = np.loadtxt(f"./data/{file}.txt")
x = data[:, 0]  # Time (t)
y = data[:, 1]  # Amplitude (T)

# Specify the range for fitting (e.g., indices 20 to 50)
start_idx, end_idx = 800, 2271
x_fit = x[start_idx:end_idx]
y_fit = y[start_idx:end_idx]

# Fit the exponential function to the specified range
initial_guess = (np.max(y_fit)-np.min(y_fit), -0.0002, np.min(y_fit))
bounds = ([1e-6, -np.inf, 1e-6], [np.inf, np.inf, np.inf])
popt, pcov = curve_fit(exponential_func, x_fit, y_fit, bounds=bounds, p0=initial_guess, maxfev = 1000000000)
a, b, c = popt

# Extract the uncertainty (standard deviation) of parameter b
b_uncertainty = np.sqrt(pcov[1, 1])  # The variance of 'b' is at pcov[1, 1]
print(f"b = {popt[1]:.7f} ± {b_uncertainty:.7f}")

# Calculate fitted y values
y_fitted = exponential_func(x_fit, *popt)

# Calculate R^2 (coefficient of determination)
residuals = y_fit - y_fitted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
r_squared = 1 - (ss_res / ss_tot)

# Plot the data and the fitted curve
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data', color='blue', s=10)  # Original data
plt.plot(x_fit, y_fitted, label=f'Fitted: y = {a:.5f}*exp({b:.5f}*x) + {c:.5f}\n$R^2$ = {r_squared:.5f}', color='red')
plt.axvspan(x[start_idx], x[end_idx - 1], color='gray', alpha=0.2, label='Fit Range')

# Labeling the plot
plt.title('Exponential Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig(f'./{file}_ffff.png')
plt.show()
