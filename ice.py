import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data: y, uy, x, ux
data = np.array([
    [0.0186375, 1.44338E-05, 0.014179075, 0.000118614],
    [0.01615, 1.44338E-05, 0.012153492, 0.000106037],
    [0.03275, 1.44338E-05, 0.024675273, 0.000188236],
])

# Extract values
y, uy, x, ux = data.T
uy_eff = np.sqrt(uy ** 2 + 1.3319 ** 2 * ux ** 2)

# Define the linear function
def linear_model(x, a, b):
    return a * x + b

# Perform weighted linear regression using curve_fit
popt, pcov = curve_fit(
    linear_model,
    x, y,
    sigma=uy_eff,  # Weights from uncertainties in y
    absolute_sigma=True,
)

# Extract fitted parameters and uncertainties
a, b = popt
a_err, b_err = np.sqrt(np.diag(pcov))

# Calculate the correlation coefficient
correlation_matrix = np.corrcoef(x, y)
correlation_coefficient = correlation_matrix[0, 1]

# Generate points for the regression line and confidence interval
x_fit = np.linspace(min(x), max(x), 500)
y_fit = linear_model(x_fit, a, b)
y_upper = linear_model(x_fit, a + a_err, b + b_err)
y_lower = linear_model(x_fit, a - a_err, b - b_err)

# Plot the data and regression line
plt.figure(figsize=(10, 6))

# Scatter plot of data with error bars
plt.errorbar(x, y, xerr=ux, yerr=uy, fmt='o', label='Data', color='blue', capsize=3)

# Regression line
plt.plot(x_fit, y_fit, label=f'Regression: y = {a:.3f}x + {b:.3f}', color='red')

# Confidence interval
plt.fill_between(x_fit, y_lower, y_upper, color='red', alpha=0.2, label='68% Confidence Interval')

# Labels, title, legend
plt.xlabel(r'$\frac{\Delta T A}{\Delta x L}\Delta t$ (kg·m·K/W)')
plt.ylabel(r'$\Delta m$ (kg)')
# plt.title(r'$\Delta m - \frac{\Delta T A}{\Delta x L}\Delta t$')
plt.legend()

# Show plot
plt.grid()
plt.show()

# Print regression results
print(f"Regression equation: y = {a:.6f}x + {b:.6f}")
print(f"Uncertainty in a: {a_err:.6f}")
print(f"Uncertainty in b: {b_err:.6f}")
print(f"Correlation coefficient: {correlation_coefficient:.6f}")
