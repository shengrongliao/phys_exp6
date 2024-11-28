import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, find_peaks
from scipy.interpolate import interp1d

# Read data from a document file (e.g., 'data.txt')
data = np.loadtxt('./data/A1801.txt')

# Example: Non-uniform x data
x = data[:, 0] # t
y = data[:, 1] # T

# Adjust find_peaks parameters
peaks, _ = find_peaks(y, distance=20, prominence=0.1)
valleys, _ = find_peaks(-y, distance=20, prominence=0.1)

# Interpolate envelopes
upper_envelope = interp1d(x[peaks], y[peaks], kind='cubic', fill_value="extrapolate")(x)
lower_envelope = interp1d(x[valleys], y[valleys], kind='cubic', fill_value="extrapolate")(x)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Original Data', color='blue', alpha=0.7)
plt.plot(x, upper_envelope, label='Upper Envelope', color='red', linestyle='--')
plt.plot(x, lower_envelope, label='Lower Envelope', color='green', linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Envelope Analysis with Adjusted Peak Detection')
plt.legend()
plt.grid()
plt.show()
