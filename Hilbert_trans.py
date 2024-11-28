import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Read data from a document file (e.g., 'data.txt')
data = np.loadtxt('./data/A1801.txt')

# Example: Non-uniform x data
x = data[:, 0] # t
y = data[:, 1] # T

from scipy.signal import butter, filtfilt

# High-pass filter design
cutoff = 0.01  # Adjust cutoff frequency
order = 2
b, a = butter(order, cutoff, btype='high', analog=False)

# Apply the high-pass filter
filtered_y = filtfilt(b, a, y)

# Apply Hilbert transform to filtered data
analytic_signal = hilbert(filtered_y)
hilbert_envelope = np.abs(analytic_signal)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Original Data', color='blue', alpha=0.7)
plt.plot(x, hilbert_envelope, label='Envelope (Filtered)', color='orange', linestyle='--', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Envelope Analysis with High-Pass Filter')
plt.legend()
plt.grid()
plt.show()
