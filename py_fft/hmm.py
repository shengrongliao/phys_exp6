import numpy as np
import matplotlib.pyplot as plt

file = 'A1802'

# Load data
data = np.loadtxt(f"./data/{file}.txt")
x = data[:, 0]  # Time (t)
y = data[:, 1]  # Amplitude (T)

# Example FFT results (replace with your data)
fft_result = np.fft.fft(y)  # Your FFT result
frequencies = np.fft.fftfreq(len(y), d=(x[1] - x[0]))  # Your frequency array

# Calculate magnitude and argument
magnitude = np.abs(fft_result)
argument = np.angle(fft_result)

# Filter for display range (optional)
display_range = (frequencies >= -1) & (frequencies <= 1)
filtered_frequencies = frequencies[display_range]
filtered_magnitude = magnitude[display_range]
filtered_argument = argument[display_range]

# Plotting
plt.figure(figsize=(12, 6))

# Magnitude plot
plt.subplot(2, 1, 1)
plt.plot(filtered_frequencies, filtered_magnitude, label='Magnitude')
plt.title('FFT Magnitude Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(0.001, 0.02)
plt.xlim(0.001, 0.02)
plt.legend()

# Argument (phase) plot
plt.subplot(2, 1, 2)
plt.plot(filtered_frequencies, filtered_argument, label='Phase', color='orange')
plt.title('FFT Phase Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.xlim(0.001, 0.02)
plt.legend()

plt.tight_layout()
plt.show()