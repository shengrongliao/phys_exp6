import numpy as np
import matplotlib.pyplot as plt

file = 'A1807'

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

# Define frequency range for filtering
freq_min = 0.0025
freq_mask = ~((-freq_min <= frequencies) & (frequencies <= freq_min))

# Apply frequency filter
filtered_fft = np.zeros_like(fft_result, dtype=complex)
filtered_fft[freq_mask] = fft_result[freq_mask]
fft_result = filtered_fft

# Plotting
plt.figure(figsize=(12, 8))

# # Original Signal
# plt.subplot(3, 1, 1)
# plt.plot(x, y, label='Original Signal', color='blue')
# plt.title('Original Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()

# Frequency Spectrum
plt.subplot(2, 1, 1)
# plt.plot(frequencies, np.abs(fft_result), label='Original Spectrum', color='blue', alpha=0.7)
plt.plot(frequencies, np.abs(fft_result), label='Filtered Spectrum', color='orange', alpha=0.7)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.xlim(0.0045, 0.0065)  # Adjust this range based on data
plt.legend()

# phase Spectrum
plt.subplot(2, 1, 2)
# plt.plot(frequencies, np.abs(fft_result), label='Original Spectrum', color='blue', alpha=0.7)
plt.plot(frequencies, np.angle(fft_result), label='Filtered Spectrum', color='orange', alpha=0.7)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Phase')
plt.xlim(0.0045, 0.0065)  # Adjust this range based on data
plt.legend()

plt.tight_layout()
plt.savefig(f'./{file}_zoomin.png')
plt.show()