import numpy as np
import matplotlib.pyplot as plt

def fft_plot(file):
    # Load data
    data = np.loadtxt(f"./data/{file}.txt")
    x = data[:, 0]  # Time (t)
    y = data[:, 1]  # Amplitude (T)

    # FFT computation
    fft_result = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(y), d=(x[1] - x[0]))

    # Define frequency range for filtering
    freq_min, freq_max = 0.0025, 1
    freq_mask = ((frequencies >= freq_min) & (frequencies <= freq_max)) | ((frequencies <= -freq_min) & (frequencies >= -freq_max))

    # Apply frequency filter
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[freq_mask] = fft_result[freq_mask]

    # Perform inverse FFT to obtain filtered signal
    filtered_signal = np.fft.ifft(filtered_fft)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Original Signal
    plt.subplot(3, 1, 1)
    plt.plot(x, y, label='Original Signal', color='blue')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # Frequency Spectrum
    plt.subplot(3, 1, 2)
    # plt.plot(frequencies, np.abs(fft_result), label='Original Spectrum', color='blue', alpha=0.7)
    plt.plot(frequencies, np.abs(filtered_fft), label='Filtered Spectrum', color='orange', alpha=0.7)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.xlim(-0.01, 0.01)  # Adjust this range based on data
    plt.legend()

    # phase Spectrum
    plt.subplot(3, 1, 3)
    # plt.plot(frequencies, np.abs(fft_result), label='Original Spectrum', color='blue', alpha=0.7)
    plt.plot(frequencies, np.angle(filtered_fft), label='Filtered Spectrum', color='orange', alpha=0.7)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.xlim(-0.01, 0.01)  # Adjust this range based on data
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./{file}.png')
    plt.show()

for file in ['A1801', 'A1802', 'A1807', 'A1808']:
    fft_plot(file)