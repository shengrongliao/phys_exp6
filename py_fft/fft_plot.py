import numpy as np
import matplotlib.pyplot as plt

def fft_plot(file1, file2):
    # Load data
    data = np.loadtxt(f"./data/{file1}.txt")
    x1 = data[:, 0]  # Time (t)
    y1 = data[:, 1]  # Amplitude (T)

    data = np.loadtxt(f"./data/{file2}.txt")
    x2 = data[:, 0]  # Time (t)
    y2 = data[:, 1]  # Amplitude (T)

    # FFT computation
    fft_result1 = np.fft.fft(y1)[:1000]
    fft_result2 = np.fft.fft(y2)[:1000]
    frequencies = np.fft.fftfreq(len(y1), d=(x1[1] - x1[0]))[:1000]

    # # Define frequency range for filtering
    # freq_min = 0.0025
    # freq_mask = ~((-freq_min <= frequencies) & (frequencies <= freq_min))

    fft_result = fft_result1 / fft_result2

    print("hahahaha")
    for i in range(20):
        print(f"{np.abs(fft_result1[i]):.8f} {frequencies[i]:.8f} {np.abs(fft_result[i]):.8f} {np.angle(fft_result[i]):.8f}")

    # # Apply frequency filter
    # filtered_fft = np.zeros_like(fft_result, dtype=complex)
    # filtered_fft[freq_mask] = fft_result[freq_mask]
    # fft_result = filtered_fft

    # Plotting
    plt.figure(figsize=(12, 8))

    # Original Signal
    # plt.subplot(3, 1, 1)
    # plt.plot(x, y, label='Original Signal', color='blue')
    # plt.title('Original Signal')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.legend()

    # Frequency Spectrum
    plt.subplot(2, 1, 1)
    # plt.plot(frequencies, np.abs(fft_result), label='Original Spectrum', color='blue', alpha=0.7)
    plt.plot(frequencies, np.abs(fft_result), label='Filtered Spectrum', color='blue', alpha=0.7)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.xlim(0.0045, 0.0065)  # Adjust this range based on data
    plt.ylim(0, 25)
    plt.grid(True)
    plt.legend()

    # phase Spectrum
    plt.subplot(2, 1, 2)
    # plt.plot(frequencies, np.abs(fft_result), label='Original Spectrum', color='blue', alpha=0.7)
    plt.plot(frequencies, np.angle(fft_result), label='Filtered Spectrum', color='orange', alpha=0.7)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.xlim(0.0045, 0.0065)  # Adjust this range based on data
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # plt.savefig(f'./{file1}_AAA.png')
    plt.show()

for file1, file2 in [['A2', 'A1'], ['A7', 'A8']]:
    fft_plot(file1, file2)