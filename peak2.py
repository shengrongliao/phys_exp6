import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, find_peaks
from scipy.interpolate import interp1d

def plot_envelope(file):
    # Read data from a document file (e.g., 'data.txt')
    data = np.loadtxt(file)

    # Example: Non-uniform x data
    x = data[:, 0] # t
    y = data[:, 1] # T

    # # Adjust find_peaks parameters
    # peaks, _ = find_peaks(y, distance=20, prominence=0.1)
    # valleys, _ = find_peaks(-y, distance=20, prominence=0.1)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Original Data', color='blue', alpha=0.7)
    # print(peaks)
    # plt.plot(x, peaks, label='Upper Envelope', color='red', linestyle='--')
    # plt.plot(x, valleys, label='Lower Envelope', color='green', linestyle='--')
    plt.xlabel('t')
    plt.ylabel('T')
    plt.legend()
    plt.grid()
    plt.show()

plot_envelope('./data/Si.txt')