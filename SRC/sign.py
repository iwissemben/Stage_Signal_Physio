import numpy as np
import matplotlib.pyplot as plt

# signal generation
# Fs = 2000
# t_res = 1/Fs
# f0 = 100  # signal freq
# amplitude0 = 1
# N = int(100*Fs/f0)  # number of samples"""


def generate_sine_wave(Fs, amplitude, signal_freq, duration):
    t_res = 1/Fs
    N = int(duration*Fs/signal_freq)
    time_vect = np.linspace(0, (N-1)*t_res, N)  # time vector
    amplitudes_vect = amplitude*np.sin(2*np.pi*signal_freq*time_vect)
    amplitudes_vect2 = amplitude/10*np.sin(2*np.pi*signal_freq*time_vect)
    amplitudes_vect[(N-1)//2:] = amplitudes_vect2[(N-1)//2:]
    return time_vect, amplitudes_vect


def fft_compute_on_single_channel2(signal, Fs):
    N = len(signal)
    f_res = Fs/N  # freq res

    freq_vect = np.linspace(0, (N-1)*f_res, N)
    amplitude_fft = np.fft.fft(signal)
    amplitude_fft_magnitude = np.abs(amplitude_fft)/N

    freq_vect_for_plot = freq_vect[0:int(N/2+1)]  # +1 cf slicing
    amplitude_fft_magnitude_for_plot = 2*amplitude_fft_magnitude[0:int(N/2+1)]
    return freq_vect_for_plot, amplitude_fft_magnitude_for_plot


def compute_fft_on_all_channels2(channels_signals: np.ndarray, Fs: int | float):
    print(np.shape(channels_signals))
    channels_fft_frequencies = []
    channels_fft_magnitudes = []

    for signal in channels_signals.T:
        frequencies, magnitudes = fft_compute_on_single_channel2(signal, Fs)
        channels_fft_frequencies.append(frequencies)
        channels_fft_magnitudes.append(magnitudes)

    channels_fft_frequencies = np.transpose(np.array(channels_fft_frequencies))
    channels_fft_magnitudes = np.transpose(np.array(channels_fft_magnitudes))
    print(np.shape(channels_fft_frequencies))

    return {"FFT_frequencies": channels_fft_frequencies,
            "FFT_magnitudes": channels_fft_magnitudes}


"""times, amplitudes = generate_sine_wave(
    Fs=Fs, amplitude=amplitude0, signal_freq=f0, duration=100)
frequencies, magnitudes = fft_compute_on_single_channel2(
    Fs=Fs, amplitudes=amplitudes)
frequencies1, magnitudes1 = fft_compute_on_single_channel2(
    Fs=Fs, amplitudes=amplitudes[:(len(amplitudes)//2)])
frequencies2, magnitudes2 = fft_compute_on_single_channel2(
    Fs=Fs, amplitudes=amplitudes[(len(amplitudes)//2):])


# plots
# signal
figure, axis = plt.subplots(4, layout="constrained")
axis[0].set_title("whole time signal")
axis[0].plot(times, amplitudes, '.-')
axis[1].set_title("whole signal FFT (signal leakage)")
axis[1].plot(frequencies, magnitudes, '.-')
axis[2].set_title("first half signal FFT (original amplitude)")
axis[2].plot(frequencies1, magnitudes1, '.-')
axis[3].set_title("second half signal FFT (changed amplitude)")
axis[3].plot(frequencies2, magnitudes2, '.-')
plt.show()
"""
