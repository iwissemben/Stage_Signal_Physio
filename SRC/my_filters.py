import os
import matplotlib.pyplot as plt
import numpy as np
from my_functions import *
# library for creating filters
from scipy.signal import butter, iirnotch, filtfilt, welch, freqz, freqs, TransferFunction


def plot_filter_frequency_response(fig_number, fig_title: str, frequencies: np.ndarray, magnitude: np.ndarray, order: int, fc1: float, fc2: float = None):
    """
    Plots the frequency response of the chosen filter.
    """
    figure, axis = plt.subplots(
        num=fig_number, figsize=(6, 3), layout='constrained')
    axis.plot(frequencies, abs(magnitude), color="blue")
    # axis.plot(fig_number,figsize=(8, 6))
    axis.axhline(0.7, linestyle="dotted", color='green',
                 label="G="+str(-3)+"dB")  # first cutoff frequency
    axis.axvline(fc1, linestyle="--", color='red',
                 label="fc1="+str(fc1))  # first cutoff frequency
    if fc2 is not None:
        axis.axvline(fc2, linestyle="--", color='orange',
                     label="fc2="+str(fc2))  # second cutoff frequency

    axis.grid(True, which="both")
    axis.set_title(fig_title+" order: "+str(order))
    axis.set_ylabel("Magnitude")
    axis.set_xlabel("Frequency (Hz)")
    # axis.set_ylim([-25, 10])
    plt.legend()
    # plt.show()

    # fig.savefig("./DAT/OUTPUT/fig_title"+".svg", format="svg", dpi=1200)


def low_pass_filter(input_signal, sample_rate: float, cutoff_freq: float, filter_order: int):
    """
    Creates a low pass filter and filters the input signal.
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis
    """
    # Creation of low-pass filter
    b_low_pass, a_low_pass = butter(filter_order, cutoff_freq,
                                    btype='lowpass', fs=sample_rate)
    # filtering the signal if there is an input signal
    if input_signal is not None:
        output_signal = filtfilt(b_low_pass, a_low_pass, input_signal, axis=0)
    else:
        output_signal = None
    # Frequency response of the filter
    freq, h = freqz(b_low_pass, a_low_pass, fs=sample_rate)
    return output_signal, freq, h


def high_pass_filter(input_signal, sample_rate: float, cutoff_freq: float, filter_order: int):
    """
    Creates a high pass filter and filters the input signal.
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis
    """
    # Creation of high-pass filter
    b_high_pass, a_high_pass = butter(filter_order, cutoff_freq,
                                      btype='highpass', fs=sample_rate)
    # filtering the signal if there is an input signal
    if input_signal is not None:
        output_signal = filtfilt(
            b_high_pass, a_high_pass, input_signal, axis=0)
    else:
        output_signal = None
    # Frequency response of the filter
    freq, h = freqz(b_high_pass, a_high_pass, fs=sample_rate)
    return output_signal, freq, h


def band_pass_filter(input_signal, sample_rate: float,
                     low_cutoff_freq: float, high_cutoff_freq: float, filter_order: int):
    """
    Creates a regular band_pass filter and filters the input signal.
    Uses the bandpass type of the scipy.signal.butter() function
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis
    """
    # Creation of band-pass filter
    b_band_pass, a_band_pass = butter(filter_order, [
                                      low_cutoff_freq, high_cutoff_freq],
                                      fs=sample_rate, btype='bandpass')

    # filtering the signal if there is an input signal
    if input_signal is not None:
        output_signal = filtfilt(
            b_band_pass, a_band_pass, input_signal, axis=0)
    else:
        output_signal = None

    # Frequency response of the filter
    freq, h = freqz(b_band_pass, a_band_pass, fs=sample_rate)
    return output_signal, freq, h


def custom_band_pass_filter(input_signal, sample_rate: float,
                            low_cutoff_freq: float, high_cutoff_freq: float, filter_order: int):
    """
    Custom made band_pass filter using a combination of Low Pass and High Pass filter.
    The input signal is filtered by the filter resulting of the convolution of the two filters mentioned above.
    Note: the signal is not filtered by one then the other but by the combination.
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis
    """
    # Creation of High-Pass and Low-Pass filters
    b_high_pass, a_high_pass = butter(filter_order, low_cutoff_freq,
                                      btype='highpass', fs=sample_rate)

    b_low_pass, a_low_pass = butter(filter_order, high_cutoff_freq,
                                    btype='lowpass', fs=sample_rate)

    b_band = np.convolve(b_high_pass, b_low_pass)
    a_band = np.convolve(a_high_pass, a_low_pass)

    # filtering the signal if there is an input signal
    if input_signal is not None:
        output_signal = filtfilt(b_band, a_band, input_signal, axis=0)
    else:
        output_signal = None

    # Frequency response of the filter
    freq, h = freqz(b_band, a_band, fs=sample_rate)
    return output_signal, freq, h


def notch_filter(input_signal, sample_rate: float, cutoff_freq: float, stop_band_width: float):
    """
    Creates a notch filter and filters the input signal.
    The stop-band width (in Hz) is centered on the cutoff frequency .
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis
    If no signal is inputted into the function, it still generates the filter and its frequency response from other arguments.
    """
    # Q is the quality factor (-3db point).
    Q = cutoff_freq/stop_band_width

    # Creation of notch filter
    b_notch, a_notch = iirnotch(cutoff_freq, Q, sample_rate)

    # filtering the signal if there is an input signal
    if input_signal is not None:
        output_signal = filtfilt(b_notch, a_notch, input_signal, axis=0)
    else:
        output_signal = None

    # Frequency response of the filter
    freq, h = freqz(b_notch, a_notch, fs=sample_rate)

    return output_signal, freq, h
