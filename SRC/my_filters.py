import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import Optional, Union, Literal, Tuple, List, Any
# library for creating filters
from scipy.signal import butter, iirnotch, filtfilt, welch, freqz, freqs, TransferFunction

# =============================================================================
########################## Cutoff frequency corrector  ########################
# =============================================================================


def filtfilt_cutoff_frequency_corrector(order: int, cutoff_freq: Union[float, int], sampling_freq: Union[float, int], pass_type: Literal['low_pass', 'high_pass']) -> float:
    """
    Function that corrects cutoff frequencies to use in combination with filt.filt()

    As a zero-phase filter (linear filter) is applied to a signal the cutoff freq are diminished. The correction depends also on the order. 
    The adjustment is made on the angular cutoff frequency, which depends also on the filter direction (LP,HP).

    SPECIFY the type of the filter with `pass_type` at either "low_pass" or "high_pass"

    Parameters:
    -----------
        - `order` (int): Order of the filter to use
        - `cutoff_freq` (Union[float, int]): Desired cutoff frequency (ideal)
        - `sampling_freq` (Union[float, int]) : Signal sampling frequenct
        - `pass_type` (str): Type of the pass  filter (high or low pass) as the correction depends on the filter's direction.

    Returns:
    -----------
        - `f_cutoff_corrected` (float): corrected frequency

    # Source of the correction method:
        - Biomechanics and Motor Control of Human Movement, 4th-edition (page 69)
        - https://www.codeproject.com/Articles/1267916/Multi-pass-Filter-Cutoff-Correction
    """
    filter_passes = 2
    Un = False
    C = np.power((np.power(2, (1/filter_passes))-1),
                 (1/(2*order)))  # ordre 4:0.25
    Wc = 2*np.pi*cutoff_freq  # angular cutoff frequency
    Uc = np.tan(Wc/(2*sampling_freq))  # adjusted angular cutoff frequency
    if pass_type == "low_pass":
        Un = Uc / C  # David A. Winter correction
    if pass_type == "high_pass":
        # Multiply by C for highpass (Research Methods in Biomechanics)
        Un = Uc * C
    f_cutoff_corrected = np.arctan(Un)*sampling_freq/np.pi
    f_cutoff_corrected = round(f_cutoff_corrected, 3)
    return f_cutoff_corrected


def plot_filter_frequency_response(fig_title: str, frequencies: np.ndarray, magnitude: np.ndarray, order: int,
                                   fc1: Union[float, int], fc2: Optional[Union[float, int]] = None, fig_number: Optional[int] = None) -> Axes:
    """
    Plots the frequency response of the chosen filter.

    Parameters:
    -----------
        - `fig_number` (int,optional) : Figure number 
        - `fig_title` (str): Figure title
        - `frequencies` (np.ndarray): Frequencies at which the frequency response  is computed
        - `magnitude` (np.ndarray): Magnitude of the frequency response for each frequency
        - `order` (int): order of the filter - for legends
        - `fc1` (int) : Cutoff frequency N°1
        - `fc2` (float,optional): Cutoff frequency N°2

    Returns:
    -----------
        - `figure` (Figure): Matplotlib figure
    """
    figure, axis = plt.subplots(
        num=fig_number, figsize=(6, 3), layout='constrained')
    axis.plot(frequencies, abs(magnitude), color="blue")
    # axis.plot(fig_number,figsize=(8, 6))
    axis.axhline(0.7, linestyle="dotted", color='green',
                 label="G="+str(-3)+"dB")  # filter gain at -3dB (Vs/Ve=70%) (should meet the curve at the cutoff freq)

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

    return axis


def low_pass_filter(sample_rate: Union[float, int], cutoff_freq: Union[float, int],
                    filter_order: int, input_signal: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Creates a Low pass Butterworth filter and applies it to filter the input signal.

    The filter is first created (using `scipy.butter`) and applied as a zero phase filter (using `scipy.filtfilt`). 
    The use of a zero-phase filter leads to a cutoff frequency shift.
    To compensate the shift, the desired frequency can be corrected (using `my_filters.filtfilt_cutoff_frequency_corrector`) 
    before passing it to the `cutoff_freq` variable.

    The `input_signal` argument can be a 2D array of signals, one per column. 
        - If specified, the created filter is applied each signal of the array it.
        - If not specified, the function only creates the desired filter.

    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequencies.

    Parameters:
    -----------
        - `input_signal` (np.ndarray, optional) : input signal on which to apply the filter. Can be a 2D array where each column (axis=0) is a signal.
        - `sample_rate` (Union[float, int]) : signal sampling rate
        - `cutoff_freq` (Union[float, int]) : cutoff frequency of the filter
        - `filter_order` (int): order of the filter

    Returns:
    -----------
        - `output_signal` (np.ndarray) : Filtered signal
        - `freq` (np.ndarray) : Frequencies for which the frequency response h is computed
        - `h` (np.ndarray) : Frequency response for each frequency

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


def high_pass_filter(sample_rate: Union[float, int], cutoff_freq: Union[float, int],
                     filter_order: int, input_signal: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Creates a High pass Butterworth filter and applies it to filter the input signal.

    The filter is first created (using `scipy.butter`) and applied as a zero phase filter (using `scipy.filtfilt`). 
    The use of a zero-phase filter leads to a cutoff frequency shift.
    To compensate the shift, the desired frequency can be corrected (using `my_filters.filtfilt_cutoff_frequency_corrector`) 
    before passing it to the `cutoff_freq` variable.

    The `input_signal` argument can be a 2D array of signals, one per column. 
        - If specified, the created filter is applied each signal of the array it.
        - If not specified, the function only creates the desired filter.

    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequencies.

    Parameters:
    -----------
        - `input_signal` (np.ndarray, optional) : input signal on which to apply the filter. Can be a 2D array where each column (axis=0) is a signal.
        - `sample_rate` (Union[float, int]) : signal sampling rate
        - `cutoff_freq` (Union[float, int]) : cutoff frequency of the filter
        - `filter_order` (int): order of the filter

    Returns:
    -----------
        - `output_signal` (np.ndarray) : Filtered signal
        - `freq` (np.ndarray) : Frequencies for which the frequency response h is computed
        - `h` (np.ndarray) : Frequency response for each frequency
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


def band_pass_filter(sample_rate: Union[float, int], low_cutoff_freq: Union[float, int],
                     high_cutoff_freq: Union[float, int], filter_order: int,
                     input_signal: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Creates a regular band_pass filter and filters the input signal.
    Uses the bandpass type of the `scipy.signal.butter()` function
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis

    Creates a band pass Butterworth filter and applies it to filter the input signal.

    The filter is first created (using `scipy.butter` with btype="bandpass") and applied as a zero phase filter (using `scipy.filtfilt`). 
    The use of a zero-phase filter leads to a cutoff frequency shift.
    To compensate the shift, the desired frequency can be corrected (using `my_filters.filtfilt_cutoff_frequency_corrector`) 
    before passing it to the `cutoff_freq` variable.

    The `input_signal` argument can be a 2D array of signals, one per column. 
        - If specified, the created filter is applied each signal of the array it.
        - If not specified, the function only creates the desired filter.

    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequencies.

    Parameters:
    -----------
        - `input_signal` (np.ndarray, optional) : input signal on which to apply the filter. Can be a 2D array where each column (axis=0) is a signal.
        - `sample_rate` (Union[float, int]) : signal sampling rate
        - `cutoff_freq` (Union[float, int]) : cutoff frequency of the filter
        - `filter_order` (int): order of the filter

    Returns:
    -----------
        - `output_signal` (np.ndarray) : Filtered signal
        - `freq` (np.ndarray) : Frequencies for which the frequency response h is computed
        - `h` (np.ndarray) : Frequency response for each frequency
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


def custom_band_pass_filter(sample_rate: Union[float, int], low_cutoff_freq: Union[float, int], high_cutoff_freq: Union[float, int],
                            filter_order: int, input_signal: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Custom made band_pass filter using a combination of Low Pass and High Pass filter.
    The input signal is filtered by the filter resulting of the convolution of the two filters mentioned above.

    Note: the signal is not filtered by one then the other but by the combination.
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis
    """

    """
    Creates a custom made band_pass filter using a combination of Low Pass and High Pass filter and filters the input signal.

    The custom filter is created from the convolution of a high-pass and low-pass Butterworth filter (created using `scipy.butter`).
    The resulting filter is then applied as a zero phase filter (using `scipy.filtfilt`) on the signal.
    The use of a zero-phase filter leads to a cutoff frequency shift.
    To compensate the shift, the desired frequency can be corrected (using `my_filters.filtfilt_cutoff_frequency_corrector`) 
    before passing it to the `cutoff_freq` variable.
        
    Note: 
    - The signal is not filtered by one then the other but by the combination.
    - The input signal is filtered by the filter resulting of the convolution of the two filters mentioned above.
    
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis

    The `input_signal` argument can be a 2D array of signals, one per column. 
        - If specified, the created filter is applied each signal of the array it.
        - If not specified, the function only creates the desired filter.

    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequencies.

    Parameters:
    -----------
        - `input_signal` (np.ndarray, optional) : input signal on which to apply the filter. Can be a 2D array where each column (axis=0) is a signal.
        - `sample_rate` (Union[float, int]) : signal sampling rate
        - `low_cutoff_freq` (Union[float, int]) : cutoff frequency of the filter
        - `high_cutoff_freq` (Union[float, int]) : cutoff frequency of the filter
        - `filter_order` (int): order of the filter

    Returns:
    -----------
        - `output_signal` (np.ndarray) : Filtered signal
        - `freq` (np.ndarray) : Frequencies for which the frequency response h is computed
        - `h` (np.ndarray) : Frequency response for each frequency
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


def notch_filter(sample_rate: Union[float, int], cutoff_freq: Union[float, int],
                 stop_band_width: Union[float, int], input_signal: Optional[np.ndarray] = None) -> Tuple[Union[np.ndarray, None], np.ndarray, np.ndarray]:
    """
    Creates a notch filter and filters the input signal.

    The stop-band width (in Hz) is centered on the cutoff frequency .
    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequenceis
    If no signal is inputted into the function, it still generates the filter and its frequency response from other arguments.

    Creates a notch IIR filter and applies it to filter the input signal.

    The filter is first created (using `scipy.iirnotch`) and applied as a zero phase filter (using `scipy.filtfilt`). 
    The use of a zero-phase filter leads to a cutoff frequency shift.
    To compensate the shift, the desired frequency can be corrected (using `my_filters.filtfilt_cutoff_frequency_corrector`) 
    before passing it to the `cutoff_freq` variable.

    The `input_signal` argument can be a 2D array of signals, one per column. 
        - If specified, the created filter is applied each signal of the array it.
        - If not specified, the function only creates the desired filter.

    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequencies.

    Parameters:
    -----------
        - `input_signal` (np.ndarray, optional) : input signal on which to apply the filter. Can be a 2D array where each column (axis=0) is a signal.
        - `sample_rate` (Union[float, int]) : signal sampling rate
        - `cutoff_freq` (Union[float, int]) : cutoff frequency of the filter
        - `filter_order` (int): order of the filter
        - `stop_band_width` (Union[float, int]) : width of the stop band (in Hz), centered on the `cutoff_freq`
    Returns:
    -----------
        - `output_signal` (np.ndarray) : Filtered signal
        - `freq` (np.ndarray) : Frequencies for which the frequency response h is computed
        - `h` (np.ndarray) : Frequency response for each frequency
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
    freq, h = freqz(b_notch, a_notch, fs=sample_rate)  # type: ignore

    return output_signal, freq, h
