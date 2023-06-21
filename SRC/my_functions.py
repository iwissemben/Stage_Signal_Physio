
import matplotlib.pyplot as plt
import numpy as np
# library for creating filters
from scipy.signal import welch, periodogram, get_window, hamming, boxcar
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

# =============================================================================
############################# Stream_Finder_by_name  ##########################
# =============================================================================


def stream_Finder_by_name(datalist, name):
    """
    This function browse the dictionary  

    Finds the index of the datalist element loaded from the ".xdf file" and returns it
    input arguments the list containing the data, and the name of the stream to search for

    inputs: object (plt, or axis[a,b]), numpy.ndarray(2D).
    outputs: [None]
    """
    print("StreamFinder executed")
    streamindex = None
    for i in datalist:
        if any(str(name).upper() in string.upper() for string in i["info"]["name"]):
            streamindex = datalist.index(i)
            print(streamindex)
            # print(i["info"]["type"])
            return streamindex
# =============================================================================
############################# show_markers  #####################################
# =============================================================================


def show_markers(plot_type, markers_times_array: np.ndarray):
    """
    Custom function to display event markers as vertical lines on a graph (plt or axis). 

    Inherits of the plot_type object to add marker to figure.

    Arguments:
    Markers_times_labels as a 2d array of markers with corresponding timestamps as [Marker,Timestamp]
    Required : plot_type, markers_times_array.

    inputs: object (plt, or axis[a,b]), numpy.ndarray(2D).
    outputs: [None]
    """

# iterate over an array of markers
    for i in markers_times_array:
        if i[1] == 111:
            # print(i)
            # plot a line x=time_stamp associated to the i marker of type 111 (begining of task)
            # print("plot_type is : ", type(plot_type), plot_type)
            marker111 = plot_type.axvline(x=i[0], color="b", label="111")
        else:
            # print(i)
            # plot a line x=time_stamp associated to the i marker of type 110 (begining of task)
            marker110 = plot_type.axvline(x=i[0], color="r", label="100")
    return marker111, marker110

# =============================================================================
############################# Single_plotter  #################################
# =============================================================================


def single_plot(filename: str, fig_number: int, x: np.ndarray, y: np.ndarray, fig_title: str,
                xlabel: str, ylabel: str, markers_times_array: np.ndarray = None, point_style: str = "-k",
                line_width: int | float = 1):
    """
    Custom multipurpose function that displays a single graph

    Single_plot uses the x and y datas as inputs for the plot.
    Optionally calls show_markers function.

    Arguments:
        Required: filename, figure number,figure, title, labels are other arguments used to generate the figure.
        Optional : Markers_times_labels as a 2d array of markers with corresponding timestamps as [Marker,Timestamp]

    inputs: [str,int,numpy.ndarray(1D),numpy.ndarray(1D),str,str,str,numpy.ndarray(2D),str,float]
    outputs: [None]
    """
    # Creation of the figure
    plt.figure(fig_number)
    plt.plot(x, y, point_style, lw=line_width)
    plt.title(str(fig_title+"\n"+filename))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))

    # Displays markers (optional)
    if markers_times_array is not None:
        show_markers(plt, markers_times_array)
    plt.legend()

# =============================================================================
############################# Mosaic_plotter  #################################
# =============================================================================


def mosaic_plot(figure, axis, filename: str, x: np.ndarray, y: np.ndarray, fig_title: str,
                xlabel: str, ylabel: str, channels: dict, markers_labels_times: np.ndarray = None):
    """
    Custom function that display a mosaic of graphs for each channel

    Mosaic_plot uses the figure and axis objects formerly instanciated to plot x and y data on each cell of the figure.
    The cells are defined by coordinates a and b (2,4 for now).Channel name is provided by the dictionary of channels. 
    Figure and plot titles are directly given by arguments.

    Plotting and labels goes by iteration on each cell before showing the figure with title.

    Arguments:
        Required: filename, x, y, figure, axis, fig_title, xlabel, ylabel, channels
        Optional : Markers_times_labels as a 2d array of markers with corresponding timestamps as [Marker,Timestamp]

    inputs: [str,numpy.ndarray(1D),numpy.ndarray(2D),numpy.ndarray(2D),matplotlib.figure.Figure,numpy.ndarray,str,str,dict(str:str)] 
    outputs: [None]
    """
    count = 0
    # Get the figure's geometry
    rows, cols = axis.shape
    print("FIGURE ROWS:", rows, "COLS:", cols)

    # Iterate on each coordinate of the plot to assign figure content and titles
    for a in range(rows):  # two rows of graphs
        for b in range(cols):  # each row conists of 4 graphs
            axis[a, b].plot(x, y[:, count])
            # show_markers("axis["+str(a)+","+str(b)+"]", Markers_times_labels)
            # Displays markers (optional)
            if markers_labels_times is not None:
                show_markers(axis[a, b], markers_labels_times)

            axis[a, b].set_title("Electrode "+str(count+1) +
                                 ":" + channels["Channel_"+str(count+1)])
            axis[a, b].set_xlabel(xlabel)
            axis[a, b].set_ylabel(ylabel)
            count = count+1
    plt.suptitle(fig_title+"\n"+filename)
    plt.legend()

# =============================================================================
############################# Compute_FFT_on_channels  ########################
# =============================================================================


def fft_create_positive_frequency_vector(signal: np.ndarray, Sampling_rate: int | float):
    """
    produces a positive frequency vector for fft

    inputs: numpy.ndarray(2D) and float as EEG_channels_signal and Sampling_rate
    outputs: Dictionary [key1:array1,key2:array2] as [frequencies:values,amplitudes:values]
    """
    # Return the Discrete Fourier Transform sample positive frequencies.
    fft_frequencies = np.fft.fftfreq(len(signal), d=1/Sampling_rate)
    print("fft_frequencies last value", fft_frequencies[-1])
    print("fft_frequencies len ", len(fft_frequencies))

    # last coordinate not comprised
    print("coordin: ", (len(fft_frequencies)//2))
    fft_frequencies = fft_frequencies[0:(len(fft_frequencies)//2)]
    print("fft_frequencies half last value: ",  fft_frequencies[-1])
    print("fft_frequencies half len ", len(fft_frequencies))
    return fft_frequencies


def fft_compute_on_single_channel(signal: np.ndarray):
    """
    Function that performs the fft of a single channel (column)

    Only returns the returns the FFT result associated with positive frequencies.

    inputs: numpy.ndarray(2D) as signal of an electrode
    outputs: Dictionary [key1:array1,key2:array2] as [frequencies:values,amplitudes:values]
    """
    # Return the FFT signal of positive frequencies.
    fft_signal = abs(np.fft.fft(signal))
    print("fft_signal length: ", len(fft_signal))
    fft_signal = fft_signal[0:(len(fft_signal)//2)]
    # fft_signal = fft_signal[0:(len(fft_signal)//2)]
    # print("fft_signal half length: ", len(fft_signal))

    return fft_signal


def compute_fft_on_all_channels(EEG_channels_signal: np.ndarray, Sampling_rate: int | float):
    """
    Function that computes the FFT on each channel.

    Each column of the 2d signal array is the signal of an electrode.
    Iterates over each column to compute its FFT, on positive frequencies.
    Returns a 2 key dictionary (frequencies,results eeg) associated with 2 arrays.

    inputs: numpy.ndarray(2D) and float as EEG_channels_signal and Sampling_rate
    outputs: Dictionary [key1:array1(1D),key2:array2(2D)] as [frequencies:values,amplitudes:values per electrodes]
    """
    # Create positive vector of frequencies
    frequencies = fft_create_positive_frequency_vector(
        EEG_channels_signal, Sampling_rate)

    # compute fft iteratively on each channel, store them in array, each column an electrode
    FFT_Results_EEG_channels = []
    for column in EEG_channels_signal.T:
        fft_signal_electrodei = fft_compute_on_single_channel(column)
        FFT_Results_EEG_channels.append(fft_signal_electrodei)

    # Consistent shaping of data
    FFT_Results_EEG_channels = np.array(FFT_Results_EEG_channels)
    FFT_Results_EEG_channels = FFT_Results_EEG_channels.transpose()

    return {"fft_frequencies": frequencies, "FFT_Results_EEG_channels": FFT_Results_EEG_channels}
# =============================================================================
########################## Cutoff frequency corrector  ########################
# =============================================================================


def filtfilt_cutoff_frequency_corrector(order: int, cutoff_freq: float | int, sampling_freq: float | int, pass_type: str = "low_pass"):
    """
    Function that corrects cutoff frequencies to use in combination with filt.filt()

    As a zero-phase filter (linear filter) is applied to a signal 
    the cutoff freq are diminished. The correction depends also on the order. 
    The adjustment is made on the angular cutoff frequency, which depends on the filter direction (LP,HP).
    SPECIFY the type of the filter with either "low_pass" or "high_pass"

    inputs: numpy.ndarray(2D) and float as EEG_channels_signal and Sampling_rate
    outputs: Dictionary [key1:array1,key2:array2] as [frequencies:values,amplitudes:values]

    # Biomechanics and Motor Control of Human Movement, 4th-edition (page 69)
    # https://www.codeproject.com/Articles/1267916/Multi-pass-Filter-Cutoff-Correction
    """
    filter_passes = 2
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

# =============================================================================
###################### Marker Nearest time-index finder  ######################
# =============================================================================


def nearest_timestamps_array_finder(EEG_times_stamps: np.ndarray, markers: np.ndarray):
    """
    Function that finds the nearest timestamps in EEG signal to each marker

    Useful when the marker timestamps may not be found in the EEG data due to Srate and time launch.
    Computes the 

    inputs: numpy.ndarray(1D),numpy.ndarray(1D)
    outputs: numpy.ndarray(2D) as [nearest_time_stamps_indices EEG,nearest_time_stamps]
    """
    nearest_time_stamps = []
    nearest_time_stamps_indices = []
    print("MARKERS LEN:", len(markers))
    for i in range(len(markers)):
        original_time_stamp = markers[i, 0]
        # array of differences beween the eeg times and the original marker' timestamp
        difference_array = np.absolute(EEG_times_stamps-original_time_stamp)
        # find the index of the minimal difference in the markers times stamps (nearest timestamp)
        index = difference_array.argmin()
        # append it to the liste of nearest timestamps
        nearest_time_stamps.append(EEG_times_stamps[index])
        nearest_time_stamps_indices.append(index)

    nearest_time_stamps = np.array(nearest_time_stamps)
    nearest_time_stamps_indices = np.array(nearest_time_stamps_indices)
    nearest_indices_timestamps = np.column_stack(
        (nearest_time_stamps_indices, nearest_time_stamps))

    return nearest_indices_timestamps

# =============================================================================
############################ Compute lagged PSD  ##############################
# =============================================================================


def get_segment_coordinates(reference_index: int, segment_length: int):
    """
    Computes the coordinates of a segment for psd calculation

    inputs:int,int
    outputs:int,int,int
    """

    # Define the segment coordinates (start,end)
    # PSD on a range of time delta_time before the time stamp
    lower_end = int(reference_index)-segment_length
    print("lower_end:", lower_end)

    # PSD on a range of time delta_time after the time stamp
    higher_end = int(reference_index)+segment_length
    print("Higher_end:", higher_end)

    # index_range=np.arange(lower_range,int(timestamp_index)+1)
    # +1 to inclue the value at the time stamp
    reference_end = int(reference_index)

    print("segment coordinates before marker:", "(", lower_end, ";",
          reference_end, "), delta_index:", segment_length)

    print("segment coordinates after marker:", "(", reference_end, ";",
          higher_end, "), delta_index:", segment_length)
    return lower_end, higher_end, reference_end


def compute_welch_estimation_on_segment(signal: np.ndarray, direction: str, sample_rate: int,
                                        reference_end: int, lower_end: int, higher_end: int, delta_index: int):
    """
    Computes the psd estimation(welch method) on a specific segment of a time signal.

    inputs:numpy.ndarray(1D),str,numpy.ndarray(2D),float,str
    outputs:numpy.ndarray(1D),numpy.ndarray(1D) as columns
    """
    if direction == "before":
        freq, Pxx_density = welch(signal[lower_end:reference_end+1],
                                  fs=sample_rate, window="hann",
                                  nperseg=delta_index, noverlap=delta_index//2, axis=0, detrend=False)
    elif direction == "after":
        freq, Pxx_density = welch(signal[reference_end:higher_end+1],
                                  fs=sample_rate, window="hann",
                                  nperseg=delta_index, noverlap=delta_index//2, axis=0, detrend=False)
    else:
        print("Wrong direction provided, please specify either 'before' or 'after'")
    return freq, Pxx_density


def compute_lagged_psds_one_signal(signal: np.ndarray, Srate: float | int, markers: np.ndarray,
                                   time_lag: float | int = 1, direction: str = "before"):
    """
    Computes psd estimation (welch) on segments of a time signal around list of references.

    For each index references (markers) the function delimits a segment of chosen time length and direction (before after ref).
    Performs the welch method on the segment and returns two 1D arrays(column) on for frequencies other for psd resutls.
    The resulting columns are stacked in two respective 2D arrays to make a layer representing the PSDs and Freqs of an electrode.

    inputs:numpy.ndarray(1D),float,numpy.ndarray(2D),float,str
    outputs:numpy.ndarray(2D),numpy.ndarray(2D)
    """
    # time expressed in number of points, Srate number of points per sec
    delta_index = int(Srate*time_lag)
    electrode_lagged_PSDS = []
    elecrode_frequencies = []

    # iterate on the markers to compute on each the psd on a given direction
    for timestamp_index in markers[:, 0]:
        # get the coordinates of the segment on which the psd will be computed
        lower_end, higher_end, reference_end = get_segment_coordinates(
            reference_index=timestamp_index, segment_length=delta_index)

        # Compute the welch method on the segment in accordance with the deisred direction
        freq, Pxx_density = compute_welch_estimation_on_segment(
            signal, direction, Srate, reference_end, lower_end, higher_end, delta_index)
        # Store the result columns in lists (Nmarkers length)
        electrode_lagged_PSDS.append(Pxx_density)
        elecrode_frequencies.append(freq)

        # Create layers: Stack elements of the lists (1D array) to create two 2D arrays (x=PSD,y=markeri) (x=freqs,y=markeri)
        electrode_stacked_markers = np.column_stack(electrode_lagged_PSDS)
        electrode_stacked_frequencies = np.column_stack(elecrode_frequencies)
    return electrode_stacked_frequencies, electrode_stacked_markers


def compute_lagged_psd2_all_electrodes(EEG_data: np.ndarray, Srate: float | int, markers: np.ndarray,
                                       time_lag: float | int = 1, direction: str = "before"):
    """
    Computes psd estimation (welch) on segments of multiple time signals around list of references

    inputs:numpy.ndarray(2D),float,numpy.ndarray(2D),float,str
    outputs:numpy.ndarray(3D),numpy.ndarray(3D)
    """
    layers_psds = []
    layers_frequencies = []
    for electrode in EEG_data.T:  # iterate on electrodes
        # Produce a layer (2d array) of PSDs for an electrode
        electrode_stacked_frequencies, electrode_stacked_markers = compute_lagged_psds_one_signal(electrode, Srate, markers,
                                                                                                  time_lag=time_lag, direction=direction)
        # store the layers in a list
        layers_psds.append(electrode_stacked_markers)
        layers_frequencies.append(electrode_stacked_frequencies)

    # Stack each layer to get a 3d array (x=psdx or freqx ,y=markery,z=electrodez)
    tridi_PSDs = np.stack(layers_psds, axis=2)
    tridi_frequencies = np.stack(layers_frequencies, axis=2)

    return tridi_frequencies, tridi_PSDs

# =============================================================================
##################### Plot temporal signal and its DSPs  ######################
# =============================================================================


def compute_signal_time_dsps(signal: np.ndarray, sample_rate: int):
    """
    Computes the PSD of a signal using 3 different methods (via FFT, via Scipy's periodogram and welch functions).

    Parameters:
        signal (np.ndarray): 1D array of amplitudes
        sample_rate (int): sampling rate of the signal

    Return:
        time_signal (dict): Dictionary containing signal's timepoints and amplitudes as ndarray under key1 "time_vector" and key2 "amplitudes".
        PSD_fft (dict) : Dictionary containing signal's DSP results computed via FFT: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
        PSD_p (dict) : Dictionary containing signal's DSP results computed via periodogram: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
        PSD_w (dict) : Dictionary containing signal's DSP results computed via welch: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
    """
    N = len(signal)
    print("N: ", N)
    duration = N/sample_rate
    print("duration: ", duration)
    time_vector = np.arange(0, duration, 1/sample_rate)
    print("time_vector shape: ", time_vector.shape)

    # compute FFT of the signal
    signal_fft = np.fft.fft(signal)
    signal_frequency_vector = np.fft.fftfreq(len(signal), 1/sample_rate)

    # Only keep the positive frequencies and associated amplitudes
    signal_frequency_vector = signal_frequency_vector[0:(
        (len(signal_frequency_vector)//2)+1)]  # +1 due to python intervals

    signal_fft = signal_fft[0:((len(signal_fft)//2)+1)]

    # compute PSD via FFT
    psd_from_fft = (np.abs(signal_fft)**2)/(N*sample_rate)

    # compute PSD via periodogram
    freq1, Pxx_density1 = periodogram(
        signal,  fs=sample_rate, window=boxcar(N), detrend=False)
    # print(type(psd_from_periodogram))
    freq2, Pxx_density2 = welch(signal, fs=sample_rate, window=hamming(1000),
                                nperseg=1000, noverlap=500, nfft=N, detrend=False,
                                axis=0)

    # create dictionaries of frequencies with psd results for each method to return
    time_signal = {"time_vector": time_vector, "amplitudes": signal}
    PSD_fft = {"frequencies": signal_frequency_vector, "psds": psd_from_fft}
    PSD_p = {"frequencies": freq1, "psds": Pxx_density1}
    PSD_w = {"frequencies": freq2, "psds": Pxx_density2}
    return time_signal, PSD_fft, PSD_p, PSD_w


def plot_signal_time_dsps(fig_number: int, signal: np.ndarray, sample_rate: int, signal_name: str, external_results: np.array = None):
    """
    Plots the time signal alongside its 3 PSDs.

    Calls compute_signal_time_dsps() and plots the results as figure of 4 subplots (lines).
    If external_results provided the function will superimpose the external results to each corresponding PSD subplot and add an inset zoom to check differences.

    Parameters:
        fig_number (int): Number of the figure.
        signal (np.ndarray): 1D array of amplitudes.
        sample_rate (int): sampling rate (in Hz).
        signal_name (str): Name of the signal for figure title.
        external_results (ndarray): array of PSD results. Each results is formed by 2 columns (frequencies, respective psd result). Must be 3*2 columns (3 PSD methods)
    External results must be an array array of 6 columns arranged by two as 3*(frequenceies,psds) for each PSD calculation method.

    Return:
        PSD_fft (dict) : Dictionary containing signal's DSP results computed via FFT: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
        PSD_p (dict) : Dictionary containing signal's DSP results computed via periodogram: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
        PSD_w (dict) : Dictionary containing signal's DSP results computed via welch: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
    """
    # compute the PSDs of a signal using 3 different methods
    time_signal, PSD_fft, PSD_p, PSD_w = compute_signal_time_dsps(
        signal=signal, sample_rate=sample_rate)

    # Show the time signal and the 3 different results of the PSD
    figure, axis = plt.subplots(4, figsize=(
        10, 7), layout="constrained", num=fig_number)
    figure.suptitle(signal_name + " :\n Time-signal and DSPs")

    # plot time signal
    axis[0].plot(time_signal["time_vector"], time_signal["amplitudes"], "-k")
    # axis[0].set_title('Time signal')
    axis[0].set_ylabel("Amplitude(µV)")
    axis[0].set_xlabel("time(s)")
    axis[0].set_xlim(0)
    axis[0].grid()

    # plot signal's DSP via FFT
    axis[1].plot(PSD_fft["frequencies"], PSD_fft["psds"], label="Python")
    # axis[1].set_title('PSD from FFT')
    axis[1].set_xlim(0)
    axis[1].set_ylabel("PSD from \n FFT (µV²/Hz)")
    axis[1].set_xlabel("Frequency (Hz)")
    axis[1].grid()

    # plot signal's DSP via periodogramm
    axis[2].plot(PSD_p["frequencies"], PSD_p["psds"], label="_Python")
    # axis[2].set_title('PSD from periodogramm (µV²/Hz)')
    axis[2].set_xlim(0)
    axis[2].set_ylabel("PSD from \n periodogramm \n (µV²/Hz)")
    axis[2].set_xlabel("Frequency (Hz)")
    axis[2].grid()

    # plot signal's DSP via scipy.signal.welch
    axis[3].plot(PSD_w["frequencies"], PSD_w["psds"], label="_Python")
    # axis[3].set_title('DSP')
    axis[3].set_xlim(0)
    axis[3].set_ylabel("PSD signal.welch \n (µV²/Hz)")
    axis[3].set_xlabel("Frequency (Hz)")
    axis[3].grid()

    # Superimpose to each PSD subplot other results (ex from matlab)
    if external_results is not None:
        PSD_fft_external = {
            "frequencies": external_results[:, 0], "psds": external_results[:, 1]}
        PSD_p_external = {
            "frequencies": external_results[:, 2], "psds": external_results[:, 3]}
        PSD_w_external = {
            "frequencies": external_results[:, 4], "psds": external_results[:, 5]}

        axis[1].plot(PSD_fft_external["frequencies"],
                     PSD_fft_external["psds"], '--r', label="Matlab")
        axis[2].plot(PSD_p_external["frequencies"],
                     PSD_p_external["psds"], '--r', label="_Matlab")
        axis[3].plot(PSD_w_external["frequencies"],
                     PSD_w_external["psds"], '--r', label="_Matlab")

        add_inset_zoom(ax=axis[1], x_data=(PSD_fft["frequencies"], PSD_fft_external["frequencies"]), y_data=(PSD_fft["psds"], PSD_fft_external["psds"]),
                       zoom_region=(0, 20, 0, np.max(np.maximum(PSD_fft_external["psds"], PSD_fft["psds"]))))
        add_inset_zoom(ax=axis[2], x_data=(PSD_p["frequencies"], PSD_p_external["frequencies"]), y_data=(PSD_p["psds"], PSD_p_external["psds"]),
                       zoom_region=(0, 20, 0, 10))
        add_inset_zoom(ax=axis[3], x_data=(PSD_p["frequencies"], PSD_w_external["frequencies"]), y_data=(PSD_w["psds"], PSD_w_external["psds"]),
                       zoom_region=(0, 20, 0, 6))
    else:
        add_inset_zoom(ax=axis[1], x_data=PSD_fft["frequencies"], y_data=PSD_fft["psds"],
                       zoom_region=(0, 50, 0, np.max(PSD_fft["psds"])))

    figure.legend(title="Results source", loc="upper right")
    return PSD_fft, PSD_p, PSD_w


def add_inset_zoom(ax: plt.Axes, x_data, y_data: np.ndarray, zoom_region: tuple):
    """
    Add a child inset axes plot to the given Axes object. Can show multiple overlapping series.
    The child inset axes object inherits the line style of the parent plot.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to add the inset zoom to.
        x_data (tuple of 1D array or array-like): x-coordinates of the data points.
        y_data (tuple of 1D array or array-like): y-coordinates of the data points.
            If a single array is provided, it represents a single series.
            If a tuple of arrays is provided, array represents a separate series.
        zoom_region (tuple): The region to be shown in the zoomed inset in the format (x1, x2, y1, y2).

    Returns: axins object
    """
    # axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc='upper right')
    axins = inset_axes(ax, width=2, height=0.7, loc='upper right')

    # Get the lines settings from the main plot to inherit them
    lines = ax.get_lines()
    if isinstance(y_data, tuple):
        # if multiple series provided
        for xserie, yserie, line in zip(x_data, y_data, lines):
            axins.plot(xserie, yserie, color=line.get_color(), linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth())

    else:
        line = lines[0]
        # if single series provided
        axins.plot(x_data, y_data, color=line.get_color(), linestyle=line.get_linestyle(),
                   linewidth=line.get_linewidth())

    x1, x2, y1, y2 = zoom_region
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none",
               ec="0.5", lw=1.5, linestyle="--")
    axins.tick_params(axis='both', which='both',
                      labelleft=True, labelbottom=True)
    axins.grid(visible=True, linestyle='--', linewidth=0.5)


# =============================================================================
############################ Generate test signal  ############################
# =============================================================================


def generate_sine_wave(amplitude, frequency, duration, change_time, new_amplitude, sample_rate):
    """
    Generate a sine wave that change amplitude after a specific amount of time.

    Parameters:
        amplitude (float): Signal's original amplitude (A.U).
        frequency (float): Signal's frequency (in Hz).
        duration (float): Duration of the signal (in seconds).
        change_time (float): Time  of amplitude change (in seconds).
        new_amplitude (float): Signal's new amplitude (A.U).
        sample_rate (float): Sampling rate of the signal (in Hz). 

    Returns:
        t (ndarray): Array of timepoints (in seconds).
        signal (ndarray): Array of signal's amplitudes (A.U)
    """
    t = np.linspace(0, duration, num=int(duration*sample_rate))
    signal = amplitude * np.cos(2 * np.pi * frequency * t)
    signal2 = new_amplitude * np.sin(2 * np.pi * frequency * t)

    # Change amplitude after a certain time
    change_index = int(change_time * sample_rate)
    signal[change_index:] = signal2[change_index:]

    return t, signal
# =============================================================================
######################## Compute average ERSP on blocks  ######################
# =============================================================================


def compute_average_ratio_for_event_on_blocks_for_all_electrodes(mat3d):
    """
    # compute the average densities among trials of a block and returns 3d arrays of 1 column (average), PSD values, 8
    # returns array1{block1_event_typex_averagedPSDs,dim=[averaged_ratios_block1_eventx,electrodei]},array2{block2_event_typex_averagedPSDs,dim=[averaged_ratios_block2_eventx,electrodei]]}

    inputs: numpy.ndarray(3D)
    outputs: numpy.ndarray(2D),numpy.ndarray(2D) as mean_per_row_per_plane_test_block1, mean_per_row_per_plane_test_block2
    """
    mean_per_row_per_plane_blocks = []
    mean_per_row_per_plane_test_block1 = []
    mean_per_row_per_plane_test_block2 = []
    # Boucle sur les plans (matrices 2D) de la matrice 3D
    for i in range(mat3d.shape[0]):
        current_plane = mat3d[i]
        mean_per_row_per_plane_test_block1.append(
            np.mean(current_plane[:, :3], axis=1))
        mean_per_row_per_plane_test_block2.append(
            np.mean(current_plane[:, 3:], axis=1))
    mean_per_row_per_plane_test_block1 = np.column_stack(
        mean_per_row_per_plane_test_block1)
    mean_per_row_per_plane_test_block2 = np.column_stack(
        mean_per_row_per_plane_test_block2)

    return mean_per_row_per_plane_test_block1, mean_per_row_per_plane_test_block2
