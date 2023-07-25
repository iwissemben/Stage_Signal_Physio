
import matplotlib.pyplot as plt
import numpy as np
# xdf file importation
import pyxdf 
# library for creating filters
from scipy.signal import welch, periodogram, get_window, hamming, boxcar
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

from my_filters import *


# =============================================================================
############################### Manage Xdf files  #############################
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
        
def create_marker_times_labels_array(marker_time_stamps:np.ndarray=None,marker_labels:np.ndarray=None,xdf_input_filepath:str=None):
    """
    Create an array combining the markers labels and their timestamps.
        If xdf file specified, timestamps are retrieved from the file and processed to start relative to recording start, not to unix epoch anymore.
        If marker_time_stamps and marker_labels, the arrays are stacked and returned as defined.


    Parameters
    ----------
        marker_time_stamps(np.ndarray): 1D array containing the marker timestamps.
        marker_labels(np.ndarray): 1D array containing the markers labels.
        xdf_input_filepath(str): Filepath of the EEG recordings as xdf file.
        
    Returns
    -------
        markers_times_labels(np.ndarray): 2D array containing the markers's timestamps alongside their labels.
    """
    all_args =[xdf_input_filepath,marker_time_stamps,marker_labels]
    is_all_none = all(element is None for element in all_args)
    
    if is_all_none:
        print("No arguments specified.")
        markers_times_labels=None
    elif xdf_input_filepath:
        #Retrieve directly from xdf file markers timestamps relative to recording start and their labels
        xdf_data, header = pyxdf.load_xdf(xdf_input_filepath, select_streams=[{'type': 'EEG'}, {
                                    'type': 'Markers', 'name': 'MouseToNIC'}])
        EEG_stream=xdf_data[0]
        Mouse_markers_stream=xdf_data[1]
        Mouse_markers_labels=Mouse_markers_stream["time_series"]
        Mouse_markers_times=Mouse_markers_stream["time_stamps"]-EEG_stream["time_stamps"][0]
        markers_times_labels=np.column_stack((Mouse_markers_times,Mouse_markers_labels))
    else:
        #stack given arrays to create the marker_times_labels array
        markers_times_labels=np.column_stack((marker_time_stamps,marker_labels))
    return markers_times_labels
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


def single_plot(filename: str, x: np.ndarray, y: np.ndarray, fig_title: str,
                xlabel: str, ylabel: str, markers_times_array: np.ndarray = None, point_style: str = "-k",
                line_width: int | float = 1,fig_number: int=None):
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
    if fig_number:
        plt.figure(fig_number)
    plt.plot(x, y, point_style, lw=line_width)
    plt.title(str(fig_title+"\n"+filename))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))

    # Displays markers (optional)
    if markers_times_array is not None:
        show_markers(plt, markers_times_array)
    plt.legend()
    plt.show()

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


def fft_compute_on_single_channel2(signal, Fs):
    N = np.shape(signal)[0]
    f_res = Fs/N  # freq res

    freq_vect = np.linspace(0, (N-1)*f_res, N)
    amplitude_fft = np.fft.fft(signal)
    amplitude_fft_magnitude = np.abs(amplitude_fft)/N

    freq_vect_for_plot = freq_vect[0:int(N/2+1)]  # +1 cf slicing
    amplitude_fft_magnitude_for_plot = 2*amplitude_fft_magnitude[0:int(N/2+1)]
    return freq_vect_for_plot, amplitude_fft_magnitude_for_plot


def compute_fft_on_all_channels2(channels_signals: np.ndarray, Fs: int | float):
    channels_fft_frequencies = []
    channels_fft_magnitudes = []

    for signal in channels_signals.T:
        frequencies, magnitudes = fft_compute_on_single_channel2(signal, Fs)
        channels_fft_frequencies.append(frequencies)
        channels_fft_magnitudes.append(magnitudes)

    channels_fft_frequencies = np.transpose(np.array(channels_fft_frequencies))
    channels_fft_magnitudes = np.transpose(np.array(channels_fft_magnitudes))

    return {"FFT_frequencies": channels_fft_frequencies,
            "FFT_magnitudes": channels_fft_magnitudes}
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


def nearest_timestamps_array_finder2(EEG_times_stamps: np.ndarray, markers: np.ndarray):
    """
    Finds the nearest timestamps to each marker in EEG signal timestamps array 

        Application: 
            Useful when the markers timestamps may not be found in the EEG data due to Srate and time launch.
                ie: if the marker timestamps are foudn between two signal samples.

    Parameters:
    -----------
        EEG_times_stamps(np.ndarray): 1D array of the signal timestamps
        markers (np.ndarray): 2D array (markers_original_timestamps,marker_labels)
    Returns:
    -------
        nearest_indices_timestamps(np.ndarray): 2D array  as [nearest_time_stamps_indices EEG,nearest_time_stamps]
    inputs: numpy.ndarray(1D),numpy.ndarray(1D)

    """
    nearest_time_stamps = []
    nearest_time_stamps_indices = []
    print("MARKERS LEN:", len(markers))
    for i in range(len(markers)):
        original_time_stamp = markers[i, 0]
        # array of differences beween the eeg times and the original marker' timestamp
        difference_array = np.absolute(
            EEG_times_stamps-original_time_stamp)
        # find the index of the minimal difference in the markers times stamps (nearest timestamp)
        index = difference_array.argmin()
        # append it to the liste of nearest timestamps
        nearest_time_stamps.append(EEG_times_stamps[index])
        nearest_time_stamps_indices.append(index)

    nearest_time_stamps = np.array(nearest_time_stamps)
    nearest_time_stamps_indices = np.array(
        nearest_time_stamps_indices)
    nearest_indices_timestamps = np.column_stack(
        (nearest_time_stamps_indices, nearest_time_stamps))

    return nearest_indices_timestamps

def nearest_timestamps_array_finder(EEG_times_stamps: np.ndarray, markers: np.ndarray):
    """
    Finds the nearest timestamps to each marker in EEG signal timestamps array.

    For each marker timestamp, the function looks for the nearest timestamp in EEG signal timestamps array
    and its corresponding index.
    
    Application: 
        Useful when the markers timestamps may not be found in the EEG data due to Srate and time launch.
            ie: if the marker timestamps are foudn between two signal samples.

    Parameters:
    -----------
        EEG_times_stamps (np.ndarray): 1D array of the signal timestamps
        markers (np.ndarray): 2D array (markers_original_timestamps,marker_labels)

    Returns:
    -------
        nearest_indices_timestamps (np.ndarray): 3D array as (markers_new_timestamps_indices, markers_new_timestamps, marker_labels)

    """
    nearest_time_stamps = []
    nearest_time_stamps_indices = []
    print("MARKERS LEN:", len(markers))

    #iterate over the  marker timestamps
    for y in markers[:, 0]:
        original_time_stamp = y
        # array of differences beween the eeg times and the original marker' timestamp
        difference_array = np.absolute(
            EEG_times_stamps-original_time_stamp)
        # find the index of the minimal difference in the markers times stamps (nearest timestamp)
        index = difference_array.argmin()
        # append it to the liste of nearest timestamps
        nearest_time_stamps.append(EEG_times_stamps[index])
        nearest_time_stamps_indices.append(index)

    #convert the list to array
    nearest_time_stamps = np.array(nearest_time_stamps)
    nearest_time_stamps_indices = np.array(
        nearest_time_stamps_indices)
    marker_labels=markers[:,1]

    #stack arrays (nearest_index,nearest_timestamp,label)
    """
    nearest_indices_timestamps = np.column_stack(
        (nearest_time_stamps_indices, nearest_time_stamps,marker_labels))
    """
    #store data in dictionary
    nearest_indices_timestamps={
        "markers_timestamp_indices":nearest_time_stamps_indices,
        "markers_timestamps":nearest_time_stamps,
        "marker_labels":marker_labels
    }

    return nearest_indices_timestamps

# =============================================================================
############################ Compute lagged PSD  ##############################
# =============================================================================


def get_segment_coordinates(reference_index: int, segment_length: int, debug: bool = False):
    """
    Computes the coordinates of a segment for psd calculation

    inputs:int,int
    outputs:int,int,int
    """

    # Define the segment coordinates (start,end)
    # PSD on a range of time delta_time before the time stamp
    lower_end = int(reference_index)-segment_length
    # print("lower_end:", lower_end)

    # PSD on a range of time delta_time after the time stamp
    higher_end = int(reference_index)+segment_length
    # print("Higher_end:", higher_end)

    # index_range=np.arange(lower_range,int(timestamp_index)+1)
    # +1 to inclue the value at the time stamp
    reference_end = int(reference_index)

    if debug is True:

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
def create_positive_frequency_vector(Fs:int, N:int):
    """
        Create positive frequency vector.
        Parameters:
        ----------
        Fs (int): Sampling frequency of the signal (in Hz)
        N (int): Length of the signal

        Return:
        -------
        frequencies(np.ndarray): 1D array of frequencies ranging from 0 to the Nyquist frequency by Fs/2 step.
    """
    # Calculate the frequency resolution
    freq_resolution = Fs / N

    # Create the frequency vector from 0 Hz to Fs/2
    frequencies = np.linspace(0, Fs/2, N//2 + 1)

    return frequencies

def compute_signal_time_dsps(signal: np.ndarray, sample_rate: int):
    """
    Computes the PSD of a signal using 3 different methods (via FFT, via Scipy's periodogram and welch functions).

    Parameters:
    ----------
        signal (np.ndarray): 1D array of amplitudes
        sample_rate (int): sampling rate of the signal

    Return:
    -------
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
    #signal_frequency_vector = np.fft.fftfreq(N, 1/sample_rate)
    signal_frequency_vector=create_positive_frequency_vector(Fs=sample_rate,N=N)
    freq_vector_len=len(signal_frequency_vector)

    #signal_frequency_vector = np.arange(0,(sample_rate//2)+freq_res,freq_res)
    print(f"signal_frequency_vector before crop len:{len(signal_frequency_vector)},half_val: {signal_frequency_vector[-(freq_vector_len//2)]}")

    # Only keep the positive frequencies and associated amplitudes
    """
    signal_frequency_vector = signal_frequency_vector[:(
        ((freq_vector_len//2 +1)))]  # +1 due to python intervals
    """
    print(f"signal_frequency_vector last freq : {signal_frequency_vector[-1]}")
    signal_fft = signal_fft[:((N//2)+1)]
    
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

def plot_single_signal_time_dsps(fig_number: int, signal: np.ndarray, sample_rate: int, fig_title: str, external_results: str = None):
    """
    Plots a single time signal alongside its 3 PSDs.

    Calls compute_signal_time_dsps() and plots the results as figure of 4 subplots (lines).
    If external_results specified, calls import_psd_results2() to superimpose the external results to each corresponding PSD subplot and add an inset zoom to check differences.
    
    Returns 2 dictionaries (psd_python_results, psd_matlab_results), each containing the PSD results (from fft,periodogram,welch) of the desired signal.
        For both, results are stored under respective keys (key1,key2,key3)=("PSD_FFT","PSD_P","PSD_W")

    Parameters:
    ----------
        fig_number (int): Number of the figure.
        signal (np.ndarray): 1D array of amplitudes.
        sample_rate (int): sampling rate (in Hz).
        fig_title (str): Title of the figure.
        external_results (str): external psd results filename with extension.csv (cf function import_psd_results2()).
            External results must be an array of 6 columns arranged by two, as 3*(frequenceies,psds), for each PSD calculation method.

    Return:
    ----------
        psd_results (dict) : Dictionary of two key:value pairs containing a signal's DSP results (FFT,periodogram,welch).
            key1:"Python_PSD_results"; value1: Dictionary of results from 'compute_signal_time_dsps()' function
            key2:"Matlab_PSD_results"; value1: Dictionary of results from 'import_psd_results2() function
    """
    # compute the PSDs of a signal using 3 different methods
    time_signal, PSD_fft, PSD_p, PSD_w = compute_signal_time_dsps(
        signal=signal, sample_rate=sample_rate)
    psd_python_results={'PSD_FFT':PSD_fft,'PSD_P':PSD_p,'PSD_W':PSD_w}


    # Show the time signal and the 3 different results of the PSD
    figure, axis = plt.subplots(4, figsize=(
        10, 7), layout="constrained", num=fig_number)
    figure.suptitle(fig_title + " :\n Time-signal and DSPs")

    # plot time signal
    axis[0].plot(time_signal["time_vector"], time_signal["amplitudes"], "-k")
    # axis[0].set_title('Time signal')
    axis[0].set_ylabel("Amplitude(µV)")
    axis[0].set_xlabel("time(s)")
    #axis[0].set_xlim(0)
    axis[0].grid()

    # plot signal's DSP via FFT
    axis[1].plot(PSD_fft["frequencies"], PSD_fft["psds"], label="Python")
    # axis[1].set_title('PSD from FFT')
    #axis[1].set_xlim(0)
    axis[1].set_ylabel("PSD from \n FFT (µV²/Hz)")
    axis[1].set_xlabel("Frequency (Hz)")
    axis[1].grid()

    # plot signal's DSP via periodogramm
    axis[2].plot(PSD_p["frequencies"], PSD_p["psds"], label="_Python")
    # axis[2].set_title('PSD from periodogramm (µV²/Hz)')
    #axis[2].set_xlim(0)
    axis[2].set_ylabel("PSD from \n periodogramm \n (µV²/Hz)")
    axis[2].set_xlabel("Frequency (Hz)")
    axis[2].grid()

    # plot signal's DSP via scipy.signal.welch
    axis[3].plot(PSD_w["frequencies"], PSD_w["psds"], label="_Python")
    # axis[3].set_title('DSP')
    #axis[3].set_xlim(0)
    axis[3].set_ylabel("PSD signal.welch \n (µV²/Hz)")
    axis[3].set_xlabel("Frequency (Hz)")
    axis[3].grid()

    # Superimpose to each PSD subplot other results (ex from matlab)
    if external_results is not None:        
        PSD_fft_external,PSD_p_external,PSD_w_external=import_psd_results2(psd_results_file_name=external_results)
        psd_matlab_results={'PSD_FFT':PSD_fft_external,'PSD_P':PSD_p_external,'PSD_W':PSD_w_external}
        print(f"len psd matlab fft:{len(psd_matlab_results['PSD_FFT']['frequencies'])}")


        axis[1].plot(PSD_fft_external["frequencies"],
                     PSD_fft_external["psds"], '--r', label="Matlab")
        axis[2].plot(PSD_p_external["frequencies"],
                     PSD_p_external["psds"], '--r', label="_Matlab")
        axis[3].plot(PSD_w_external["frequencies"],
                     PSD_w_external["psds"], '--r', label="_Matlab")

        add_inset_zoom(ax=axis[1], x_data=(PSD_fft["frequencies"], PSD_fft_external["frequencies"]), y_data=(PSD_fft["psds"], PSD_fft_external["psds"]),
                       zoom_region=(0, 30, 0, np.max(np.maximum(PSD_fft_external["psds"], PSD_fft["psds"]))))
        add_inset_zoom(ax=axis[2], x_data=(PSD_p["frequencies"], PSD_p_external["frequencies"]), y_data=(PSD_p["psds"], PSD_p_external["psds"]),
                       zoom_region=(0, 30, 0, 10))
        add_inset_zoom(ax=axis[3], x_data=(PSD_p["frequencies"], PSD_w_external["frequencies"]), y_data=(PSD_w["psds"], PSD_w_external["psds"]),
                       zoom_region=(0, 30, 0, 6))
        psd_results={'Python_PSD_results':psd_python_results,'Matlab_PSD_results':psd_matlab_results}

    else:
        add_inset_zoom(ax=axis[1], x_data=PSD_fft["frequencies"], y_data=PSD_fft["psds"],
                       zoom_region=(0, 40, 0, np.max(PSD_fft["psds"])))
        psd_results={'Python_PSD_results':psd_python_results}

    figure.legend(title="Results source", loc="upper right")    
    return psd_results

def plot_multi_signal_time_dsps(multi_channel_signals: np.ndarray, sample_rate:float,
                                channels_dict:dict, selected_channel_numbers:np.ndarray,input_signal_filename:str):    
    """
    Plots multiple time signals alongside their respective 3 PSDs.
    Based on the plot_single_signal_time_dsps() which is repeated for each selected channel.
        If multiple channels (n) are selected, plot_single_signal_time_dsps() is called n times.
    Calls list_matlab_psd_results_filenames() to list external psd results filename of selected channels.
    
    Returns a dictionary containing for each electrodes the results of plot_single_signal_time_dsps() function

    Parameters:
    ----------
        multi_channel_signals(np.ndarray): 2D array of amplitudes
        sample_rate (int): sampling rate (in Hz)
        channels_dict(dict): dictionary of channels names
        selected_channel_numbers(np.ndarray): 1D array of channel numbers to study
        input_signal_filename(str): name of the input file containing time-signals (used also to retrieve matlab results)

    Return:
    ----------
        electordes_results_dict (dict) : Dictionary of n key:value pairs containing n signal's (python and matlab) DSP results (FFT,periodogram,welch).
            keyi:"channel_name_i" ; valuei:psd_results
            where psd_results(dict) is output of plot_single_signal_time_dsps().
    """
    selected_channel_indexes=selected_channel_numbers-1

    #For each selected electrode, determine the filenames of the appropriate external (psd) results
    MATLAB_PSD_results_filenames_list=list_matlab_psd_results_filenames(input_signal_filename=input_signal_filename,
                                                                        channels_dict=channels_dict,
                                                                        selected_channel_numbers=selected_channel_numbers)
    electordes_results_dict={}
    for iter,(channel_num,channel_index,matlab_results_filename) in enumerate(zip(selected_channel_numbers,selected_channel_indexes,MATLAB_PSD_results_filenames_list)):
        channel_name=f"Channel_{str(channel_num)}_{channels_dict['Channel_'+str(channel_num)]}"
        print("channel name:",channel_name)
        figure_title=f"{input_signal_filename}\n {matlab_results_filename} \n {channel_name}"
        print(figure_title)
        channel_i_signal=multi_channel_signals[:,channel_index] #select correct signal
        electrode_i_psd_results=plot_single_signal_time_dsps(fig_number=iter, signal=channel_i_signal,
                                                             sample_rate=sample_rate, fig_title=figure_title,
                                                             external_results=matlab_results_filename)
        key=f"{channel_name}"
        value=electrode_i_psd_results
        electordes_results_dict[key] = value  # Ajoute la paire clé-valeur au dictionnaire
    return electordes_results_dict

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
    """    t = np.linspace(0, duration, num=int(duration*sample_rate))
    signal = amplitude * np.cos(2 * np.pi * frequency * t)
    """
    
    # t = np.arange(0, (sample_rate*duration), (1/sample_rate))
    t = np.linspace(0, duration, num=int(duration*sample_rate), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
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

# =============================================================================
############################ New compute lagged PSD  ##########################
# =============================================================================


def get_segment_coordinates2(reference_index:int, segment_time_length, sample_rate):
    """
    Gets the coordinates of segments of specified time length on each side of a reference index.

    Returns 3 coordinates defining 2 segments:
        (Lower end, reference_index) and (reference_index,higher_end)

    Application: Allows to retieve the coordinates of the 2 segments surrounding a marker :
        ie:1s before the marker and 1s after the marker.

    Parameters:
    -----
        reference_index (int): sample index used as reference to define a segment
        segment_time_length (int): expressed in seconds
        sample_rate (int): Sample rate

    Returns:
    ------
        segments_coordinates (dict): coordinates of the segments as indexes
            key1:"reference_end"
            key2:"lower_end"
            key3: "higher_end"
    """

    # Length of the segment expressed in number of points
    nperseg = segment_time_length*sample_rate  
    
    lower_end = reference_index - nperseg
    higher_end = reference_index + nperseg
    reference_end = reference_index
    # print("lower end: ", lower_end, ",reference: ",reference_end, "higher end: ", higher_end)

    segments_coordinates={"reference_end": reference_end, "lower_end": lower_end, "higher_end": higher_end}
    return segments_coordinates


def get_signal_segement2(signal:np.ndarray, lower_end: int, higher_end: int):
    """
    Get a segment of a given signal.
        The segment to extract is defined by its coordinates (lower_end,higher_end) expressed as the corresponding indexes of the signal array.
            Note: array slicing already takes into account python [lower_end,higher_end+1]

    Parameters:
    ------
        lower_end (int): index of the lower end of the segment
        higher_end (int): index of the higher end of the segment

    Retruns:
    ------
        signal_segment (np.ndarray): signal segment data
        signal_length (int): length of original segment
    """
    signal_length = len(signal)
    signal_segment = signal[lower_end:(higher_end+1)]  # cf python slicing
    segment_length=len(signal_segment)
    return signal_segment, signal_length

def extract_data_epochs(signal:np.ndarray,sample_rate:float,markers_labels_times:np.ndarray,select_events:tuple,epoch_limits:tuple[float,float]):
    """
    Extract epochs from a signal.

    Parameters
    ---------
        signal (np.array): 1D array of samples
        srate (float): sampling rate of the signal
        marker_labels_times (np.array): 2D array of marker information as ("markers_timestamp_indices","markers_timestamps", "marker_labels")
        select_events (tuple): tuple of selected event types ex (111,100)
        epoch_limits (tuple): 2 element tuple specifying the start and end of the epoch relative to the event (in seconds). (from 1 sec before to 2 sec after the time-locking event)

    Returns
    --------
        epoched_signals (dict): Dictionary of epoched signals for each marker type
            Keys (str): "label_markertype"
            values (np.ndarray): 2D array of signal data epochs arranged as a column per event
    """
    #convert the marker dictionary to a 2D array
    print(f"event structure type: {type(markers_labels_times)}")
    array_markers_labels_times=np.column_stack(markers_labels_times.values())
    
    #convert the epoch limits in number of points
    n_points_before=epoch_limits[0]*sample_rate
    n_points_after=epoch_limits[1]*sample_rate
    print(f"n_points_before_marker:{n_points_before} - n_points_after_marker:{n_points_after}")

    epoched_signals={}
    for event in select_events:
        signal_segments=[]
        for row in array_markers_labels_times:
            if row[2]==event:
                marker_index=row[0]
                #calculate the coordinate of the segments
                first_seg_coord=int(marker_index-n_points_before)
                second_seg_coord=int(marker_index+n_points_after)
                print(f"first_seg_coord: {first_seg_coord} - second_seg_coord: {second_seg_coord}")
                #extract the segments and list them
                signal_segment,_ = get_signal_segement2(signal=signal, lower_end=first_seg_coord, higher_end=second_seg_coord)
                signal_segments.append(signal_segment)
        #stack segments as columns in a 2D array
        signal_segments=np.column_stack(signal_segments)
        print(f"signal_segments shape: {signal_segments.shape}")

        #store the stacked segments under key value corresponding to event label
        key=f"label_{event}"
        epoched_signals[key]=signal_segments
    return epoched_signals

def compute_welch_estimation_on_segment2(segment:np.ndarray, sample_rate:float, nfft: int= None):
    """
    Compute PSD using welch method on a segment.
        Relies on the scipy.welch function.
    
    Parameters:
    ------
        segment (np.ndarray): samples of signal. 
        sample_rate (float) : sampling rate
        nfft (int): Length of the FFT used, for zero padding welch.

    Retruns:
    ------
        freqs (np.ndarray): array of welch frequencies.
        Pxx_density (np.ndarray): array of welch power densities.
    """

    segment_length = len(segment)  # number of points in the segment
    sub_segment_length=segment_length/4
    # split the segment in two sub segments with overlap of 50%
    freqs, Pxx_density = welch(segment, fs=sample_rate,
                               window="hann",
                               nperseg=sub_segment_length,detrend=False,
                               noverlap=sub_segment_length//2,nfft=nfft,axis=0)
    return freqs, Pxx_density

def compute_averaged_psds_over_trials(trials:np.ndarray,sample_rate:float,nfft:int=None):
    """
    Computes the the PSD of several sample trials and averages their results.

    Parameters
    ----------
        trials (np.ndarray): 2D array of trials where each column represents the signal of a trial
        sample_rate (float): Sampling rate of the signal
        nfft (int): Length of the FFT used, for zero padding welch
 
    Returns
    ----------
        mean_frequencies (np.ndarray): 1D array of PSD estimation frequency results
        mean_pxx_densities (np.ndarray): 1D array of averaged PSD estimation power results
    """
    print(f"segment shapes - {trials.shape}")
    frequencies=[]
    Pxx_densities=[]
    #compute the PSD estimation for each trial
    for trial in trials.T:
        freqs, Pxx_density=compute_welch_estimation_on_segment2(segment=trial, sample_rate=sample_rate,nfft=nfft)
        frequencies.append(freqs)
        Pxx_densities.append(Pxx_density)

    Pxx_densities=np.column_stack(Pxx_densities)
    frequencies=np.column_stack(frequencies)
    #average the PSDs
    mean_frequencies=np.mean(frequencies,axis=1) #equals every frequency column of the frequencies array anyway
    mean_pxx_densities=np.mean(Pxx_densities,axis=1)
    return mean_frequencies,mean_pxx_densities

def compute_welch_on_a_signal_before_each_marker(signal:np.ndarray, sample_rate:float, markers_array:np.ndarray, segment_duration:float):
    """
    Computes the DSP estimations of all segments of a signal.

    The signal has as many segments as it has of markers. Each segment streches over the specified segment duration and ends at a marker.
    The DSPs are then computed BEFORE each marker
    
    Note: 
        Uses the markers_array indexes to define the segments.

    For opposite operation, see the complementary function 'compute_welch_on_a_signal_after_each_marker()'.


    Parameters:
    ------
        signal (np.ndarray): 1D array of signal samples.
        sample_rate (float) : Signal sampling rate.
        markers_array (dict): Dictionary of 2D arrays under 3 keys as ("markers_timestamp_indices","markers_timestamps", "marker_labels")
        segment_duration (float): length of the segment on which to compute the DSP estimation.
        
    Retruns:
    ------
        freqs (np.ndarray): array of welch frequencies.
        Pxx_density (np.ndarray): array of welch power densities.
    """
    N = len(signal)
    freqs_for_all_markers = []
    PSDs_for_all_markers = []
    for marker in markers_array["markers_timestamp_indices"]:

        segment_coordinates = get_segment_coordinates2(
            reference_index=marker, segment_time_length=segment_duration, sample_rate=sample_rate)

        higher_end = int(segment_coordinates["reference_end"])
        lower_end = int(segment_coordinates["lower_end"])

        signal_segment, full_signal_length = get_signal_segement2(
            signal=signal, lower_end=lower_end, higher_end=higher_end)
        freqs, Pxx_density = compute_welch_estimation_on_segment2(
            signal_segment, sample_rate=sample_rate, nfft=None)
        freqs_for_all_markers.append(freqs)
        PSDs_for_all_markers.append(Pxx_density)

    freqs_for_all_markers = np.column_stack(freqs_for_all_markers)
    PSDs_for_all_markers = np.column_stack(PSDs_for_all_markers)

    return {"PSD_frequencies": freqs_for_all_markers, "PSD_magnitudes": PSDs_for_all_markers}


def compute_welch_on_a_signal_after_each_marker(signal:np.ndarray, sample_rate:float, markers_array:np.ndarray, segment_duration:float):
    """
    Computes the DSP estimations of all segments of a signal.

    The signal has as many segments as it has of markers. Each segment starts at a marker and streches over the specified segment duration.
    The DSPs are then computed AFTER each marker
    
    Note: 
        Uses the markers_array indexes to define the segments.

    For opposite operation, see the complementary function 'compute_welch_on_a_signal_before_each_marker()'.

    Parameters:
    ------
        signal (np.ndarray): 1D array of signal samples.
        sample_rate (float) : Signal sampling rate.
        markers_array (dict): Dictionary of 2D arrays under 3 keys as ("markers_timestamp_indices","markers_timestamps", "marker_labels")
        segment_duration (float): length of the segment on which to compute the DSP estimation.
        
    Retruns:
    ------
        freqs (np.ndarray): array of welch frequencies.
        Pxx_density (np.ndarray): array of welch power densities.
    """

    N = len(signal)
    freqs_for_all_markers = []
    PSDs_for_all_markers = []
    for marker in markers_array["markers_timestamp_indices"]:

        segment_coordinates = get_segment_coordinates2(
            reference_index=marker, segment_time_length=segment_duration, sample_rate=sample_rate)

        higher_end = int(segment_coordinates["higher_end"])
        reference_end = int(segment_coordinates["reference_end"])

        signal_segment, full_signal_length = get_signal_segement2(
            signal=signal, lower_end=reference_end, higher_end=higher_end)
        freqs, Pxx_density = compute_welch_estimation_on_segment2(
            signal_segment, sample_rate=sample_rate, nfft=None)
        freqs_for_all_markers.append(freqs)
        PSDs_for_all_markers.append(Pxx_density)

    freqs_for_all_markers = np.column_stack(freqs_for_all_markers)
    PSDs_for_all_markers = np.column_stack(PSDs_for_all_markers)

    return {"PSD_frequencies": freqs_for_all_markers, "PSD_magnitudes": PSDs_for_all_markers}


# =============================================================================
############################# Signal preprocessing  ###########################
# =============================================================================
def detrend_signals(raw_signals:np.array):
    """
    Remove linear trends from signals.

    Parameters
    ----------
    raw_signals : ndarray
        Array of raw signals arranged as columns.

    Returns
    ----------
    EEG_amplitudes_centered : ndarray
        Array of detrended signals arranged as columns (same shape as raw_signals).
    """
    print(f"input_signals shape:\n {raw_signals.shape}")
    print(f"input_signals mean per signal:\n {np.mean(raw_signals,axis=0)}")
    EEG_amplitudes_centered=raw_signals-np.mean(raw_signals,axis=0)
    return EEG_amplitudes_centered

def rereference_signals(input_signals:np.array):
    """
    Reference signals to average.

    Parameters
    ----------
    input_signals : ndarray
        Array of signals arranged as columns.

    Returns
    ----------
    EEG_amplitudes_rereferenced : ndarray
        Array of signals arranged as columns (same shape as input_signals).
    """
    print(f"input_signals shape:{input_signals.shape}")
    print(f"input_signals whole mean:{np.mean(input_signals)}")
    EEG_amplitudes_rereferenced=input_signals-np.mean(input_signals)
    
    return EEG_amplitudes_rereferenced

def filter_signal(input_signals:np.array,sample_rate:int,order:int,cutofffreq:tuple):
    """
    Applies a series of filters on signals.
    The first two input frequencies of the tuple cutofffreq are corrected before use by the adequate filters.

    Parameters
    ----------
    input_signals : ndarray
        Array of signals arranged as columns.
    sample_rate : int
        Sampling rate of the signals.
    order : int
        Order of the band pass filter. Also required to apply right correction to cutofffreq[0,2]
    cutofffreq : tuple
        Cutoff frequencies to be used by filters. 
        Tuple must be len(cutofffreq) = 2 or 3.
        Ordered as: cutofffreq=(low_cutoff_freq,low_cutoff_freq,notch_cutoff_freq)
        

    Returns
    ----------
    EEG_Filtered_NOTCH_BP : ndarray
        Array of signals arranged as columns (same shape as input_signals).
    freq_test_BP : ndarray
        Frequency vector for verification of the filter response.
    magnitude_test_BP : ndarray
        magnitude vector for verification of the filter response.
    """
    if len(cutofffreq) < 2 or len(cutofffreq) > 3:
        raise ValueError(f"cutofffreq tuple length:{len(cutofffreq)} - Input tuple length must be between 2 and 3")
    try:
        if len(cutofffreq)==3:
            LOW_CUTOFF_FREQ_THEORETICAL,HIGH_CUTOFF_FREQ_THEORETICAL,NOTCH_CUTOFF_FREQ=cutofffreq
        elif len(cutofffreq)==2:
            LOW_CUTOFF_FREQ_THEORETICAL,HIGH_CUTOFF_FREQ_THEORETICAL=cutofffreq
            NOTCH_CUTOFF_FREQ=None
        # cutoff frequency correction for filtfilt application
        LOW_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
            order, LOW_CUTOFF_FREQ_THEORETICAL, sample_rate, pass_type="high_pass")

        HIGH_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
            order, HIGH_CUTOFF_FREQ_THEORETICAL, sample_rate, pass_type="low_pass")

        """        print("LOW_CUTOFF_FREQ_THEORETICAL="+str(LOW_CUTOFF_FREQ_THEORETICAL) +
            ", HIGH_CUTOFF_FREQ_THEORETICAL="+str(HIGH_CUTOFF_FREQ_THEORETICAL))
                print("LOW_CUTOFF_FREQ_CORRECTED="+str(LOW_CUTOFF_FREQ_CORRECTED) +
            ", HIGH_CUTOFF_FREQ_CORRECTED="+str(HIGH_CUTOFF_FREQ_CORRECTED))"""
        
        print(f"LOW_CUTOFF_FREQ_THEORETICAL={LOW_CUTOFF_FREQ_THEORETICAL},HIGH_CU-TOFF_FREQ_THEORETICAL={HIGH_CUTOFF_FREQ_THEORETICAL}")
        print(f"LOW_CUTOFF_FREQ_CORRECTED={LOW_CUTOFF_FREQ_CORRECTED},HIGH_CUTOFF_FREQ_CORRECTED={HIGH_CUTOFF_FREQ_CORRECTED}")




        # Filtering on all channels
        if NOTCH_CUTOFF_FREQ is not None:
            # 1-Notch-filter 50Hz [49,51] the centered-rereferenced signal
            print(f"NOTCH_CUTOFF_FREQ={NOTCH_CUTOFF_FREQ}")

            input_signals, freqs_test_NOTCH, magnitudes_test_NOTCH = notch_filter(input_signal=input_signals,
                                                                                    sample_rate=sample_rate,
                                                                                    cutoff_freq=NOTCH_CUTOFF_FREQ,
                                                                                    stop_band_width=2)

        # 2-Then Band-pass filter the signal filtered by notch
        EEG_Filtered_NOTCH_BP, freq_test_BP, magnitude_test_BP = band_pass_filter(input_signal=input_signals,
                                                                                sample_rate=sample_rate,
                                                                                low_cutoff_freq=LOW_CUTOFF_FREQ_CORRECTED,
                                                                                high_cutoff_freq=HIGH_CUTOFF_FREQ_CORRECTED,
                                                                                filter_order=order)
        print("Filtered signal shape:",np.shape(EEG_Filtered_NOTCH_BP))
    except UnboundLocalError :
        print(f"Specified cutofffreq={cutofffreq} - The tuple must contain only 2 or 3 elements as (low_cutoff_freq,high_cutoff_freq,notch_cutoff_freq)")
    return EEG_Filtered_NOTCH_BP, freq_test_BP, magnitude_test_BP


# =============================================================================
############################## Matlab vs Python  ##############################
# =============================================================================
def rms(series1,series2,name:str,units:str="Units (NA)"): #litteral formula
    """
    Compute the root mean squared error of two series.
    Returns the value and prints it

    Parameters:
    ----------
        series1 (np.ndarray): 1D series to compare
        series2 (np.ndarray): 1D series to compare

    Returns:
    ----------
        rms (float): root mean squared error of two series
    """
    #rms=np.sqrt(((python - matlab) ** 2).mean())

    diff=series1-series2
    squared_diff=diff**2
    mean_squared_diff=np.mean(squared_diff)
    rms=np.sqrt(mean_squared_diff)
    print(name+" = ",rms," (µV²/Hz)")
    return rms

def cv_percent(series1,series2):
    """
    [UNUSED]Compute coeffcient of variation (CV) of two series.
    Returns the result expressed in (%)

    Parameters:
    ----------
        series1 (np.ndarray): 1D series to compare
        series2 (np.ndarray): 1D series to compare

    Returns:
    ----------
        cv (float): coeffcient of variation  of two series
    """
    diff=series1-series2
    #diff=np.std([series1,series2])
    
    squared_diff=diff**2
    var=np.sqrt(squared_diff)
    mean=(series1+series2)/2
    cv=(var/mean)*100
    return cv

def abs_distance(series1,series2):
    """
    Compute the absolute differences of two series elementwise.
    Returns a series of absolute differences.

    Parameters:
    ----------
        series1 (np.ndarray): 1D series to compare
        series2 (np.ndarray): 1D series to compare

    Returns:
    ----------
        absolute_diff (np.ndarray): array of absolute differences.
    """
    diff=series1-series2
    absolute_diff=abs(diff)
    return(absolute_diff)

def list_matlab_psd_results_filenames(input_signal_filename:str,channels_dict:dict[str],selected_channel_numbers:list[int]):
    """
    Lists the expected csv filenames of the matlab psd results for the selected channels.

    Parameters:
    ----------
        input_signal_filename (str): name of the input eeg data file (with its csv extension)
        channels_dict (dict): Dictionary of channel numbers as keys (ie Channel_x ) and names as values (ie C3)
        selected_channel_numbers (list): List of the selected channel numbers (integers)

    Returns:
    ----------
        matlab_results_filename_list(list): dictionary of the frequencies and PSDs estimates from matlabs FFT
        PSD_p_results (dict): dictionary of the frequencies and PSDs estimates from matlabs periodogram function
        PSD_w_results (dict): dictionary of the frequencies and PSDs estimates from matlabs welch function
    """
    matlab_results_filename_list=[]
    channel_indexes=selected_channel_numbers-1
    print("selected channels :")
    for (i,y) in zip(selected_channel_numbers,channel_indexes):
        print(f"channel number:{i}, channel index:{y}")
        channel_name=f"Channel_{str(i)}_{channels_dict['Channel_'+str(i)]}"
        print(channel_name)

        #get name of corresponding matlab psd result 
        filenamei=f"MATLAB_PSD_res_EEG_{channel_name}_{input_signal_filename }"

        matlab_results_filename_list.append(filenamei)

    print(f"matlab psd results file names: {matlab_results_filename_list}")

    return matlab_results_filename_list

def import_psd_results2(psd_results_file_name:str):
    """
    Imports psd data results generated (beforehand) by the matlab script.
    Matlab psd results must be stored in the 'STAGE_SIGNAL_PHYSIO/DAT/OUTPUT/Matlab_PSD_Results' folder as csv files to be retrieved.

        Returns 3 dictionaries (for each PSD estimation method) of two key-value pairs : 
        (key1:"frequencies", value:(1D)array of frequencies) \n
        (key2:"psds", value: (1D)array of PSD)

    Parameters:
    ----------
        psd_results_file_name (str) :name of the input eeg data file (with its extension)

    Returns:
    ----------
        PSD_fft_results (dict): dictionary of the frequencies and PSDs estimates from matlabs FFT
        PSD_p_results (dict): dictionary of the frequencies and PSDs estimates from matlabs periodogram function
        PSD_w_results (dict): dictionary of the frequencies and PSDs estimates from matlabs welch function

    """
    #filename="MATLAB_PSD_res_EEG_Channel_5_C3_001_MolLud_20201112_1_c_preprocessed_499.998_Hz"

    filepath=f"./DAT/OUTPUT/Matlab_PSD_Results/{psd_results_file_name}"
    print(filepath,type(filepath))
    
    matlab_data = np.genfromtxt(filepath, delimiter=';',skip_header=1)

    PSD_fft_results = {
        "frequencies": matlab_data[:, 0], "psds": matlab_data[:, 1]}
    PSD_p_results = {
        "frequencies": matlab_data[:, 2], "psds": matlab_data[:, 3]}
    PSD_w_results = {
        "frequencies": matlab_data[:, 4], "psds": matlab_data[:, 5]}

    return PSD_fft_results,PSD_p_results,PSD_w_results

def export_xdf_eeg_to_csv(xdf_filepath:str,PROCESS_SIGNAL:bool=False):
    """
    Access the xdf file, finds eeg stream and exports all channels data to csv.
    Parameters:
    ----------
        xdf_filepath (str): filepath to the xdf file.
        PROCESS_SIGNAL (bool): boolean to specify if xdf data must be preprocessed or not before export

    Returns:
    ----------
         exportfilename (str): filename of the exported file
    """
    #### import raw data

    # Define xdf file path
    input_filepath = xdf_filepath
    INPUT_FILENAME = os.path.splitext(os.path.basename(input_filepath))[0]
    # path=os.path.normpath("../DAT/Input/001_MolLud_20201112_1_c.xdf")


    print("Input filepath: ",input_filepath)
    print("Input filename: ",INPUT_FILENAME)

    # Loading streams of interest

        # load_xdf returns selected streams as list of streams (ie.dictionary)
    EEG_Stream, EEG_fileheader = pyxdf.load_xdf(input_filepath, select_streams=[{'type': 'EEG'}])
    Mouse_markers_Stream, Mouse_markers_header = pyxdf.load_xdf(input_filepath, select_streams=[{'type': 'Markers', 'name': 'MouseToNIC'}])

        # in case multiple streams havee been found
    if len(EEG_Stream) and len(Mouse_markers_Stream) != 1:
        raise ValueError("Multiple streams matching type restriction")
    else: 
        EEG_Stream=EEG_Stream[-1] #access to the only EEG stream of the list
        Mouse_markers_Stream=Mouse_markers_Stream[-1]

    # Get sampling rate
    Srate = EEG_Stream["info"]["effective_srate"]
    
    # Manage EEG data
        # Get amplitudes of each electrode
    EEG_raw_amplitudes = EEG_Stream["time_series"]

        # EEG channel name definition
    channels_dict = {"Channel_1": "C4",
                    "Channel_2": "FC2",
                    "Channel_3": "FC6",
                    "Channel_4": "CP2",
                    "Channel_5": "C3",
                    "Channel_6": "FC1",
                    "Channel_7": "FC5",
                    "Channel_8": "CP1"}
    
        
        # Format EEG times_stamps (in unix time epoch) to seconds relative to the execution time of the recording
    EEG_times = EEG_Stream["time_stamps"]-EEG_Stream["time_stamps"][0]

    # Manage Markers data
        # Select the marker labels
    Markers_labels = Mouse_markers_Stream["time_series"]

        # Markers start with a time relative to the execution time of the recording
    Marker_times = (
        Mouse_markers_Stream["time_stamps"]-EEG_Stream["time_stamps"][0])

        # Stacking maker timestamps and labes as 2D array : [[markers_timesstamps],[markers_labels]]
    Markers_times_labels = np.column_stack((Marker_times, Markers_labels))

    #### Prepare data for export
        # Process signals or not?
    print("PROCESS_SIGNAL ? --",PROCESS_SIGNAL)

    if PROCESS_SIGNAL is False:
            print("Keeping raw signals...")
            EEG_for_export=EEG_raw_amplitudes
            DATA_STATUS="raw"

    elif PROCESS_SIGNAL is True:
        print("Processing signals")
        print("Detrending...")
        EEG_amplitudes_centered=detrend_signals(EEG_raw_amplitudes)
        print("Rereferencing...")
        EEG_amplitudes_rereferenced=rereference_signals(input_signals=EEG_amplitudes_centered)
        print("Filtering...")
        EEG_amplitudes_centered_filtered,_,_=filter_signal(input_signals=EEG_amplitudes_rereferenced,
                                                    sample_rate=Srate,
                                                    order=8,cutofffreq=(5,100,50))
        EEG_for_export=EEG_amplitudes_centered_filtered
        DATA_STATUS="prepro"
    else:
        raise ValueError('PROCESS_SIGNAL must be Boolean (True or False)')
    
    print(f"EEG_for_export shape : {EEG_for_export.shape}")

        #Stack columns as electrodes signals followed by column of timestamps
    amplitudes_times=np.column_stack((EEG_for_export,EEG_times))

    #### Export to CSV file

    # Create header for CSV
    header = ', '.join([f"{key}:{value}" for key, value in channels_dict.items()])
    header=header+',time(sec)'
    print("export data header :",header)

    # Create filename
    exportfilename=f"{INPUT_FILENAME}_{DATA_STATUS}_{round(Srate,3)}_Hz.csv"
    exportfilepath=os.path.normpath("DAT/INPUT/"+exportfilename)

    print(f"Input filepath : {input_filepath}")
    print(f"Output filepath : {exportfilepath}")

    # Export
    np.savetxt(exportfilepath, amplitudes_times, delimiter=',', header=header, comments='', fmt='%d')
    #np.savetxt(exportfilepath,times_amplitudes,delimiter=",")

    return exportfilename
