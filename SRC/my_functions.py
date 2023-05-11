
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch  # library for creating filters


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
            streamindex = data.index(i)
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
            print(i)
            # plot a line x=time_stamp associated to the i marker of type 111 (begining of task)
            print("plot_type is : ", type(plot_type), plot_type)
            marker111 = plot_type.axvline(x=i[0], color="b", label="111")
        else:
            print(i)
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

            axis[a, b].set_title("Electrode "+str(count) +
                                 ":" + channels["Channel_"+str(count+1)])
            axis[a, b].set_xlabel(xlabel)
            axis[a, b].set_ylabel(ylabel)
            count = count+1
    plt.suptitle(fig_title+"\n"+filename)
    plt.legend()
    figure.show()

# =============================================================================
############################# Compute_FFT_on_channels  ########################
# =============================================================================


def compute_fft_on_channels(EEG_channels_signal: np.ndarray, Sampling_rate: int | float):
    """
    Function that computes the FFT on each channel.

    Each column of the array is the signal of an electrode.

    inputs: numpy.ndarray(2D) and float as EEG_channels_signal and Sampling_rate
    outputs: Dictionary [key1:array1,key2:array2] as [frequencies:values,amplitudes:values]
    """

    # Computes the FFT returns one array of frequencies
    fft_frequencies = np.fft.fftfreq(
        len(EEG_channels_signal), d=1/Sampling_rate)

    # only consider the positive frequencies
    fft_frequencies = fft_frequencies[0:len(fft_frequencies)//2]

    # compute fft iteratively on each channel, store them in array, each column an electrode
    FFT_Results_EEG_channels = []
    for column in range(EEG_channels_signal.shape[1]):
        print(column)
        fft_signal_electrodes = abs(np.fft.fft(EEG_channels_signal[:, column]))
        fft_signal_electrodes = fft_signal_electrodes[0:len(
            fft_signal_electrodes)//2]
        FFT_Results_EEG_channels.append(fft_signal_electrodes)

    # Consistent shaping of data
    FFT_Results_EEG_channels = np.array(FFT_Results_EEG_channels)
    FFT_Results_EEG_channels = FFT_Results_EEG_channels.transpose()

    return {"fft_frequencies": fft_frequencies, "FFT_Results_EEG_channels": FFT_Results_EEG_channels}

# =============================================================================
########################## Cutoff frequency corrector  ########################
# =============================================================================


def filtfilt_cutoff_frequency_corrector(order: int, cutoff_freq: float | int, sampling_freq: float | int, pass_type: str = ["low_pass", "high_pass"]):
    """
    Function that corrects cutoff frequencies to use in combination with filt.filt()

    As a zero-phase filter (linear filter) is applied to a signal 
    the cutoff freq are diminished. The correction depends also on the order. 
    The adjustment is made on the angular cutoff frequency, which depends on the filter direction (LP,HP).

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


def compute_lagged_psd(EEG_data: np.ndarray, Srate: float | int, markers: np.ndarray, time_lag: float | int = 1, direction: str = "before"):
    """
    Computes the time lagged PSDs (ex 1 s before each marker)

    Uses markers|EEG time stamps as reference to compute PSD around each.
    Specify the direction and segment length relative to marker for calculation (1s default).
    Iterate over each marker for each electrode
    Stores the results as two arrays  

    returns results as 2 3d arrays of frequencies and PSDs for each electrode.
    Each layer of the returned 3d array being an electrode with for each a column per marker(rows,marker,electrode).

    inputs: numpy.ndarray(2D),float,numpy.ndarray(2D),float,str
    outputs: numpy.ndarray(3D),numpy.ndarray(3D) as tridi_frequencies, tridi_PSDs
    """

    # time expressed in number of points, Srate number of points per sec
    delta_index = int(Srate*time_lag)

    print("delta_index:", delta_index)
    layersPSDs = []
    layersFrequencies = []
    for column in range(EEG_data.shape[1]):  # iterate on electrodes
        electrode_lagged_PSDS = []
        elecrode_frequencies = []
        print("eeg filtered col:", column)
        for timestamp_index in markers[:, 0]:  # iteration on markers
            print("marker timestamp:", int(timestamp_index),
                  "delta_index:", delta_index)

            # Define the segment coordinates (start,end)
            # PSD on a range of time delta_time before the time stamp
            lower_end = int(timestamp_index)-delta_index
            print("lower_end:", lower_end)

            # PSD on a range of time delta_time after the time stamp
            higher_end = int(timestamp_index)+delta_index
            print("Higher_end:", higher_end)

            # index_range=np.arange(lower_range,int(timestamp_index)+1)
            # +1 to inclue the value at the time stamp
            reference_end = int(timestamp_index)

            print("index_range:", "(", lower_end, reference_end, ")")
            print("delta_index:", delta_index)

            # Compute the welch method in accordance to the direction deisred
            if direction == "before":
                freq, Pxx_density = welch(EEG_data[lower_end:reference_end+1, column],
                                          fs=Srate, window="hann", nperseg=delta_index, noverlap=delta_index//2, axis=0)
            elif direction == "after":
                freq, Pxx_density = welch(EEG_data[reference_end:higher_end+1, column],
                                          fs=Srate, window="hann", nperseg=delta_index, noverlap=delta_index//2, axis=0)

            # liste d'array (chaque array est le PSD calculé pour chaque marqueur bas)
            electrode_lagged_PSDS.append(Pxx_density)
            elecrode_frequencies.append(freq)

            # array 2d (x=PSD,y=quelmarqueur) une electrode
            electrode_stacked_markers = np.column_stack(electrode_lagged_PSDS)
            electrode_stacked_frequencies = np.column_stack(
                elecrode_frequencies)

        layersPSDs.append(electrode_stacked_markers)
        layersFrequencies.append(electrode_stacked_frequencies)

        tridi_PSDs = np.stack(layersPSDs)
        tridi_frequencies = np.stack(layersFrequencies)

    return tridi_frequencies, tridi_PSDs

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
