
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch  # library for creating filters


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


def fft_create_positive_frequency_vector(signal: np.ndarray, Sampling_rate: int | float):
    """
    produces a positive frequency vector for fft

    inputs: numpy.ndarray(2D) and float as EEG_channels_signal and Sampling_rate
    outputs: Dictionary [key1:array1,key2:array2] as [frequencies:values,amplitudes:values]
    """
    # Return the Discrete Fourier Transform sample positive frequencies.
    fft_frequencies = np.fft.fftfreq(len(signal), d=1/Sampling_rate)
    fft_frequencies = fft_frequencies[0:len(fft_frequencies)//2]
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
    fft_signal = fft_signal[0:len(fft_signal)//2]
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

def get_segment_coordinates(reference_index:int,segment_length:int):
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

    print("index_range:", "(", lower_end, reference_end, ")")
    print("delta_index:", segment_length)
    return lower_end,higher_end,reference_end

def compute_welch_estimation_on_segment(signal:np.ndarray,direction:str,sample_rate:int,
                                        reference_end:int,lower_end:int,higher_end:int,delta_index:int):
    """
    Computes the psd estimation(welch method) on a specific segment of a time signal.

    inputs:numpy.ndarray(1D),str,numpy.ndarray(2D),float,str
    outputs:numpy.ndarray(1D),numpy.ndarray(1D) as columns
    """
    if direction == "before":
        freq, Pxx_density = welch(signal[lower_end:reference_end+1],
                                    fs=sample_rate, window="hann", nperseg=delta_index, noverlap=delta_index//2, axis=0)
    elif direction == "after":
        freq, Pxx_density = welch(signal[reference_end:higher_end+1],
                                    fs=sample_rate, window="hann", nperseg=delta_index, noverlap=delta_index//2, axis=0)
    else:
        print("Wrong direction provided, please specify either 'before' or 'after'")
    return freq,Pxx_density


def compute_lagged_psds_one_signal(signal:np.ndarray,Srate:float|int, markers:np.ndarray, 
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

    #iterate on the markers to compute on each the psd on a given direction
    for timestamp_index in markers[:, 0]:
        #get the coordinates of the segment on which the psd will be computed
        lower_end,higher_end,reference_end=get_segment_coordinates(reference_index=timestamp_index,segment_length=delta_index)

        # Compute the welch method on the segment in accordance with the deisred direction 
        freq,Pxx_density=compute_welch_estimation_on_segment(signal,direction,Srate,reference_end,lower_end,higher_end,delta_index)
        # Store the result columns in lists (Nmarkers length)
        electrode_lagged_PSDS.append(Pxx_density)
        elecrode_frequencies.append(freq)

        #Create layers: Stack elements of the lists (1D array) to create two 2D arrays (x=PSD,y=markeri) (x=freqs,y=markeri) 
        electrode_stacked_markers = np.column_stack(electrode_lagged_PSDS)
        electrode_stacked_frequencies = np.column_stack(elecrode_frequencies)
    return electrode_stacked_frequencies,electrode_stacked_markers


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
        #Produce a layer (2d array) of PSDs for an electrode
        electrode_stacked_frequencies,electrode_stacked_markers=compute_lagged_psds_one_signal(electrode, Srate, markers,
                                                                    time_lag=time_lag, direction=direction)
        #store the layers in a list
        layers_psds.append(electrode_stacked_markers)
        layers_frequencies.append(electrode_stacked_frequencies)

    #Stack each layer to get a 3d array (x=psdx or freqx ,y=markery,z=electrodez)
    tridi_PSDs = np.stack(layers_psds,axis=2)
    tridi_frequencies = np.stack(layers_frequencies,axis=2)

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