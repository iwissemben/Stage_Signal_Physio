
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch  # library for creating filters


# =============================================================================
############################# Stream_Finder_by_name  #####################################
# =============================================================================

# EEG stream finder custom function
# Finds the index of the datalist element loaded from the ".xdf file" and returns it
# input arguments the list containing the data, and the name of the stream to search for


def Stream_Finder_by_name(datalist, name):
    print("StreamFinder executed")
    streamindex = None
    for i in datalist:
        if any(str(name).upper() in string.upper() for string in i["info"]["name"]):
            streamindex = data.index(i)
            print(streamindex)
            # print(i["info"]["type"])
            return streamindex


# =============================================================================
############################# Show_markers  #####################################
# =============================================================================

# Custom function to display on graphs Markers

def Show_markers(plot_type, markers):
    for i in markers:
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

# Custom multipurpose function that display single graphs


def Single_plot(filename, fig_number, x, y, fig_title, xlabel, ylabel, markers_times_array=None, point_style="-k", line_width=1):

    plt.figure(fig_number)
    plt.plot(x, y, point_style, lw=line_width)
    plt.title(str(fig_title+"\n"+filename))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    if markers_times_array is not None:
        # cf custom function to display the markers' position on the graph
        Show_markers(plt, markers_times_array)
    plt.legend()

# =============================================================================
############################# Mosaic_plotter  #################################
# =============================================================================

# Custom function that display a mosaic of graphs for each channel's time signal
# inputs: [numpy.ndarray(1D),numpy.ndarray(2D),numpy.ndarray(2D),matplotlib.figure.Figure,numpy.ndarray] as [x_axis,y_axis,markers_and_their_timestamps,figure_name,figure_axis]


def Mosaic_plot(filename, x, y, Markers_times_labels, figure, axis, fig_title, xlabel, ylabel, channels):
    count = 0
    for a in range(0, 2):  # two rows of graphs
        for b in range(0, 4):  # each row conists of 4 graphs
            axis[a, b].plot(x, y[:, count])
            # cf custom function to display the markers' position on the graph
            # Show_markers("axis["+str(a)+","+str(b)+"]", Markers_times_labels)
            Show_markers(axis[a, b], Markers_times_labels)

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

# Function Compute of the FFT on each channel
# inputs: [array,float] as [electrodes' signals, sampling rate]


def Compute_FFT_on_channels(EEG_channels_signal, Sampling_rate):
    fft_frequencies = np.fft.fftfreq(
        len(EEG_channels_signal), d=1/Sampling_rate)
    fft_frequencies = fft_frequencies[0:len(fft_frequencies)//2]
    FFT_Results_EEG_channels = []

    for column in range(EEG_channels_signal.shape[1]):
        print(column)
        fft_signal_electrodes = abs(np.fft.fft(EEG_channels_signal[:, column]))
        fft_signal_electrodes = fft_signal_electrodes[0:len(
            fft_signal_electrodes)//2]
        FFT_Results_EEG_channels.append(fft_signal_electrodes)

    FFT_Results_EEG_channels = np.array(FFT_Results_EEG_channels)
    FFT_Results_EEG_channels = FFT_Results_EEG_channels.transpose()

    return {"fft_frequencies": fft_frequencies, "FFT_Results_EEG_channels": FFT_Results_EEG_channels}

# =============================================================================
########################## Cutoff frequency corrector  ########################
# =============================================================================

# Function that corrects cutoff frequencies to use in combination with filt.filt()


def filtfilt_cutoff_frequency_corrector(order, cutoff_freq, sampling_freq, pass_type="low_pass"):
    # Biomechanics and Motor Control of Human Movement, 4th-edition (page 69)
    # https://www.codeproject.com/Articles/1267916/Multi-pass-Filter-Cutoff-Correction
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

# Function that finds nearest index (in EEG_times) to the each marker timestamp
# inputs={EEG_times array,Markers_times_labels_array} outputs={array of neareast EEG_times indices with their times (s)}


def Nearest_timestamps_array_finder(EEG_times_stamps, markers):
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

# Function that computes the time lagged PSDS (ex 1 s before each marker)
# returns results as 3d arrays of frequencies and PSDs for each electrode
# array1{Frequencies,dim=[electroden,frequencies,markeri]} array2{Psds,dim=[electroden,PSD,markeri]}


def Compute_lagged_PSD(EEG_data, Srate, markers, time_lag=1, direction="before"):
    # 1s lag by default
    # time expressed in number of points, Srate number of points per sec
    delta_index = int(Srate*time_lag)

    print("delta_index:", delta_index)
    layersPSDs = []
    layersFrequencies = []
    for column in range(EEG_data.shape[1]):  # iteration sur les electrodes
        electrode_lagged_PSDS = []
        elecrode_frequencies = []
        print("eeg filtered col", column)
        for timestamp_index in markers[:, 0]:  # iteration sur les marqueurs
            print("marker timestamp:", int(timestamp_index),
                  "delta_index:", delta_index)

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

# compute the average densities among trials of a block and returns 3d arrays of 1 column (average), PSD values, 8
# returns array1{block1_event_typex_averagedPSDs,dim=[averaged_ratios_block1_eventx,electrodei]},array2{block2_event_typex_averagedPSDs,dim=[averaged_ratios_block2_eventx,electrodei]]}


def Compute_average_ratio_for_event_on_blocks_for_allelectrodes(mat3d):
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
