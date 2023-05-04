# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:02:00 2022

@author: iWiss
"""

# xdf file importation

import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt, welch  # library for creating filters

plt.close("all")  # close all figure windows

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

# Custom function to display on graphs Markers


def Show_markers(plot_type, markers):
    for i in markers:
        if i[1] == 111:
            print(i)
            # plot a line x=time_stamp associated to the i marker of type 111 (begining of task)
            marker111 = eval(plot_type).axvline(x=i[0], color="b", label="111")
        else:
            print(i)
            # plot a line x=time_stamp associated to the i marker of type 110 (begining of task)
            marker110 = eval(plot_type).axvline(x=i[0], color="r", label="100")
    return marker111, marker110

# Custom multipurpose function that display single graphs


def Single_plot(fig_number, x, y, fig_title, xlabel, ylabel, markers_times_array=None, point_style="-k", line_width=1):

    plt.figure(fig_number)
    plt.plot(x, y, point_style, lw=line_width)
    plt.title(str(fig_title+"\n"+filename))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    if markers_times_array is not None:
        # cf custom function to display the markers' position on the graph
        Show_markers("plt", markers_times_array)
    plt.legend()

# Custom function that display a mosaic of graphs for each channel's time signal


# inputs: [numpy.ndarray(1D),numpy.ndarray(2D),numpy.ndarray(2D),matplotlib.figure.Figure,numpy.ndarray] as [x_axis,y_axis,markers_and_their_timestamps,figure_name,figure_axis]
def Mosaic_plot(x, y, Markers_times_labels, figure, axis, fig_title):
    count = 0
    for a in range(0, 2):  # two rows of graphs
        for b in range(0, 4):  # each row conists of 4 graphs
            axis[a, b].plot(x, y[:, count])
            # cf custom function to display the markers' position on the graph
            Show_markers("axis["+str(a)+","+str(b)+"]", Markers_times_labels)
            axis[a, b].set_title("Electrode "+str(count) +
                                 ": "+channels_dic["Channel_"+str(count+1)])
            axis[a, b].set_xlabel("Temps (s)")
            axis[a, b].set_ylabel(
                "Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][count]["unit"][0])+")")
            count = count+1
    plt.suptitle(fig_title+"\n"+filename)
    plt.legend()
    figure.show()

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
################################## initialization #############################
# =============================================================================
# Define the xdf file path
filename = "001_MolLud_20201112_1_c.xdf"
# filename="020_DesMar_20211129_1_c.xdf"
# path=os.path.normpath("../DAT/Input/001_MolLud_20201112_1_c.xdf")
path = os.path.normpath("../DAT/INPUT/"+filename)


# Load only streams of interest (EEG signal and Mouse task Markers) from the xdf data file
# data, header = pyxdf.load_xdf(path, select_streams=[{'type': 'EEG', 'name': 'LSLOutletStreamName-EEG'},{'type': 'Markers', 'name': 'MouseToNIC'}] )
data, header = pyxdf.load_xdf(path, select_streams=[{'type': 'EEG'}, {
                              'type': 'Markers', 'name': 'MouseToNIC'}])

# data, header = pyxdf.load_xdf(path)

# Selection of the streams of interest in the xdf file
for i in range(len(data)):
    print(i)
    if data[i]["info"]["type"] == ['EEG']:
        print(i, "est EEG")
        EEG_Stream = data[i]  # selecting EEG stream
    elif data[i]["info"]["type"] == ['Markers']:
        print(i, "est Marker")
        Mouse_markers_Stream = data[i]  # selecting markers stream

# declaring recording parameters
Srate = EEG_Stream["info"]["effective_srate"]


# definition of the EEG chanels' names
channels_dic = {"Channel_1": "C4",
                "Channel_2": "FC2",
                "Channel_3": "FC6",
                "Channel_4": "CP2",
                "Channel_5": "C3",
                "Channel_6": "FC1",
                "Channel_7": "FC5",
                "Channel_8": "CP1"}


# Specifying the timestamps of the markers relative to recording start
# Markers start with a time relative to the execution time of the recording
Marker_times = (
    Mouse_markers_Stream["time_stamps"]-EEG_Stream["time_stamps"][0])
# selecting the marker labels
Markers_labels = Mouse_markers_Stream["time_series"]

# Creation of a 2D array [[markers_timesstamps],[markers_labelslabels]]
Markers_times_labels = np.column_stack((Marker_times, Markers_labels))


# Data selection and formatting
# times_stamps in seconds relative to the execution time of the recording
EEG_times = EEG_Stream["time_stamps"]-EEG_Stream["time_stamps"][0]
# Amplitude of voltage recorded by each electrode of the recording set-up
EEG_raw_amplitudes = EEG_Stream["time_series"]

# =============================================================================
################################## Re-referencement ###########################
# =============================================================================
# Re-referencing: Uniformly distributed electrodes
EEG_raw_amp_mean = np.mean(EEG_raw_amplitudes)
EEG_raw_rereferenced_amplitudes = EEG_raw_amplitudes - \
    EEG_raw_amp_mean  # rereferencing


# =============================================================================
############################## Event-Markers ##################################
# =============================================================================
# Selection of one electrode and plotting its signal with markers

i = 2  # electrode number
electrodei = EEG_raw_rereferenced_amplitudes[:, i-1]

# plotting electrode i-1's signal for verification
Single_plot(fig_number=1, x=EEG_times, y=electrodei,
            fig_title=" Raw EEG Signal Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][i-1]["unit"][0])+")",
            markers_times_array=Markers_times_labels)

# Plotting all of the electrodes' RAW signals in one figure with 4*2=8 graphs with the markers

figure, axis = plt.subplots(2, 4)
Mosaic_plot(EEG_times, EEG_raw_rereferenced_amplitudes, Markers_times_labels,
            figure, axis, fig_title="Raw EEG signals per electrodes")

# =============================================================================
################################## FFT ########################################
# =============================================================================

# compute each channel's RAW FFT
EEG_FFT = Compute_FFT_on_channels(EEG_raw_rereferenced_amplitudes, Srate)

# plotting electrode i-1's raw signal FFT for verification
Single_plot(fig_number=3, x=EEG_FFT["fft_frequencies"], y=EEG_FFT["FFT_Results_EEG_channels"][:, i],
            fig_title="FFT of raw Signal EEG Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][i]["unit"][0])+")",
            point_style="-r", line_width=0.5)

# =============================================================================
################################## Filtering ##################################
# =============================================================================

filter_order = 4
# corrected cutoff frequency Low-pass butterworth filter
LPf = filtfilt_cutoff_frequency_corrector(
    filter_order, 40, Srate, pass_type="low_pass")
# corrected cutoff frequency High-pass butterworth filter
HPf = filtfilt_cutoff_frequency_corrector(
    filter_order, 1, Srate, pass_type="high_pass")


# Creation of a 4th order butterworth band pass filter
F_Nyquist = Srate/2

# creation of butterworth filters
# b, a = butter(4,[(0.3)/F_Nyquist,50/F_Nyquist],btype='bandpass') #band-pass with direct frequencies
# b, a = butter(4,[(0.3),50],btype='bandpass',fs=Srate)    #Band-pass with direct frequencies
# Band-pass with rectified frequencies
b, a = butter(filter_order, [HPf, LPf], btype='bandpass', fs=Srate)
# Low-pass  with rectified frequencies
bl, al = butter(filter_order, LPf, btype='low', fs=Srate)
# High-pass with rectified frequencies
bh, ah = butter(filter_order, HPf, btype='High', fs=Srate)

# Filtering of electrodes' signals

# Filtering on one channel
# filtered_signal_electrodei=filtfilt(b,a,electrodei) # Band-pass Filtering
filtered_signal_electrodei = filtfilt(
    bh, ah, electrodei)  # 1 High-pass Filtering
filtered_signal_electrodei = filtfilt(
    bl, al, filtered_signal_electrodei)  # 2 Then Low-pass Filtering

# Filtering on all channels
EEG_Filtered = filtfilt(b, a, EEG_raw_rereferenced_amplitudes,
                        axis=0)          # Band-pass Filtering
EEG_Filtered_LFHF = filtfilt(
    bh, ah, EEG_raw_rereferenced_amplitudes, axis=0)  # 1 High-pass Filtering
# 2 Then Low-pass Filtering
EEG_Filtered_LFHF = filtfilt(bl, al, EEG_Filtered_LFHF, axis=0)

# Plotting the filtered  electrode i-1's signal for verification
Single_plot(fig_number=4, x=EEG_times, y=filtered_signal_electrodei,
            fig_title=" Filtered EEG Signal Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][i-1]["unit"][0])+")",
            markers_times_array=Markers_times_labels, point_style=".k")


# compute each channel's FILTERED signal's FFT
EEG_Filtered_FFT = Compute_FFT_on_channels(EEG_Filtered, Srate)
EEG_Filtered_LFHF_FFT = Compute_FFT_on_channels(EEG_Filtered_LFHF, Srate)

# Plotting the filtered electrode i-1's FFT for verification
Single_plot(fig_number=5, x=EEG_Filtered_FFT["fft_frequencies"], y=EEG_Filtered_FFT["FFT_Results_EEG_channels"][:, i],
            fig_title="FFT of filtered Signal EEG Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][i]["unit"][0])+")",
            point_style=".r", line_width=0.5)

# Plotting all of the electrodes' FILTERED signals in one figure with 4*2=8 graphs with the markers

figure, axis = plt.subplots(2, 4)
Mosaic_plot(EEG_times, EEG_Filtered, Markers_times_labels, figure,
            axis, fig_title="Filtered EEG signals per electrodes")

# =============================================================================
############################# Periodogram #####################################
# =============================================================================

freq_res = 0.5  # Desired Frequency resolution in Hz (0.5)
nbpoints_per_epoch = Srate/freq_res  # as freq_res=Sampling_freq/Nbpoints
Acq_time = len(EEG_Filtered)/Srate

# PSD estimation for each electrodes
# freqs,Pxx_density=welch(EEG_Filtered[:,i],fs=Srate,window="hann",nperseg=nbpoints_per_epoch,noverlap=nbpoints_per_epoch//2)

freqs, Pxx_densities = welch(EEG_Filtered, fs=Srate, window="hann",
                             nperseg=nbpoints_per_epoch, noverlap=nbpoints_per_epoch//2, axis=0)


# Plotting electrodei's PSD
Single_plot(fig_number=7, x=freqs, y=Pxx_densities[:, i],
            fig_title="PSD of filtered Signal EEG Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="frequency (Hz)", ylabel="PSD Amplitude ("+str(EEG_Stream["info"]["desc"][0]["channel"][i-1]["unit"][0])+"²/Hz)",
            point_style=".g")


# =============================================================================

# Function that finds nearest index (in EEG_times) to the each marker timestamp
# inputs={EEG_times array,Markers_times_labels_array} outputs={array of neareast EEG_times indices with their times (s)}
def Nearest_timestamps_array_finder(EEG_times_stamps, markers):
    nearest_time_stamps = []
    nearest_time_stamps_indices = []
    for i in range(len(markers)):
        original_time_stamp = Markers_times_labels[i, 0]
        # array of differences beween the eeg times and the original marker' timestamp
        difference_array = np.absolute(EEG_times-original_time_stamp)
        # find the index of the minimal difference in the markers times stamps (nearest timestamp)
        index = difference_array.argmin()
        # append it to the liste of nearest timestamps
        nearest_time_stamps.append(EEG_times[index])
        nearest_time_stamps_indices.append(index)

    nearest_time_stamps = np.array(nearest_time_stamps)
    nearest_time_stamps_indices = np.array(nearest_time_stamps_indices)
    nearest_indices_timestamps = np.column_stack(
        (nearest_time_stamps_indices, nearest_time_stamps))

    return nearest_indices_timestamps


# function returns a 2d array of EEGtimes_indices associated with the nearest times to the markers timestamps
nearest_marker_indices_timestamps = Nearest_timestamps_array_finder(
    EEG_times, Markers_times_labels)
# =============================================================================

# Function that computes the time lagged PSDS (ex 1 s before each marker)
# returns results as 3d arrays of frequencies and PSDs for each electrode
# array1{Frequencies,dim=[electroden,frequencies,markeri]} array2{Psds,dim=[electroden,PSD,markeri]}


def Compute_lagged_PSD(EEG_data, markers, time_lag=1, direction="before"):
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


# PSD lag:1s before
tridi_freqs_before, tridi_Pxx_densities_before = Compute_lagged_PSD(EEG_Filtered, nearest_marker_indices_timestamps,
                                                                    time_lag=1, direction="before")
# PSD lag:1s after
tridi_freqs_after, tridi_Pxx_densities_after = Compute_lagged_PSD(EEG_Filtered, nearest_marker_indices_timestamps,
                                                                  time_lag=1, direction="after")
# testing
# np.unique(tridi_freqs_after==tridi_freqs_before)
# np.unique(tridi_Pxx_densities_after==tridi_Pxx_densities_before)


# compute the ratio of the Pxx_densities of each side of each marker(12*2) of each of the 8 channel
# need (PSDafter-PSDbefore/PSDbefore)*100
tridi_Pxx_densities_ratio = (
    tridi_Pxx_densities_after-tridi_Pxx_densities_before)/tridi_Pxx_densities_before

# 3d array of frequencies for the ratios
if (tridi_freqs_after == tridi_freqs_before).all() == True:
    tridi_freqs_ratio = tridi_freqs_after
else:
    print("frequencies arrays are not matching")

# 2 blocks of testing each for each arm (ie.Hemisphere)

# 12 markers total, with each block has 6 events of two type(=3rest,3task)

# Separate the columns associated with each type of marker
# 3d array of ratios for each task marker {dims=[electroden,PSDsratios,test_markeri]
# Extract the columns asociated to the tasks markers (111)
tridi_Pxx_densities_ratio_trial_task_markers = tridi_Pxx_densities_ratio[:, :, ::2]
# 3d array of ratios for each rest marker {dims=[electroden,PSDsratios,rest_markeri]
# Extract the columns asociated to the tasks markers (100)
tridi_Pxx_densities_ratio_trial_rest_markers = tridi_Pxx_densities_ratio[:, :, 1::2]

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


# 2d arrays containing for each of the 8 electrodes the PXX average (rows) for the task computed on each block(3 trials)
block1_task, block2_task = Compute_average_ratio_for_event_on_blocks_for_allelectrodes(
    tridi_Pxx_densities_ratio_trial_task_markers)
block1_rest, block2_rest = Compute_average_ratio_for_event_on_blocks_for_allelectrodes(
    tridi_Pxx_densities_ratio_trial_rest_markers)

# testing
# np.mean(tridi_Pxx_densities_ratio_trial_rest_markers[0,0,:]) #ok
# np.mean(tridi_Pxx_densities_ratio_trial_rest_markers[0,1,:]) #ok


# Plotting electrodei's averaged PSD ratio (computed over the 1st block's task 3 trials) over frequencies

Single_plot(fig_number=8, x=tridi_freqs_ratio[1, :, 0], y=block1_task[:, i],
            fig_title="Block 1 task trials' averaged PSD ratio \n Signal EEG Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)]+" marker:"+str(i),
            xlabel="Frequencies(Hz)", ylabel="PSD ratio(after-Before/before) (%)")

# =============================================================================

plt.show()
