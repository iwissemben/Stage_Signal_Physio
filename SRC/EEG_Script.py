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
# library for creating filters
from scipy.signal import butter, iirnotch, filtfilt, welch
from my_functions import *

plt.close("all")  # close all figure windows

# =============================================================================
################################## initialization #############################
# =============================================================================
# Define the xdf file path
filename = "001_MolLud_20201112_1_c.xdf"
# filename="020_DesMar_20211129_1_c.xdf"
# path=os.path.normpath("../DAT/Input/001_MolLud_20201112_1_c.xdf")
path = os.path.normpath("DAT/INPUT/"+filename)


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
############ Centering - Substracting the average of each signal ##############
# =============================================================================
EEG_raw_amplitudes_means = np.mean(EEG_raw_amplitudes, axis=0)
EEG_raw_amplitudes_centered = EEG_raw_amplitudes-EEG_raw_amplitudes_means

# =============================================================================
################################## Re-referencement ###########################
# =============================================================================
# Re-referencing: Uniformly distributed electrodes
EEG_raw_amp_mean = np.mean(EEG_raw_amplitudes_centered)
EEG_raw_rereferenced_amplitudes = EEG_raw_amplitudes - \
    EEG_raw_amp_mean  # rereferencing


# =============================================================================
############################## Event-Markers ##################################
# =============================================================================
# Selection of one electrode and plotting its signal with markers

i = 2  # electrode number
electrodei = EEG_raw_rereferenced_amplitudes[:, i-1]

# plotting electrode i-1's signal for verification
single_plot(filename, fig_number=1, x=EEG_times, y=electrodei,
            fig_title=" Raw EEG Signal Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][i-1]["unit"][0])+")",
            markers_times_array=Markers_times_labels)

# Plotting all of the electrodes' RAW signals in one figure with 4*2=8 graphs with the markers

figure, axis = plt.subplots(2, 4)
mosaic_plot(figure, axis, filename, x=EEG_times, y=EEG_raw_rereferenced_amplitudes,
            fig_title="Raw EEG signals per electrodes", xlabel="Temps (s)",
            ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][1]["unit"][0])+")", channels=channels_dic,
            markers_labels_times=Markers_times_labels)

# =============================================================================
################################## FFT ########################################
# =============================================================================

# compute each channel's RAW FFT
EEG_FFT = compute_fft_on_all_channels(EEG_raw_rereferenced_amplitudes, Srate)

# plotting electrode i-1's raw signal FFT for verification
single_plot(filename, fig_number=3, x=EEG_FFT["fft_frequencies"], y=EEG_FFT["FFT_Results_EEG_channels"][:, i],
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

# creation of a notch filter filtering the 50Hz
# bn,an=iirnotch(w0=50,Q=,fs=Srate)


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
single_plot(filename, fig_number=4, x=EEG_times, y=filtered_signal_electrodei,
            fig_title=" Filtered EEG Signal Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][i-1]["unit"][0])+")",
            markers_times_array=Markers_times_labels, point_style=".k")


# compute each channel's FILTERED signal's FFT
EEG_Filtered_FFT = compute_fft_on_all_channels(EEG_Filtered, Srate)
EEG_Filtered_LFHF_FFT = compute_fft_on_all_channels(EEG_Filtered_LFHF, Srate)

# Plotting the filtered electrode i-1's FFT for verification
single_plot(filename, fig_number=5, x=EEG_Filtered_FFT["fft_frequencies"], y=EEG_Filtered_FFT["FFT_Results_EEG_channels"][:, i],
            fig_title="FFT of filtered Signal EEG Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][i]["unit"][0])+")",
            point_style=".r", line_width=0.5)

# Plotting all of the electrodes' FILTERED signals in one figure with 4*2=8 graphs with the markers

figure, axis = plt.subplots(2, 4)
mosaic_plot(figure, axis, filename, x=EEG_times, y=EEG_Filtered, fig_title="Filtered EEG signals per electrodes", xlabel="Temps(s)",
            ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][1]["unit"][0])+")", channels=channels_dic,
            markers_labels_times=Markers_times_labels)

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
single_plot(filename, fig_number=7, x=freqs, y=Pxx_densities[:, i],
            fig_title="PSD of filtered Signal EEG Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)],
            xlabel="frequency (Hz)", ylabel="PSD Amplitude ("+str(EEG_Stream["info"]["desc"][0]["channel"][i-1]["unit"][0])+"²/Hz)",
            point_style=".g")


# =============================================================================

# function returns a 2d array of EEGtimes_indices associated with the nearest times to the markers timestamps
nearest_marker_indices_timestamps = nearest_timestamps_array_finder(
    EEG_times, Markers_times_labels)
# =============================================================================

# PSD lag:1s before
tridi_freqs_before, tridi_Pxx_densities_before = compute_lagged_psd(EEG_Filtered, Srate, nearest_marker_indices_timestamps,
                                                                    time_lag=1, direction="before")
# PSD lag:1s after
tridi_freqs_after, tridi_Pxx_densities_after = compute_lagged_psd(EEG_Filtered, Srate, nearest_marker_indices_timestamps,
                                                                  time_lag=1, direction="after")

tridi_freqs_before2, tridi_Pxx_densities_before2 = compute_lagged_psd2_all_electrodes(EEG_Filtered, Srate, nearest_marker_indices_timestamps,
                                                                    time_lag=1, direction="before")
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

# 2d arrays containing for each of the 8 electrodes the PXX average (rows) for the task computed on each block(3 trials)
block1_task, block2_task = compute_average_ratio_for_event_on_blocks_for_all_electrodes(
    tridi_Pxx_densities_ratio_trial_task_markers)
block1_rest, block2_rest = compute_average_ratio_for_event_on_blocks_for_all_electrodes(
    tridi_Pxx_densities_ratio_trial_rest_markers)

# testing
# np.mean(tridi_Pxx_densities_ratio_trial_rest_markers[0,0,:]) #ok
# np.mean(tridi_Pxx_densities_ratio_trial_rest_markers[0,1,:]) #ok


# Plotting electrodei's averaged PSD ratio (computed over the 1st block's task 3 trials) over frequencies

single_plot(filename, fig_number=8, x=tridi_freqs_ratio[1, :, 0], y=block1_task[:, i],
            fig_title="Block 1 task trials' averaged PSD ratio \n Signal EEG Derivation " +
            str(i)+": "+channels_dic["Channel_"+str(i)]+" marker:"+str(i),
            xlabel="Frequencies(Hz)", ylabel="PSD ratio(after-Before/before) (%)")

# =============================================================================
# test stacking arrays
testa1 = np.arange(1, 10)
testa2 = np.arange(11, 20)
testa3d = np.dstack((testa1, testa2))

plt.show()
