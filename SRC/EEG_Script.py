# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:02:00 2022

@author: iWiss
"""

# xdf file importation
import os
import pyxdf
import matplotlib.pyplot as plt
import numpy as np

# library for creating filters
from scipy.signal import butter, iirnotch, filtfilt, welch
from my_functions import *

plt.close("all")  # close all figure windows

# =============================================================================
################################## initialization #############################
# =============================================================================
# Define the xdf file path
FILENAME = "001_MolLud_20201112_1_c.xdf"
# FILENAME = "020_DesMar_20211129_1_c.xdf"
# path=os.path.normpath("../DAT/Input/001_MolLud_20201112_1_c.xdf")
path = os.path.normpath("./DAT/INPUT/"+FILENAME)


# Load only streams of interest (EEG signal and Mouse task Markers) from the xdf data file
# data, header = pyxdf.load_xdf(path, select_streams=
# [{'type': 'EEG', 'name': 'LSLOutletStreamName-EEG'},{'type': 'Markers', 'name': 'MouseToNIC'}] )
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
# Selection of one electrode

ELECTRODE_NUMBER = 2  # [1,8]
ELECTRODE_INDEX = ELECTRODE_NUMBER-1  # python indices start a 0

# plotting electrode i's signal for verification
single_plot(FILENAME, fig_number=1, x=EEG_times, y=EEG_raw_rereferenced_amplitudes[:, ELECTRODE_INDEX],
            fig_title=" Raw EEG Signal Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            markers_times_array=Markers_times_labels)

# Plotting all of the electrodes' RAW signals in one figure with 4*2=8 graphs with the markers

figure, axis = plt.subplots(2, 4)
mosaic_plot(figure, axis, FILENAME, x=EEG_times, y=EEG_raw_rereferenced_amplitudes,
            fig_title="Raw EEG signals per electrodes", xlabel="Temps (s)",
            ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][1]["unit"][0])+")", channels=channels_dic,
            markers_labels_times=Markers_times_labels)

# =============================================================================
################################## FFT ########################################
# =============================================================================

# compute each channel's RAW FFT
EEG_FFT = compute_fft_on_all_channels(EEG_raw_rereferenced_amplitudes, Srate)

# plotting electrode i's raw signal FFT for verification
single_plot(FILENAME, fig_number=3, x=EEG_FFT["fft_frequencies"], y=EEG_FFT["FFT_Results_EEG_channels"][:, ELECTRODE_INDEX],
            fig_title="FFT of raw Signal EEG Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            point_style="-r", line_width=0.5)

# =============================================================================
################################## Filtering ##################################
# =============================================================================

FILTER_ORDER = 4
LOW_CUTOFF_FREQ_THEORETICAL = 1
HIGH_CUTOFF_FREQ_THEORETICAL = 40


# correct the frequencies
LOW_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
    FILTER_ORDER, LOW_CUTOFF_FREQ_THEORETICAL, Srate, pass_type="high_pass")

HIGH_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
    FILTER_ORDER, HIGH_CUTOFF_FREQ_THEORETICAL, Srate, pass_type="low_pass")

print("LOW_CUTOFF_FREQ_THEORETICAL="+str(LOW_CUTOFF_FREQ_THEORETICAL))
print("HIGH_CUTOFF_FREQ_THEORETICAL="+str(HIGH_CUTOFF_FREQ_THEORETICAL))
print("LOW_CUTOFF_FREQ_CORRECTED="+str(LOW_CUTOFF_FREQ_CORRECTED))
print("HIGH_CUTOFF_FREQ_CORRECTED="+str(HIGH_CUTOFF_FREQ_CORRECTED))

# Filters creation

# Notch-filter 50Hz
# bn,an=iirnotch(w0=50,Q=,fs=Srate)

# butterworth filters 4th order (to compare BP vs LP+HP)
F_Nyquist = Srate/2

# creation of butterworth filters
# b, a = butter(4,[(0.3)/F_Nyquist,50/F_Nyquist],btype='bandpass') #band-pass with direct frequencies
# b, a = butter(4,[(0.3),50],btype='bandpass',fs=Srate)    #Band-pass with direct frequencies

# Band-pass with rectified frequencies
bBP, aBP = butter(FILTER_ORDER, [
                  LOW_CUTOFF_FREQ_CORRECTED, HIGH_CUTOFF_FREQ_CORRECTED], btype='bandpass', fs=Srate)
# Low-pass  with rectified frequencies
bLP, aLP = butter(FILTER_ORDER, HIGH_CUTOFF_FREQ_CORRECTED,
                  btype='lowpass', fs=Srate)
# High-pass with rectified frequencies
bHP, aHP = butter(FILTER_ORDER, LOW_CUTOFF_FREQ_CORRECTED,
                  btype='highpass', fs=Srate)

# Combine the low-pass and high-pass filters
b_band = np.convolve(bLP, bHP)
a_band = np.convolve(aLP, aHP)

# Filtering on all channels (2 methods to compare)
# on one hand Band-pass Filtering
EEG_Filtered = filtfilt(bBP, aBP, EEG_raw_rereferenced_amplitudes, axis=0)
# on the other High-Pass + Low-Pass
# using the combined filter
EEG_Filtered_LFHF = filtfilt(
    b_band, a_band, EEG_raw_rereferenced_amplitudes, axis=0)

# test if BP filtering is same as LP+BP
"""test1 = np.unique(EEG_Filtered == EEG_Filtered_LFHF)
test2 = np.unique(np.rint(EEG_Filtered) == np.rint(EEG_Filtered_LFHF))
EEG_Filtered_int = np.rint(EEG_Filtered)
EEG_Filtered_LFHF_int = np.rint(EEG_Filtered_LFHF)

print("Is BP exactly the same as LP+HP? : ", test1)
print("Is BP approximately the same as LP+HP? : ", test2)"""

# Plotting the filtered  electrode i's signal for verification
single_plot(FILENAME, fig_number=4, x=EEG_times, y=EEG_Filtered[:, ELECTRODE_INDEX],
            fig_title=" Filtered EEG Signal Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            markers_times_array=Markers_times_labels, point_style=".k")


# compute each channel's FILTERED signal's FFT for each filtering method (to compare)
EEG_Filtered_FFT = compute_fft_on_all_channels(EEG_Filtered, Srate)
EEG_Filtered_LFHF_FFT = compute_fft_on_all_channels(EEG_Filtered_LFHF, Srate)

# Plotting the filtered electrode i's FFT for verification
single_plot(FILENAME, fig_number=5, x=EEG_Filtered_FFT["fft_frequencies"], y=EEG_Filtered_FFT["FFT_Results_EEG_channels"][:, ELECTRODE_INDEX],
            fig_title="FFT of filtered Signal EEG Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            point_style=".r", line_width=0.5)

# Plotting all of the electrodes' FILTERED signals in one figure with 4*2=8 graphs with the markers

figure, axis = plt.subplots(2, 4)
mosaic_plot(figure, axis, FILENAME, x=EEG_times, y=EEG_Filtered, fig_title="Filtered EEG signals per electrodes", xlabel="Temps(s)",
            ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][1]["unit"][0])+")", channels=channels_dic,
            markers_labels_times=Markers_times_labels)

# =============================================================================
############################# Periodogram #####################################
# =============================================================================

FREQ_RES = 0.5  # Desired Frequency resolution in Hz (0.5)
nbpoints_per_epoch = Srate/FREQ_RES  # as FREQ_RES=Sampling_freq/Nbpoints
Acq_time = len(EEG_Filtered)/Srate

# PSD estimation for each electrodes
# freqs,Pxx_density=welch(EEG_Filtered[:,i],fs=Srate,window="hann",nperseg=nbpoints_per_epoch,noverlap=nbpoints_per_epoch//2)

freqs, Pxx_densities = welch(EEG_Filtered, fs=Srate, window="hann",
                             nperseg=nbpoints_per_epoch, noverlap=nbpoints_per_epoch//2, axis=0)


# Plotting electrodei's PSD over entire signal
single_plot(FILENAME, fig_number=7, x=freqs, y=Pxx_densities[:, ELECTRODE_INDEX],
            fig_title="PSD of filtered EEG signal derivation " +
            str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" + str(ELECTRODE_NUMBER)] +
            "\n (over whole signal)",
            xlabel="frequency (Hz)", ylabel="PSD Amplitude ("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+"²/Hz)",
            point_style=".g")


# =============================================================================

# function returns a 2d array of EEGtimes_indices associated with the nearest times to the markers timestamps
nearest_marker_indices_timestamps = nearest_timestamps_array_finder(
    EEG_times, Markers_times_labels)
# =============================================================================

# time window over which the PSD is computed expressed in seconds
time_window = 4
# PSD lag:1s before
tridi_freqs_before, tridi_Pxx_densities_before = compute_lagged_psd2_all_electrodes(EEG_Filtered, Srate, nearest_marker_indices_timestamps,
                                                                                    time_lag=time_window, direction="before")
# PSD lag:1s after
tridi_freqs_after, tridi_Pxx_densities_after = compute_lagged_psd2_all_electrodes(EEG_Filtered, Srate, nearest_marker_indices_timestamps,
                                                                                  time_lag=time_window, direction="after")

# separation of the markers of each 3d array (before and after)
tridi_Pxx_densities_111_before = tridi_Pxx_densities_before[:, ::2, :]
tridi_Pxx_densities_111_after = tridi_Pxx_densities_after[:, ::2, :]

tridi_Pxx_densities_100_before = tridi_Pxx_densities_before[:, 1::2, :]
tridi_Pxx_densities_100_after = tridi_Pxx_densities_after[:, 1::2, :]

# compute the ratio of the Pxx_densities of each side of each marker(12*2) of each of the 8 channel
# need (PSDafter-PSDbefore/PSDbefore)*100
tridi_Pxx_densities_ratio_111 = ((
    tridi_Pxx_densities_111_after-tridi_Pxx_densities_111_before)/tridi_Pxx_densities_111_before)*100
tridi_Pxx_densities_ratio_100 = ((
    tridi_Pxx_densities_100_before-tridi_Pxx_densities_111_after)/tridi_Pxx_densities_111_after)*100

# testing
verite = np.unique(tridi_freqs_after == tridi_freqs_before)
print(verite)

# np.unique(tridi_Pxx_densities_after==tridi_Pxx_densities_before)
# 3d array of frequencies for the ratios
if verite == True:
    tridi_freqs_ratio = tridi_freqs_after
else:
    print("frequencies arrays are not matching")
# 2 blocks of testing each for each arm (ie.Hemisphere)
# compute the average of the ratios for each block (2*(3*111),2*(3*110))
tridi_Pxx_densities_ratio_111_mean_block1 = np.mean(
    tridi_Pxx_densities_ratio_111[:, 0:3, :], axis=1)
tridi_Pxx_densities_ratio_111_mean_block2 = np.mean(
    tridi_Pxx_densities_ratio_111[:, 3:6, :], axis=1)

# Plotting electrodei's PSD before first marker 111
single_plot(FILENAME, fig_number=8, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_111_before[:, 1, ELECTRODE_INDEX],
            fig_title="PSD of filtered EEG signal derivation " +
            str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" + str(ELECTRODE_NUMBER)] +
            "\n before first marker 111 (over "+str(time_window)+"s)",
            xlabel="frequency (Hz)", ylabel="PSD Amplitude ("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+"²/Hz)",
            point_style=".g")


# Plotting electrodei's PSD after first marker 111
single_plot(FILENAME, fig_number=9, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_111_after[:, 1, ELECTRODE_INDEX],
            fig_title="PSD of filtered EEG signal derivation " +
            str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" + str(ELECTRODE_NUMBER)] +
            "\n after first marker 111 (over "+str(time_window)+"s)",
            xlabel="frequency (Hz)", ylabel="PSD Amplitude ("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+"²/Hz)", point_style=".g")


# Plotting electrodei's PSD ratio for first marker 111
single_plot(FILENAME, fig_number=10, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_ratio_111[:, 1, ELECTRODE_INDEX],

            fig_title="PSD ratio of filtered EEG signal derivation " +
            str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" + str(ELECTRODE_NUMBER)] +
            "\n for first marker 111 ("+str(time_window)+"s)",
            xlabel="Frequencies(Hz)", ylabel="PSD ratio(after-Before/before) (%)")

# Plotting electrodei's averaged PSD ratio (computed over 3 trials of the 1st block's task)
single_plot(FILENAME, fig_number=11, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_ratio_111_mean_block1[:, ELECTRODE_INDEX],
            fig_title="PSD ratio of filtered EEG signal derivation " + str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" +
                                                                                                               str(ELECTRODE_NUMBER)] + "\n Averaged on first 3 markers 111 ("+str(time_window)+"s)",
            xlabel="Frequencies(Hz)", ylabel="PSD ratio(after-Before/before) (%)")

# =============================================================================
plt.show()
