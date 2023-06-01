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
from scipy.signal import welch

# my libraries
from my_functions import *
from my_filters import *

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
############################ Electrode selection ##############################
# =============================================================================
ELECTRODE_NUMBER = 1  # [1,8]
ELECTRODE_INDEX = ELECTRODE_NUMBER-1  # python indices start a 0

# plotting electrode i's raw time signal for verification
single_plot(FILENAME, fig_number=1, x=EEG_times, y=EEG_raw_amplitudes[:, ELECTRODE_INDEX],
            fig_title="Raw time signal EEG Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            markers_times_array=Markers_times_labels)

EEG_raw_FFT = compute_fft_on_all_channels(EEG_raw_amplitudes, Srate)

# plotting electrode i's raw signal FFT for verification
single_plot(FILENAME, fig_number=2, x=EEG_raw_FFT["fft_frequencies"], y=EEG_raw_FFT["FFT_Results_EEG_channels"][:, ELECTRODE_INDEX],
            fig_title="FFT of raw signal EEG Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            point_style="-r", line_width=0.5)

# =============================================================================
#################### Effect of raw signal on all electrodes ###################
# =============================================================================

# Plotting all of the electrodes' RAW time signals in one figure with 4*2=8 graphs with the markers
figure, axis = plt.subplots(2, 4, num=3)
mosaic_plot(figure, axis, FILENAME, x=EEG_times, y=EEG_raw_amplitudes,
            fig_title="Raw EEG time signals per electrodes", xlabel="Temps (s)",
            ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][1]["unit"][0])+")", channels=channels_dic,
            markers_labels_times=Markers_times_labels)

# =============================================================================
############ Centering - Substracting the average of each signal ##############
# =============================================================================
EEG_raw_amplitudes_means = np.mean(EEG_raw_amplitudes, axis=0)
EEG_raw_amplitudes_centered = EEG_raw_amplitudes-EEG_raw_amplitudes_means

# =============================================================================
############################### Re-referencement ##############################
# =============================================================================
# Re-referencing: Uniformly distributed electrodes
EEG_raw_amp_centered_mean = np.mean(EEG_raw_amplitudes_centered)
EEG_raw_rereferenced_amplitudes = EEG_raw_amplitudes_centered - \
    EEG_raw_amp_centered_mean

# plotting electrode i's centered, rereferenced time signal for verification
single_plot(FILENAME, fig_number=4, x=EEG_times, y=EEG_raw_rereferenced_amplitudes[:, ELECTRODE_INDEX],
            fig_title=" Centered rereferenced EEG Signal Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            markers_times_array=Markers_times_labels)

# compute each channel's centered, rereferenced FFT
EEG_FFT = compute_fft_on_all_channels(EEG_raw_rereferenced_amplitudes, Srate)

# plotting electrode i's  signal centered, rereferenced FFT for verification
single_plot(FILENAME, fig_number=5, x=EEG_FFT["fft_frequencies"], y=EEG_FFT["FFT_Results_EEG_channels"][:, ELECTRODE_INDEX],
            fig_title="FFT of Centered rereferenced signal EEG Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            point_style="-r", line_width=0.5)

# =============================================================================
############ Effect of centering and referencing on all electrodes ############
# =============================================================================
# Plotting all of the electrodes' centered rereferenced time signals in one figure with 4*2=8 graphs with the markers
figure, axis = plt.subplots(2, 4, num=6)
mosaic_plot(figure, axis, FILENAME, x=EEG_times, y=EEG_raw_rereferenced_amplitudes,
            fig_title="Centered rereferenced EEG time signals per electrodes", xlabel="Temps (s)",
            ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][1]["unit"][0])+")", channels=channels_dic,
            markers_labels_times=Markers_times_labels)

# =============================================================================
################################## Filtering ##################################
# =============================================================================

FILTER_ORDER = 8
LOW_CUTOFF_FREQ_THEORETICAL = 5
HIGH_CUTOFF_FREQ_THEORETICAL = 100
NOTCH_CUTOFF_FREQ = 50

# cutoff frequency correction for filtfilt application
LOW_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
    FILTER_ORDER, LOW_CUTOFF_FREQ_THEORETICAL, Srate, pass_type="high_pass")

HIGH_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
    FILTER_ORDER, HIGH_CUTOFF_FREQ_THEORETICAL, Srate, pass_type="low_pass")

print("LOW_CUTOFF_FREQ_THEORETICAL="+str(LOW_CUTOFF_FREQ_THEORETICAL) +
      ", HIGH_CUTOFF_FREQ_THEORETICAL="+str(HIGH_CUTOFF_FREQ_THEORETICAL))
print("LOW_CUTOFF_FREQ_CORRECTED="+str(LOW_CUTOFF_FREQ_CORRECTED) +
      ", HIGH_CUTOFF_FREQ_CORRECTED="+str(HIGH_CUTOFF_FREQ_CORRECTED))

# Filtering on all channels

# 1-Notch-filter 50Hz [49,51] the centered-rereferenced signal
EEG_Filtered_NOTCH, freqs_test_NOTCH, magnitudes_test_NOTCH = notch_filter(input_signal=EEG_raw_rereferenced_amplitudes,
                                                                           sample_rate=Srate,
                                                                           cutoff_freq=NOTCH_CUTOFF_FREQ,
                                                                           stop_band_width=2)

# 2-Then Band-pass filter the signal filtered by notch
EEG_Filtered_NOTCH_BP, freq_test_BP, magnitude_test_BP = band_pass_filter(input_signal=EEG_Filtered_NOTCH,
                                                                          sample_rate=Srate,
                                                                          low_cutoff_freq=LOW_CUTOFF_FREQ_CORRECTED,
                                                                          high_cutoff_freq=HIGH_CUTOFF_FREQ_CORRECTED,
                                                                          filter_order=FILTER_ORDER)

""" Alternatives BP

# on one hand Band-pass Filtering with rectified frequencies
EEG_Filtered_BP, freq_test_BP, magnitude_test_BP = band_pass_filter(input_signal=EEG_raw_rereferenced_amplitudes,
                                                                    sample_rate=Srate,
                                                                    low_cutoff_freq=LOW_CUTOFF_FREQ_CORRECTED,
                                                                    high_cutoff_freq=HIGH_CUTOFF_FREQ_CORRECTED,
                                                                    filter_order=FILTER_ORDER)

# on the other custom band pass (High-Pass + Low-Pass)
EEG_Filtered_LFHF, freq_test_LFHF, magnitude_test_LFHF = custom_band_pass_filter(input_signal=EEG_raw_rereferenced_amplitudes,
                                                                                 sample_rate=Srate,
                                                                                 low_cutoff_freq=LOW_CUTOFF_FREQ_CORRECTED,
                                                                                 high_cutoff_freq=HIGH_CUTOFF_FREQ_CORRECTED,
                                                                                 filter_order=FILTER_ORDER)

# test if BP filtering is same as LP+BP
test1 = np.unique(EEG_Filtered_BP == EEG_Filtered_LFHF)
test2 = np.unique(np.rint(EEG_Filtered_BP) == np.rint(EEG_Filtered_LFHF))

print("Is BP exactly the same as LP+HP? : ", test1)
print("Is BP approximately the same as LP+HP? : ", test2)

EEG_Filtered_LFHF_FFT = compute_fft_on_all_channels(EEG_Filtered_LFHF, Srate)

"""

# plotting electrode i's Filtered (Notch+BP) time signal for verification
single_plot(FILENAME, fig_number=7, x=EEG_times, y=EEG_Filtered_NOTCH_BP[:, ELECTRODE_INDEX],
            fig_title=" Filtered (Notch + BP) time signal EEG Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Temps (s)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            markers_times_array=Markers_times_labels)

# compute each channel's filtered signal's FFT
EEG_Filtered_FFT_NOTCH = compute_fft_on_all_channels(EEG_Filtered_NOTCH, Srate)
EEG_Filtered_FFT_NOTCH_BP = compute_fft_on_all_channels(
    EEG_Filtered_NOTCH_BP, Srate)

# plotting electrode i's  signal centered, rereferenced FFT for verification
single_plot(FILENAME, fig_number=8, x=EEG_Filtered_FFT_NOTCH_BP["fft_frequencies"], y=EEG_Filtered_FFT_NOTCH_BP["FFT_Results_EEG_channels"][:, ELECTRODE_INDEX],
            fig_title="FFT of Filtered (Notch + BP) EEG Signal Derivation " +
            str(ELECTRODE_NUMBER)+": " +
            channels_dic["Channel_"+str(ELECTRODE_NUMBER)],
            xlabel="Frequency(Hz)", ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+")",
            point_style="-r", line_width=0.5)

# =============================================================================
#################### Effect of filtering on all electrodes ####################
# =============================================================================
# Plotting all of the electrodes' FILTERED time-signals in one figure with 4*2=8 graphs with the markers

figure, axis = plt.subplots(2, 4, num=9)
mosaic_plot(figure, axis, FILENAME, x=EEG_times, y=EEG_Filtered_NOTCH_BP, fig_title="Filtered (Notch+ BP) EEG time signals per electrodes", xlabel="Temps(s)",
            ylabel="Amplitude("+str(EEG_Stream["info"]["desc"][0]["channel"][1]["unit"][0])+")", channels=channels_dic,
            markers_labels_times=Markers_times_labels)
# =============================================================================
############################# Periodogram #####################################
# =============================================================================

FREQ_RES = 0.5  # Desired Frequency resolution in Hz (0.5)
nbpoints_per_epoch = Srate/FREQ_RES  # as FREQ_RES=Sampling_freq/Nbpoints
Acq_time = len(EEG_Filtered_NOTCH_BP)/Srate

# PSD estimation for each electrodes
# freqs,Pxx_density=welch(EEG_Filtered[:,i],fs=Srate,window="hann",nperseg=nbpoints_per_epoch,noverlap=nbpoints_per_epoch//2)

freqs, Pxx_densities = welch(EEG_Filtered_NOTCH_BP, fs=Srate, window="hann",
                             nperseg=nbpoints_per_epoch, noverlap=nbpoints_per_epoch//2, axis=0)

# Plotting electrodei's PSD over entire signal
single_plot(FILENAME, fig_number=10, x=freqs, y=Pxx_densities[:, ELECTRODE_INDEX],
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
tridi_freqs_before, tridi_Pxx_densities_before = compute_lagged_psd2_all_electrodes(EEG_Filtered_NOTCH_BP, Srate, nearest_marker_indices_timestamps,
                                                                                    time_lag=time_window, direction="before")
# PSD lag:1s after
tridi_freqs_after, tridi_Pxx_densities_after = compute_lagged_psd2_all_electrodes(EEG_Filtered_NOTCH_BP, Srate, nearest_marker_indices_timestamps,
                                                                                  time_lag=time_window, direction="after")

# separation of the markers of each 3d array (before and after)
tridi_Pxx_densities_111_before = tridi_Pxx_densities_before[:, ::2, :]
tridi_Pxx_densities_111_after = tridi_Pxx_densities_after[:, ::2, :]

tridi_Pxx_densities_100_before = tridi_Pxx_densities_before[:, 1::2, :]
tridi_Pxx_densities_100_after = tridi_Pxx_densities_after[:, 1::2, :]

# compute the ratio of the Pxx_densities of each side of each marker(12*2) of each of the 8 channel
# need (PSDafter-PSDbefore/PSDbefore)*100
tridi_Pxx_densities_ratio_111 = (
    tridi_Pxx_densities_111_after-tridi_Pxx_densities_111_before)
tridi_Pxx_densities_ratio_100 = (
    tridi_Pxx_densities_100_before-tridi_Pxx_densities_100_after)

"""tridi_Pxx_densities_ratio_111 = ((
    tridi_Pxx_densities_111_after-tridi_Pxx_densities_111_before)/tridi_Pxx_densities_111_before)*100
tridi_Pxx_densities_ratio_100 = ((
    tridi_Pxx_densities_100_before-tridi_Pxx_densities_111_after)/tridi_Pxx_densities_111_after)*100"""


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
single_plot(FILENAME, fig_number=11, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_111_before[:, 1, ELECTRODE_INDEX],
            fig_title="PSD of filtered EEG signal derivation " +
            str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" + str(ELECTRODE_NUMBER)] +
            "\n before first marker 111 (over "+str(time_window)+"s)",
            xlabel="frequency (Hz)", ylabel="PSD Amplitude ("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+"²/Hz)",
            point_style=".g")


# Plotting electrodei's PSD after first marker 111
single_plot(FILENAME, fig_number=12, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_111_after[:, 1, ELECTRODE_INDEX],
            fig_title="PSD of filtered EEG signal derivation " +
            str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" + str(ELECTRODE_NUMBER)] +
            "\n after first marker 111 (over "+str(time_window)+"s)",
            xlabel="frequency (Hz)", ylabel="PSD Amplitude ("+str(EEG_Stream["info"]["desc"][0]["channel"][ELECTRODE_INDEX]["unit"][0])+"²/Hz)", point_style=".g")


# Plotting electrodei's PSD ratio for first marker 111
single_plot(FILENAME, fig_number=13, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_ratio_111[:, 1, ELECTRODE_INDEX],

            fig_title="PSD ratio of filtered EEG signal derivation " +
            str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" + str(ELECTRODE_NUMBER)] +
            "\n for first marker 111 ("+str(time_window)+"s)",
            xlabel="Frequencies(Hz)", ylabel="PSD ratio(after-Before/before) (%)")

# Plotting electrodei's averaged PSD ratio (computed over 3 trials of the 1st block's task)
single_plot(FILENAME, fig_number=14, x=tridi_freqs_ratio[:, 0, 0], y=tridi_Pxx_densities_ratio_111_mean_block1[:, ELECTRODE_INDEX],
            fig_title="PSD ratio of filtered EEG signal derivation " + str(ELECTRODE_NUMBER)+": "+channels_dic["Channel_" +
                                                                                                               str(ELECTRODE_NUMBER)] + "\n Averaged on first 3 markers 111 ("+str(time_window)+"s)",
            xlabel="Frequencies(Hz)", ylabel="PSD ratio(after-Before/before) (%)")

plot_filter_frequency_response(fig_number=50,
                               fig_title="Notch filter frequency response",
                               frequencies=freqs_test_NOTCH,
                               magnitude=magnitudes_test_NOTCH,
                               order="none",
                               fc1=50)

plot_filter_frequency_response(fig_number=51,
                               fig_title="Band Pass filter frequency response",
                               frequencies=freq_test_BP,
                               magnitude=magnitude_test_BP,
                               order=FILTER_ORDER,
                               fc1=LOW_CUTOFF_FREQ_CORRECTED,
                               fc2=HIGH_CUTOFF_FREQ_CORRECTED)

# =============================================================================
plt.show()
