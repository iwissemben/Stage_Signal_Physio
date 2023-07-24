clear
close

% 1 - load the file
input_filepath='../../DAT/INPUT/EEG_1channel_1000_Hz.txt';
x = load(input_filepath);%pour charger le signal expérimental
xc = x-mean(x);%pour centrer le signal
t=split(input_filepath,"_");
Fs = round(str2double(t{end-1})); %sampling frequency (Hz)
disp(Fs);
channel_number=0;

% 2 - compute the PSD estimations on the signal
    %2.1 - 
[u,v,w]=estimate_signal_psds(xc,Fs);

% 3 - export the PSD estimations of the signal
[~, input_filename, ~] = fileparts(input_filepath); %get input filename from filepath
export_filename="MATLAB_PSD_res_EEG_chan_"+channel_number+"_"+input_filename; %define output filename
export_filepath="../../DAT/OUTPUT/Matlab_PSD_Results/"+export_filename;

results=[u,v,w];
header=["f_fft","PSDfft","fm","PSDm","fw","PSDw"];
table_to_export=export_psds_csv(export_filepath,results,header);


% 4 - plot the PSD results alongside the time signal
show_psd_results(xc,Fs,table_to_export);



