clear
close


% 1 - Import eeg recording in csv (raw xdf exported with export_signals.ipynb)
    % 1.1 - Import csv file
        %input_filepath='../../DAT/INPUT/001_MolLud_20201112_1_c_raw_499.998_Hz.csv';
        input_filepath='../../DAT/INPUT/001_MolLud_20201112_1_c_prepro_499.998_Hz.csv';
        csv_data = readmatrix(input_filepath);
        %table =readtable('../../DAT/INPUT/001_MolLud_20201112_1_c_499.998Hz.csv');
    % 1.2 - tidy data
        numb_of_channels = size(csv_data,2)-1; %find the number of channels, last column is for times vector
        EEG_raw_amplitudes = csv_data(:,1:numb_of_channels);
        EEG_times = csv_data(:,end);
    % 1.3 - data infos
        N = length(EEG_raw_amplitudes); %number of samples per electrode
        %get sampling frequency from file name
        t=split(input_filepath,"_");
        Fs = round(str2double(t{end-1})); %sampling frequency (Hz)
        disp(Fs);
    % 1.5- Define channel names
        channels_dict = ["Channel_1_C4","Channel_2_FC2","Channel_3_FC6","Channel_4_CP2","Channel_5_C3","Channel_6_FC1","Channel_7_FC5","Channel_8_CP1"];

    % 1.4 - Select electrode to study
        select_channel_number = [1,5]; %1d matrix of selected channels
        if all(select_channel_number <= numb_of_channels)
            fprintf("Selected channel: %g\n",select_channel_number);
            disp("Selected channels found.");
        else
            error("one or more of the selected channel does not exist.");
        end

%loop over channels
for element=select_channel_number % loop over each number
    %disp(element); %show the channel number
    fprintf("Processing channel: %s \n",channels_dict(element)); %find its corresponding name
    signal=EEG_raw_amplitudes(:,element); %get the corresponding signal

% 3 - PSD estimations
        [PSD_fft,PSD_p,PSD_w]=estimate_signal_psds(signal,Fs); %compute the psd estimations over the selected signal

% 4 - export the PSD estimations of the signal
    % 4.1 - Define export file name & path
        [~, input_filename, ~] = fileparts(input_filepath); %get input file name from filepath
        export_filename="MATLAB_PSD_res_EEG_"+channels_dict(element)+"_"+input_filename; %define output filename
        export_filepath="../../DAT/OUTPUT/Matlab_PSD_Results/"+export_filename;
    % 4.2 - Export PSD results to csv
        results=[PSD_fft,PSD_p,PSD_w];
        header=["f_fft","PSDfft","fm","PSDm","fw","PSDw"];
        table_to_export=export_psds_csv(export_filepath,results,header);

% 5 - Show PSD estimations results
    show_psd_results(signal,Fs,table_to_export,channels_dict(element));
    fprintf("Processing %s : done \n",channels_dict(element)); %find its corresponding name
end

%{
% 2 - Signal preprocessing
    % 2.1 - Centering
        % compute average of each column of EEG_raw_amplitudes
        column_averages= mean(EEG_raw_amplitudes); 
    
        % replicate the line to match EEG_raw_amplitudes dimensions
        number_of_rows = size(EEG_raw_amplitudes, 1);
        column_averages=repmat(column_averages,number_of_rows,1);
    
        % centering - substract each sample of each electrode by the electrode average
        EEG_centered_amplitudes = EEG_raw_amplitudes - column_averages;

    % 2.2 - Rereferencing to average
        whole_average=mean(EEG_centered_amplitudes,"all")
        EEG_rereferenced_amplitudes = EEG_centered_amplitudes - whole_average;
    % 2.3 - Filtering
%}






