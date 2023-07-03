clear
close


% 1 - Import eeg recording in csv (raw xdf exported with export_signals.ipynb)
    %1.1 - Import csv file
        FILENAME='../../DAT/INPUT/001_MolLud_20201112_1_c_499.998_Hz.csv';
        csv_data = readmatrix(FILENAME);
        %table =readtable('../../DAT/INPUT/001_MolLud_20201112_1_c_499.998Hz.csv');
    %1.2 - tidy data
        EEG_raw_amplitudes = csv_data(:,1:8);
        EEG_times = csv_data(:,9);
    %1.3 - data infos
        N = length(EEG_raw_amplitudes); %number of samples per electrode
        t=split(FILENAME,"_")
        Fs = round(str2double(t{end-1})) %sampling frequency (Hz)

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

% 3 - PSD estimations

psd_estimations(6,6)

    

% 4 - Export results
    % 4.1 - Select electrodes
    % 4.2 - Export to csv file


%{
x = load('../../DAT/INPUT/EEG_1channel_1000Hz.txt');%pour charger le signal expérimental
Fs = 1000;%fréquence d'échantillonnage
Ps = 1/Fs;%période d'échantillonnage
xc = x-mean(x);%pour centrer le signal
N = length(xc);%pour avoir le nombre d'échantillons du signal

t = 0:Ps:(N-1)*Ps;%échelle de temps physique

f = 0:(Fs/N):Fs/2;%échelle des fréquences en respectant la limite de Shannon Fs/2

%Calcul du périodogramme (estimateur de la DSP) "de base et à la main" via la fonction fft
Xfft = fft(xc);%calcul de la transformée de Fourier rapide
PSD = (abs(Xfft).^2)/(N*Fs);%calcul du périodogramme à partir du module au carré de la fft

%Calcul du périodogramme avec la fonction Matlab directement, pour
%comparaison
[PSDm,fm] = periodogram(xc,rectwin(N),f,Fs);%version two-sided (cf. help) qui correspond
% à la même échelle que la version précédente

%Calcul du périodogramme de Welch avec un fenêtrage de Hamming, des tailles
%de fenêtres de 1000 points (1 seconde) et un overlap de 500 points
[PSDw,fw] = pwelch(xc,hamming(1000),500,N,Fs);

%Exportation des resultats de PSD
matricePSDpw_titres=["f_fft","PSDfft","fm","PSDm","fw","PSDw"]
matricePSDfftpw=horzcat(f',PSD(1:length(f)),fm',PSDm',fw,PSDw)

table_complete= array2table(matricePSDfftpw,'VariableNames',matricePSDpw_titres)

writetable(table_complete,'../../DAT/OUTPUT/MATLAB_EEG_PSDs_data_1000Hz.csv',Delimiter=";",WriteMode="overwrite")
%dlmwrite('../../DAT/OUTPUT/MATLAB_EEG_PSDs_data_1000Hz.csv', matricePSDfftpw, 'delimiter', ';', 'precision', 100);

figure
subplot(4,1,1)
plot(t,xc,'-b')
xlim([0 t(end)])
xlabel('Temps (s)')
ylabel('Signal EEG')
grid on
subplot(4,1,2)
plot(f,PSD(1:length(f))','-b','LineWidth',1)%on se limite à une échelle de fréquence limitée à Fs/2
xlabel('Fréquence (Hz)')
ylabel('DSP par fft')
grid on
subplot(4,1,3)
plot(fm,PSDm,'g')
xlabel('Fréquence (Hz)')
ylabel('DSP periodogram')
grid on
subplot(4,1,4)
plot(fw,PSDw,'r')
xlabel('Fréquence (Hz)')
ylabel('DSP pwelch')
grid on
%}
    function output= psd_estimations_on_a_signal(signal,Fs)
    %{
    z=x+y;
    disp("the result is: ")
    disp(z)
    output=z;
    %}

    Ps = 1/Fs;%période d'échantillonnage
    N = length(xc);%pour avoir le nombre d'échantillons du signal
    
    t = 0:Ps:(N-1)*Ps;%échelle de temps physique
    
    f = 0:(Fs/N):Fs/2;%échelle des fréquences en respectant la limite de Shannon Fs/2
    
    %Calcul du périodogramme (estimateur de la DSP) "de base et à la main" via la fonction fft
    Xfft = fft(xc);%calcul de la transformée de Fourier rapide
    PSD = (abs(Xfft).^2)/(N*Fs);%calcul du périodogramme à partir du module au carré de la fft
    
    %Calcul du périodogramme avec la fonction Matlab directement, pour
    %comparaison
    [PSDm,fm] = periodogram(xc,rectwin(N),f,Fs);%version two-sided (cf. help) qui correspond
    % à la même échelle que la version précédente
    
    %Calcul du périodogramme de Welch avec un fenêtrage de Hamming, des tailles
    %de fenêtres de 1000 points (1 seconde) et un overlap de 500 points
    [PSDw,fw] = pwelch(xc,hamming(1000),500,N,Fs);
    
end