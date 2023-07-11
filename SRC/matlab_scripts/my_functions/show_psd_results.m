function show_psd_results(signal,Fs,results_table,titleText)
    Ps = 1/Fs;%période d'échantillonnage
    N = length(signal);%pour avoir le nombre d'échantillons du signal
    t = 0:Ps:(N-1)*Ps;%échelle de temps physique
    if nargin < 4 || isempty(titleText)
        titleText = 'Default Title';  % Default value for titleText
    end
    %show time signal
    figure
    title(titleText)
    subplot(4,1,1)
    plot(t,signal,'-b')
    xlim([0 t(end)])
    xlabel('Temps (s)')
    ylabel('Signal EEG')
    grid on
    title(titleText)

    %show PSD from FFT
    subplot(4,1,2)
    plot(results_table{:,1},results_table{:,2},'-b','LineWidth',1)%on se limite à une échelle de fréquence limitée à Fs/2
    xlabel('Fréquence (Hz)')
    ylabel('DSP par fft')
    grid on

    %show PSD from periodogram
    subplot(4,1,3)
    plot(results_table{:,3},results_table{:,4},'g')
    xlabel('Fréquence (Hz)')
    ylabel('DSP periodogram')
    grid on

    %show PSD from welch
    subplot(4,1,4)
    plot(results_table{:,5},results_table{:,6},'r')
    xlabel('Fréquence (Hz)')
    ylabel('DSP pwelch')
    grid on

end