function [psd_fft,psd_p,psd_w]=estimate_signal_psds(signal,Fs)
    N = length(signal);%pour avoir le nombre d'échantillons du signal
    f = 0:(Fs/N):Fs/2;%échelle des fréquences en respectant la limite de Shannon Fs/2
    disp(f)
    %Calcul du périodogramme (estimateur de la DSP) "de base et à la main" via la fonction fft
    Xfft = fft(signal);%calcul de la transformée de Fourier rapide
    PSD = (abs(Xfft).^2)/(N*Fs);%calcul du périodogramme à partir du module au carré de la fft
    
    %Calcul du périodogramme avec la fonction Matlab directement, pour
    %comparaison
    [PSDm,fm] = periodogram(signal,rectwin(N),f,Fs);%version two-sided (cf. help) qui correspond
    % à la même échelle que la version précédente
    
    %Calcul du périodogramme de Welch avec un fenêtrage de Hamming, des tailles
    %de fenêtres de 1000 points (1 seconde) et un overlap de 500 points
    [PSDw,fw] = pwelch(signal,hamming(1000),500,N,Fs);
    
    psd_fft=[f',PSD(1:length(f))];
    psd_p=[fm',PSDm'];
    psd_w=[fw,PSDw];
end
