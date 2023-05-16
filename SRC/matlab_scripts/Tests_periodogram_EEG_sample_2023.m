clear
close

x = load('EEG_1channel_1000Hz.txt');%pour charger le signal expérimental
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