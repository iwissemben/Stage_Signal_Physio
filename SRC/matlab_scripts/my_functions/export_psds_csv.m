function table_to_export=export_psds_csv(filename,tuple,header)
    %Exportation des resultats de PSD
    %matricePSDpw_titres=["f_fft","PSDfft","fm","PSDm","fw","PSDw"]
    %matricePSDfftpw=horzcat(freqs1,psd1,freqs2,PSDm',fw,PSDw)
    
    matricePSDfftpw=horzcat(tuple);
    table_to_export= array2table(matricePSDfftpw,'VariableNames',header);
    writetable(table_to_export,filename+".csv",Delimiter=";",WriteMode="overwrite");
end