function [fnum] = func_plotacfpacf(fnum, x, cutoff, alpha, tit)
    fnum = fnum + 1;
    figure(fnum)
    
    subplot(211)
    acf(x, cutoff, alpha, true, 0, 0);
    title(['ACF for ', tit])
    subplot(212)
    pacf(x, cutoff, alpha, true, 0);
    title(['PACF for ', tit])
end