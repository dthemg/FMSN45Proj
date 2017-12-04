close all
clearvars
clc

% Project part 

% Model from part A
old_model = load('ourModel.mat');

fnum = 0;
cf = 50;

load('tvxo93.mat');
load('ptvxo93.mat');
load('tid93');

y = tvxo93;
x = ptvxo93;
% Starting time is week 13 day 5 0:00
startWeek = 14;

yM = y((startWeek - 1)*7*24 + 1:(startWeek + 10 - 1)*7*24);
xM = x((startWeek - 1)*7*24 + 1:(startWeek + 10 - 1)*7*24);

yM(1) = [];
xM(1:2) = [];

% Removing interpolation
yM = downsample(yM, 3);
xM = downsample(xM, 3);

time = (1:length(y))/(24*7);
timeM = 3*(1:length(yM))/(24*7);

fnum = fnum + 1;
figure(fnum)
plot(time, x)
title('SMHI temperature data in Växjö 1993')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

fnum = fnum + 1;
figure(fnum)
plot(timeM, xM)
title('SMHI temperature prediction in Växjö 1993, weeks 14-24')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

% Removing mean values
disp('Test mean of xM')
testMean(xM, 0.05)
mxM = mean(xM);
xM = xM - mxM;

disp('Test mean of yM')
testMean(yM, 0.05)
myM = mean(yM);
yM = yM - myM;

% Normplot of xM suggests some transformation is required
fnum = fnum + 1;
figure(fnum)
normplot(xM)
title('Normality of SMHI prediction')

%% PREWHITENING: Finding B(z) and A2(z) and d (d intuitively 0)

% ACF, PACF suggest deseasoning
fnum = func_plotacfpacf(fnum, xM, cf, 0.05, 'SMHI input');

A8 = [1, zeros(1,7) -1];

rm = 30;
xM = filter(A8, 1, xM);
xM(1:rm) = [];

fnum = func_plotacfpacf(fnum, xM, cf, 0.05, 'after deseasoning');

data_xM = iddata(xM);

%% AR(1)
ar_model = arx(data_xM, 1);

res = resid(ar_model, data_xM);
fnum = func_plotacfpacf(fnum, res.y, cf, 0.05, 'residuals ar(1)');

%% ARMA(1,8)
model_init = idpoly([1, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
arma_model = pem(data_xM, model_init);

res3 = resid(arma_model, data_xM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8)');

fnum = fnum + 1;
figure(fnum)
whitenessTest(res3.y, 0.05)
title('Whiteness test for arma(1,8)')

%% ARMA(8,8)
model_init = idpoly([1, zeros(1,8)], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
model_init.Structure.a.Free = [0, 1, zeros(1,6), 1];

arma_model = pem(data_xM, model_init);

res3 = resid(arma_model, data_xM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(8,8)');

fnum = fnum + 1;
figure(fnum)
whitenessTest(res3.y, 0.05)
title('Whiteness test for arma(8,8)')




