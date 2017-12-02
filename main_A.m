
clearvars
clc
close all

fnum = 0;
cf = 50;

% Project part A

load('tvxo93.mat');
load('tid93');

y = tvxo93;
startWeek = 12;

yM = y((startWeek - 1)*7*24 + 1:(startWeek + 10 - 1)*7*24);

% Downsample by 3 to remove interpolation
yM(1) = [];
yM = downsample(yM, 3);

time = (1:length(y))/(24*7);
timeM = 3*(1:length(yM))/(24*7);

fnum = fnum + 1;
figure(fnum)
plot(time, y)
title('Temperature data in Växjö 1993')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

fnum = fnum + 1;
figure(fnum)
plot(timeM, yM)
title('Temperature data in Växjö 1993, weeks 12-22')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

% Normplot suggests some transformation is required
fnum = fnum + 1;
figure(fnum)
title('Normality of data')
normplot(yM)

% Transformation, check Box Jenkins
fnum = fnum + 1;
figure(fnum)
bcNormPlot(yM) % Suggests square root transformation

yM = yM + 10; % Deal with negative values?

yM = sqrt(yM);

fnum = fnum + 1;
figure(fnum)
plot(timeM, yM)
title('temperature, after transformation')
xlabel('Time [Weeks]')
ylabel('square root temperature')

fnum = func_plotacfpacf(fnum, yM, cf, 0.05, 'after transformation');

% Shows strong seasonality, desason by 24
A8 = [1, zeros(1,7) -1];

rm = 30;
yM = filter(A8, 1, yM);
yM(1:rm) = [];

% After deseasonalizing data appears t-distributed
fnum = fnum + 1;
figure(fnum)
normplot(yM);
title('Normplot after desasonalizing')

% Subtract mean
testMean(yM, 0, 0.05)
myM = mean(yM);
yM = yM - myM;

fnum = fnum + 1;
figure(fnum)
plot(timeM(rm+1:end), yM)
title('root temperature, after transformation')
xlabel('Time [Weeks]')
ylabel('root Temperature')

fnum = func_plotacfpacf(fnum, yM, cf, 0.05, 'deseasonalized');

data_yM = iddata(yM);

%% AR(1)
ar_model2 = arx(data_yM, 1);
present(ar_model2)

res1 = resid(ar_model2, data_yM);

fnum = func_plotacfpacf(fnum, res1.y, cf, 0.05, 'residuals ar1');


%% ARMA(1,8)
model_init = idpoly([1, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
arma_model = pem(data_yM, model_init);

present(arma_model)
res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8)');

%% ARMA(1,8) c1
model_init = idpoly([1, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [0, 1, zeros(1,6), 1];
arma_model = pem(data_yM, model_init);

present(arma_model)
res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8) with c_1');










