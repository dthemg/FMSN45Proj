
clearvars
clc
close all

% Project part A

fnum = 0;
cf = 50;

load('utempSla_9395.dat')
y = utempSla_9395(:,3);

y(24:24:end) = nan;
y = fillmissing(y,'linear');

figure
plot(y)



load('tvxo93.mat');
load('tid93');

y = tvxo93;
% Starting time is week 13 day 5 0:00

startWeek = 14;
predWeeks = 5;
totWeeks = startWeek + predWeeks;

y(1) = [];

% Remove interpolated data
y = downsample(y, 3);

yM = y((startWeek - 1)*7*8 + 1:(startWeek + 10 - 1)*7*8);
yOrig = yM;

time = (1:length(y))/(8*7);
timeM = 3*(1:length(yM))/(8*7);

fnum = fnum + 1;
figure(fnum)
plot(time, y)
title('Temperature data in Växjö 1993')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

fnum = fnum + 1;
figure(fnum)
plot(timeM, yM)
title('Temperature data in Växjö 1993, weeks 14-24')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

% Normplot suggests some transformation is required
fnum = fnum + 1;
figure(fnum)
normplot(yM)
title('Normality of data')

% Transformation, check Box Jenkins
fnum = fnum + 1;
figure(fnum)
bcNormPlot(yM) % Suggests square root transformation, but data suggests keep yM
grid on

% yM = sqrt(yM);

% Subtract mean
disp('testMean result before mean substraction and deseasoning')
testMean(yM, 0, 0.05)
myM = mean(yM);
yM = yM - myM;

% Shows strong seasonality, desason by 24
A8 = [1, zeros(1,7) -1];

rm = 30;
yM = filter(A8, 1, yM);
yM(1:rm) = [];
% yOrig(1:rm) = [];

fnum = fnum + 1;
figure(fnum)
plot(timeM((rm+1):end), yM)
title('temperature, after deseasoning')
xlabel('Time [Weeks]')

fnum = func_plotacfpacf(fnum, yM, cf, 0.05, 'after deseasoning');

% testMean() gives 0, meaning we cannot say if different from 0
disp('testMean result after deseasoning')
testMean(yM, 0.05)

% After deseasonalizing data appears t-distributed
fnum = fnum + 1;
figure(fnum)
normplot(yM);
title('Normplot after desasonalizing')

data_yM = iddata(yM);

%% AR(1)
ar_model2 = arx(data_yM, 1);

res1 = resid(ar_model2, data_yM);

fnum = func_plotacfpacf(fnum, res1.y, cf, 0.05, 'residuals ar1');


%% ARMA(8, 0)
model_init = idpoly([1, zeros(1,8)], [], []);
model_init.Structure.a.Free = [0, 1, zeros(1,6), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(8,0), a_1, a_8');

present(arma_model)

%% ARMA(1,8)
model_init = idpoly([1, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8)');

%% ARMA(1,8)- c1
model_init = idpoly([1, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [0, 1, zeros(1,6), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8) with c_1');

%% ARMA(8,0)- a1, a8
model_init = idpoly([1, zeros(1, 8)], [], []);
model_init.Structure.a.Free = [0, 1, zeros(1,6), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(8,0) with a_1, a_8');

%% ARMA(8,8)- a1, a8, c8
model_init = idpoly([1, zeros(1, 8)], [], [1, zeros(1, 8)]);
model_init.Structure.a.Free = [0, 1, zeros(1,6), 1];
model_init.Structure.c.Free = [0, zeros(1,7), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(8,8) with a_1, a_8, c_8');

%% ARMA(8,8)- a1, a8, c1, c8
model_init = idpoly([1, zeros(1, 8)], [], [1, zeros(1, 8)]);
model_init.Structure.a.Free = [0, 1, zeros(1,6), 1];
model_init.Structure.c.Free = [0, 1, zeros(1,6), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(8,8) with a_1, a_8, c_1, c_8');

present(arma_model)
figure;
whitenessTest(res3.y, 0.05)

% Passes two of the whiteness tests but a8 component is insignificant

%% ARMA(1,8)- a1, c1, c8
model_init = idpoly([1,0], [], [1, zeros(1, 8)]);
model_init.Structure.a.Free = [0, 1];
model_init.Structure.c.Free = [0, 1, zeros(1,6), 1];
arma_model = pem(data_yM, model_init);

present(arma_model)
res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8) with a_1, c_1, c_8');

fnum = fnum + 1;
figure(fnum)
whitenessTest(res3.y, 0.05)
title('Cumulative periodogram for a_1, c_1, c_8')

% This model is close to being white, proceed with prediction

%% Prediction
A = arma_model.a;
C = arma_model.c;

A_star = conv(A, A8);
%% k = 1
k = 1;

yValid = y(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(A_star,C,k);
yhat = filter(G,C,yValid);
yhat(1:k) = [];
yValid(1:k) = [];

timeV = 3*(1:length(yValid))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeV, yValid, timeV, yhat)
title([num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')

err1step = yValid - yhat;
err1step_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0)
title(['Residuals prediction k=', num2str(k)])

%% k = 3
k = 3;

yValid = y(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(A_star,C,k);
yhat = filter(G,C,yValid);
yhat(1:k) = [];
yValid(1:k) = [];

timeV = 3*(1:length(yValid))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeV, yValid, timeV, yhat)
title([num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')

err3step = yValid - yhat;
err3step_var = var(err3step);

fnum = fnum + 1;
figure(fnum)
acf(err3step, cf, 0.05, true, 0, 0)
title(['Residuals prediction k=', num2str(k)])

%% k = 8 - One full day
k = 8;

yValid = y(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(A_star,C,k);
yhat = filter(G,C,yValid);
yhat(1:k) = [];
yValid(1:k) = [];

timeV = 3*(1:length(yValid))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeV, yValid, timeV, yhat)
title([num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')

err8step = yValid - yhat;
err8step_var = var(err8step);

fnum = fnum + 1;
figure(fnum)
acf(err8step, cf, 0.05, true, 0, 0)
title(['Residuals prediction k=', num2str(k)])

%% Save polynomials for A, C

err1step_A_var = err1step_var;
err3step_A_var = err3step_var;
err8step_A_var = err8step_var;

save('Model_A', 'C', 'A_star')
save('variances_A', 'err1step_A_var', 'err3step_A_var', 'err8step_A_var')





