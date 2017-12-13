
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

startday = 430;
modelweek = 10;

yM = y(startday*24+1:startday*24+modelweek*7*24);
yM(519) = nan; % taking out the outlier
yM = fillmissing(yM,'linear');


figure
plot(yM)

time = (1:length(y))/(24*7);
timeM = (1:length(yM))/(24*7);

fnum = fnum + 1;
figure(fnum)
plot(time, y)
title('Temperature data in Svedala 1993-1995')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

fnum = fnum + 1;
figure(fnum)
plot(timeM, yM)
title('Temperature data in Svedala 1993, starting day 430')
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


% Subtract mean
disp('testMean result before mean substraction and deseasoning')
testMean(yM, 0, 0.05)
myM = mean(yM);
yM = yM - myM;

fnum = func_plotacfpacf(fnum, yM, cf, 0.05, 'before deseasoning');

% Shows strong seasonality, desason by 24
% move to ARMA process NOT!
A24 = [1, zeros(1,23) -1];


rm = 50;
yM = filter(A24, 1, yM);
yM(1:rm) = [];
% yOrig(1:rm) = [];

fnum = fnum + 1;
figure(fnum)
plot(timeM((rm+1):end), yM)
% plot(timeM, yM)
title('temperature, after deseasoning')
xlabel('Time [Weeks]')

fnum = func_plotacfpacf(fnum, yM, cf, 0.05, 'after deseasoning');

% fnum = fnum+1;
% figure(fnum)
% tacf(yM,cf,0.05,0.05,true,0)

% testMean() gives 0, meaning we cannot say if different from 0
disp('testMean result after deseasoning')
testMean(yM, 0, 0.05)

% After deseasonalizing data appears t-distributed
fnum = fnum + 1;
figure(fnum)
normplot(yM);
title('Normplot after desasonalizing')

data_yM = iddata(yM);

%% AR(24)
model_init = idpoly([1, zeros(1,24)], [], []);
model_init.Structure.a.Free = [zeros(1,24), 1];
ar_model = pem(data_yM, model_init);

res3 = resid(ar_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals ar(24)');

%% AR(24)
model_init = idpoly([1, zeros(1,25)], [], []);
model_init.Structure.a.Free = [0, 1, 1, zeros(1,21), 1, 1];
ar_model = pem(data_yM, model_init);

res3 = resid(ar_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals ar(24) with a_1 a_2');
%% AR(1)
ar_model2 = arx(data_yM, 1);

res1 = resid(ar_model2, data_yM);

fnum = func_plotacfpacf(fnum, res1.y, cf, 0.05, 'residuals ar1');

%% AR(2)
ar_model2 = arx(data_yM, 2);

res1 = resid(ar_model2, data_yM);

fnum = func_plotacfpacf(fnum, res1.y, cf, 0.05, 'residuals ar2');
%% ARMA(1,1)
model_init = idpoly([1, 0], [], [1, 0]);
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,1)');
%% ARMA(1,24)
model_init = idpoly([1, 0], [], [1, zeros(1,24)]);
model_init.Structure.c.Free = [0, 1, zeros(1,22), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,24) a_1 c_1 c_{24}');
present(arma_model)
%% ARMA(2,24)
model_init = idpoly([1, 0, 0], [], [1, zeros(1,24)]);
model_init.Structure.c.Free = [0, 1, zeros(1,22), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,24) a_1 a_2 c_1 c_{24}');
present(arma_model)
%% ARMA(24,24) with a1, a24, c1, c24
model_init = idpoly([1, zeros(1,24)], [], [1, zeros(1,24)]);
model_init.Structure.c.Free = [0, 1, zeros(1,22), 1];
model_init.Structure.a.Free = [0, 1, zeros(1,22), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(24,24) with a_1 a_{24} c_1 c_{24}');
present(arma_model)
%% ARMA(24,24)
model_init = idpoly([1, zeros(1,24)], [], [1, zeros(1,25)]);
model_init.Structure.c.Free = [zeros(1,23), 1, 1, 1];
model_init.Structure.a.Free = [0, 1, 1, zeros(1,21), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(24,24)');

%% ARMA(1,24)- c1
model_init = idpoly([1, 0], [], [1, zeros(1,24)]);
model_init.Structure.c.Free = [0, 1, zeros(1,22), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,24) with c_1');

%% ARMA(2,24)- a1, a2, c24
model_init = idpoly([1, 0, 0], [], [1, zeros(1,24)]);
model_init.Structure.c.Free = [zeros(1,24), 1];
arma_model = pem(data_yM, model_init);

res3 = resid(arma_model, data_yM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(2,24) a_1 a_2 c_{24}');
present(arma_model)

fnum = fnum +1;
figure(fnum)
whitenessTest(res3.y)
title('Cumulative periodogram for a_1, a_2, c_{24}')
best_model = arma_model;

%% Prediction
A = best_model.a;
C = best_model.c;

A_star = conv(A, A24);
%% k = 1
k = 1;
startday = 430;
modelweek = 10;
predWeeks = 10;

yValid = y((startday*24 + modelweek*24*7 + 1 - k): (startday*24 + (modelweek + predWeeks)*24*7));

[F,G] = func_poldiv(A_star,C,k);
yhat = filter(G,C,yValid);
yhat(1:k) = [];
yValid(1:k) = [];

timeV = (1:length(yValid))/(24*7);

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
acf(err1step, cf, 0.05, true, 0, 0);
title(['Residuals prediction k=', num2str(k)])

fnum = fnum + 1;
figure(fnum)
disp(['Whiteness test for k=' num2str(k)])
whitenessTest(err1step)
title(['Cumulative periodogram for k=' num2str(k)])

%% k = 8
k = 8;
filtermax = 24;
z = max(k,filtermax);

yValid = y((startday*24 + modelweek*24*7 + 1 - k): (startday*24 + (modelweek + predWeeks)*24*7));

fnum = fnum+1;
figure(fnum)
plot(yValid)

[F,G] = func_poldiv(A_star,C,k);
yhat = filter(G,C,yValid);
yhat(1:z) = [];
yValid(1:z) = [];

timeV = (1:length(yValid))/(24*7);

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
acf(err8step, cf, 0.05, true, 0, 0);
title(['Residuals prediction k=', num2str(k)])


%% Save polynomials for A, C

err1step_A_var = err1step_var;
err8step_A_var = err8step_var;

save('Model_A', 'best_model')
save('variances_valid_A', 'err1step_A_var', 'err8step_A_var')






