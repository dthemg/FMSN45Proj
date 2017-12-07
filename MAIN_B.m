close all
clearvars
clc

% Project part 

startWeek = 14;
predWeeks = 5;

fnum = 0;
cf = 50;

load('tvxo93.mat');
load('ptvxo93.mat');
load('tid93');

y = tvxo93;
u = ptvxo93;

plot(u)

% Removing interpolation
y(1) = [];
y = downsample(y, 3);
u(1:2) = [];
u = downsample(u, 3);

% Starting time is week 13 day 5 0:00
startWeek = 14;

yM = y((startWeek - 1)*7*8 + 1:(startWeek + 10 - 1)*7*8);
uM = u((startWeek - 1)*7*8 + 1:(startWeek + 10 - 1)*7*8);

time = (1:length(y))/(24*7);
timeM = 3*(1:length(yM))/(24*7);

fnum = fnum + 1;
figure(fnum)
plot(time, u)
title('SMHI temperature data in Växjö 1993')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

fnum = fnum + 1;
figure(fnum)
plot(timeM, uM)
title('SMHI temperature prediction in Växjö 1993, weeks 14-24')
xlabel('Time [Weeks]')
ylabel('Temperature [^0C]')

% Removing mean values
disp('Test mean of xM')
testMean(uM, 0.05)
muM = mean(uM);
uM = uM - muM;

disp('Test mean of yM')
testMean(yM, 0.05)
myM = mean(yM);
yM = yM - myM;

% Normplot of xM suggests some transformation is required
fnum = fnum + 1;
figure(fnum)
normplot(uM)
title('Normality of SMHI prediction')

%% PREWHITENING x: Finding B(z) and A2(z) and d (d intuitively 0)

% ACF, PACF suggest deseasoning
fnum = func_plotacfpacf(fnum, uM, cf, 0.05, 'SMHI input');

A8 = [1, zeros(1,7) -1];

% Save both x, y before deseasoning
uNonD = uM;
yNonD = yM;

rm = 30;
uM = filter(A8, 1, uM);
uM(1:rm) = [];

fnum = func_plotacfpacf(fnum, uM, cf, 0.05, 'after deseasoning');

data_uM = iddata(uM);

%% Look for outliers
fnum = fnum + 1;
figure(fnum)
plot(timeM(rm+1:end), uM)
title('Look for outliers in deseasoned data')

% Check data using truncated acf
fnum = fnum + 1;
figure(fnum)
tacf(uM, cf, 0.02, 0.05, true, 0);
title('truncated ACF of data') % No significant difference was detected, so regular acf is used

%% AR(1)
ar_model = arx(data_uM, 1);

res = resid(ar_model, data_uM);
fnum = func_plotacfpacf(fnum, res.y, cf, 0.05, 'residuals ar(1)');

%% ARMA(1,8)
model_init = idpoly([1, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
arma_model = pem(data_uM, model_init);

res3 = resid(arma_model, data_uM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8)');

fnum = fnum + 1;
figure(fnum)
whitenessTest(res3.y, 0.05)
title('Whiteness test for arma(1,8)')

best_model = arma_model;

%% ARMA(8,8)
model_init = idpoly([1, zeros(1,8)], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
model_init.Structure.a.Free = [0, 1, zeros(1,6), 1];

arma_model = pem(data_uM, model_init);

res3 = resid(arma_model, data_uM);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(8,8)');

fnum = fnum + 1;
figure(fnum)
whitenessTest(res3.y, 0.05)
title('Whiteness test for arma(8,8)')

% Residuals for arma(1,8) are decent, passing one of the whiteness tests
% Create x_pw and y_pw from this model

A_star = conv(best_model.a, A8);
uM_pw = filter(A_star, best_model.c, uNonD);
uM_pw(1:rm) = [];

yM_pw = filter(A_star, best_model.c, yM);
yM_pw(1:rm) = [];

%% Checking crosscorrelation function between prewhitened x and y

fnum = func_plotccf(fnum, uM_pw, yM_pw, cf, 'prewhitened u and y');

% d = 0, r = 0, s = 1
%% Estimating B and A2
A2 = [1]; % r
B = [0, 0]; % d, s
Mi = idpoly(1,B,[],[],A2);
zpw = iddata(yM_pw,uM_pw);
Mba2 = pem(zpw,Mi); present(Mba2);
vhat = resid(Mba2,zpw);

fnum = func_plotccf(fnum, uM_pw, vhat.y, cf, 'd=0, r=0, s=1 between u_{pw} and v');

% d = 0, r = 2, s = 0
%% Estimating B and A2
d = 0; r = 2; s = 0;
A2 = [1, zeros(1, r)]; % r
B = [0, zeros(1, d), zeros(1, s)]; % d, s

Mi = idpoly(1,B,[],[],A2);
Mi.Structure.b.Free = [1, zeros(1, d), ones(1,s)]; % d, s

zpw = iddata(yM_pw,uM_pw);
Mba2 = pem(zpw,Mi); present(Mba2);
vhat = resid(Mba2,zpw);

fnum = func_plotccf(fnum, uM_pw, vhat.y, cf, ['d=', num2str(d), ' r=', num2str(r), ...
                                            ' s=', num2str(s), ' between u_{pw} and v']);

% This selection of (r,s,d) turned out to show the strongest results

% ACF/PACF shows non-white results, not too concerning
fnum = func_plotacfpacf(fnum, vhat.y, cf, 0.05, 'v');

% Deseason y
yM = filter(A8, 1, yM);
yM(1:rm) = [];

h = filter(Mba2.b, Mba2.f, uM);
x = yM(51:end) - h(51:end);

%% Estimating orders for C1(z) A1(z)

fnum = func_plotccf(fnum, uNonD(51:end), x, cf, 'between u and x');

% Determine suitable model orders for C1, A1
fnum = func_plotacfpacf(fnum, x, cf, 0.05, 'x');

data_x = iddata(x);

%% AR(1)
ar_model = arx(data_x, 1);

res = resid(ar_model, data_x);
fnum = func_plotacfpacf(fnum, res.y, cf, 0.05, 'residuals ar(1), x');

%% AR(1,8)
model_init = idpoly([1, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
arma_model = pem(data_x, model_init);

res3 = resid(arma_model, data_x);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(1,8)');

fnum = fnum + 1;
figure(fnum)
whitenessTest(res3.y, 0.05)
title('Whiteness test for arma(1,8)')

% Not white, but try estimating entire model

%% AR(2,8)
model_init = idpoly([1, 0, 0], [], [1, zeros(1,8)]);
model_init.Structure.c.Free = [zeros(1,8), 1];
arma_model = pem(data_x, model_init);

res3 = resid(arma_model, data_x);
fnum = func_plotacfpacf(fnum, res3.y, cf, 0.05, 'residuals arma(2,8)');

fnum = fnum + 1;
figure(fnum)
whitenessTest(res3.y, 0.05)
title('Whiteness test for arma(2,8)')

% Not white, but try estimating entire model

%% Final model - (d,r,s) = (0,2,0), C1 - 8, A1 - 1, A2 - 2

A1 = [1 0]; 
A2 = [1 0 0];
B = [1];
C = [1 zeros(1,8)];
Mi = idpoly(1,B,C,A1,A2);

Mi.Structure.c.Free = [zeros(1, 8), 1];
z = iddata(yM,uM);
MboxJ = pem(z,Mi);

present(MboxJ);
ehat = resid(MboxJ,z);

% ACF/PACF show non-white residuals, but almost white
disp('%%%%%%%% Final model - (d,r,s) = (0,2,0), C1 - 8, A1 - 1, A2 - 2 %%%%%%%%%%')
whitenessTest(ehat.y, 0.05)
fnum = func_plotacfpacf(fnum, ehat.y, cf, 0.05, 'residuals final model');

%% Final model - (d,r,s) = (0,2,0), C1 - 8, A1 - 2, A2 - 2

A1 = [1 0, 0]; 
A2 = [1 0 0];
B = [1];
C = [1 zeros(1,8)];
Mi = idpoly(1,B,C,A1,A2);

Mi.Structure.c.Free = [zeros(1, 8), 1];
z = iddata(yM,uM);
MboxJ = pem(z,Mi);

present(MboxJ);
ehat = resid(MboxJ,z);

% ACF/PACF show white residuals, passes 4 out of 5 tests
disp('%%%%%%%% Final model - (d,r,s) = (0,2,0), C1 - 8, A1 - 2, A2 - 2 %%%%%%%%%%')
whitenessTest(ehat.y, 0.05)
fnum = func_plotacfpacf(fnum, ehat.y, cf, 0.05, 'residuals final model');

%% Predictions
C = MboxJ.c;
A1 = MboxJ.d;
A2 = MboxJ.f;
B = MboxJ.b;

A = conv(A1, A2);
A = conv(A, A8);
C = conv(C, A2);
B = conv(B, A1);
B = conv(B, A8);

%% k = 1
k = 1;

yValid = y(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);
uValid = u(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(A,C,k);
BF = conv(B,F);
[Fhat,Ghat] = func_poldiv(C,BF,k);

yhat = filter(Ghat,C,uValid) + filter(G,C,yValid) + filter(Fhat,1,uValid);
yhat = yhat(k+1:end);
yValid = yValid(k+1:end);

timeV = 3*(1:length(yhat))/(24*7);

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
uValid = u(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(A,C,k);
BF = conv(B,F);
[Fhat,Ghat] = func_poldiv(C,BF,k);

yhat = filter(Ghat,C,uValid) + filter(G,C,yValid) + filter(Fhat,1,uValid);
yhat = yhat(k+1:end);
yValid = yValid(k+1:end);

timeV = 3*(1:length(yhat))/(24*7);

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

%% k = 8
k = 8;

yValid = y(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);
uValid = u(((startWeek + 10 - 1)*7*8) + 1 - k: (startWeek + 10 + predWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(A,C,k);
BF = conv(B,F);
[Fhat,Ghat] = func_poldiv(C,BF,k);

yhat = filter(Ghat,C,uValid) + filter(G,C,yValid) + filter(Fhat,1,uValid);
yhat = yhat(k+1:end);
yValid = yValid(k+1:end);

timeV = 3*(1:length(yhat))/(24*7);

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

%% Save

save('Model_B', 'A', 'B', 'C');






