
% Loading data
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

% Comparing models from part A, B and C
AModel = load('Model_A.mat');
BModel = load('Model_B.mat');

% Starting 20 weeks from last model datapoint
startWeek = 14 + 10 + 19;
testWeeks = 10;

%% Model A prediction

%% k = 1
k = 1;

yTest = y(((startWeek - 1)*7*8) + 1 - k: (startWeek + testWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(AModel.A_star, AModel.C, k);
yhatA = filter(G, AModel.C, yTest);
yhatA(1:k) = [];
yTest(1:k) = [];

timeV = 3*(1:length(yTest))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeV, yTest, timeV, yhatA)
title([num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')

err1step = yTest - yhatA;
err1step_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0)
title(['Residuals prediction k=', num2str(k)])

%% k = 3
k = 3;

yTest = y(((startWeek - 1)*7*8) + 1 - k: (startWeek + testWeeks - 1)*7*8 + 1);

[F,G] = func_poldiv(AModel.A_star, AModel.C, k);
yhatA = filter(G, AModel.C, yTest);
yhatA(1:k) = [];
yTest(1:k) = [];

timeV = 3*(1:length(yTest))/(24*7);

fnum = fnum+1;
figure(fnum)
plot(timeV, yTest, timeV, yhatA)
title([num2str(k), '-step prediction'])
xlabel('Weeks')
ylabel('Temperature')

err1step = yTest - yhatA;
err1step_var = var(err1step);

fnum = fnum + 1;
figure(fnum)
acf(err1step, cf, 0.05, true, 0, 0)
title(['Residuals prediction k=', num2str(k)])





