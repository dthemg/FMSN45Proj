close all
clearvars
clc

fnum = 0;
cf = 50;

load model_A
Am = A;
Cm = C;
S = 24;

load('utempSla_9395.dat')
y = utempSla_9395(:,3);

y(24:24:end) = nan;
y = fillmissing(y,'linear');

startday = 430;
modelweek = 10;
predWeeks = 10;

yM = y(startday*24+1:startday*24+modelweek*7*24);
yM(519) = nan; % taking out the outlier
yM = fillmissing(yM,'linear');

fnum = fnum+1;
figure(fnum)
plot(yM)

yValid = y((startday*24 + modelweek*24*7 + 1): (startday*24 + (modelweek + predWeeks)*24*7));

yFull = [yM; yValid];

fnum = fnum+1;
figure(fnum)
plot(yFull); title('Modelling and validation set')

rm = 50;
yFullold = yFull;
% yFull = filter(A24, 1, yFull);
% yFull(1:rm) = [];

%%
y = yFull;
% yM = mean(y);
% y = y -yM;
N = length(y)
% State space equation definition
ord = 4;
A = eye(ord);
Re = diag([10e-4 10e-4 0 10e-4]); % Hiden state noise covariance matrix
Rw = 25; % Observation variance
% usually C should be set here to, but in this case C is a function of time

% set initial values
Rxx_1 = 10e-5 * eye(ord); % initial variance
xtt_1 = [Am(2:end) -1 Cm(end)]'; % initial state

e = zeros(1,N);
% vector to store values in
xsave = zeros(ord,N);
K = 8; % prediction
yhat = zeros(K,N);

% Kalman filter. Start from k=27, because we need old values of y
for n = 27:N
    % C is, in our case, a function of time
    yt = y(n);
    Ct = [-y(n-1)+y(n-1-S) -y(n-2)+y(n-2-S) -y(n-S) e(n-24)];
    e(n) = yt-Ct*xtt_1;
    
    % Update
    Ryy = Ct*Rxx_1*Ct' + Rw;
    Kt = Rxx_1*Ct'/Ryy;
    xtt = xtt_1+Kt*(yt-Ct*xtt_1);
    Rxx = (eye(ord)-Kt*Ct)*Rxx_1;
    
    % Prediction
    for k = 1:K
        if k == 1
            Ck = [-yFull(n)+yFull(n-1-S+k) -yFull(n-1)+yFull(n-2-S+k) -yFull(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;
        elseif k == 2
            Ck = [-yhat(k-1,n)+yFull(n-1-S+k) -yFull(n)+yFull(n-2-S+k) -yFull(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;
        else
            Ck = [-yhat(k-1,n)+yFull(n-1-S+k) -yhat(k-2,n)+yFull(n-2-S+k) -yFull(n-S+k) e(n-24+k)];
            yhat(k,n) = Ck*xtt_1;            
        end
    end
    % Save
    xsave(:,n) = xtt_1;
    
    % Predict
    Rxx_1 = A*Rxx*A'+Re;
    xtt_1 = A*xtt;  
end

fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(:,:)')
legend('a_1 Kalman','a_2 Kalman','a_{24}','c_{24} Kalman','location','southeast')

resid = e(modelweek*24*7 + 1:end);
fnum = func_plotacfpacf(fnum, resid, cf, 0.05, 'resid');
fnum = fnum +1;
figure(fnum)
whitenessTest(resid)
title('Cumulative periodogram for resid')
fnum = fnum +1;
figure(fnum)
normplot(resid)
title('Norplot for resid')

%% k = 1 prediction
kk = 1;
yhat_1 = yhat(kk,modelweek*24*7 + 1-kk:end-kk);
err_1 = yValid - yhat_1';
err_1_var = var(err_1);
time = 1:length(yValid);
fnum = fnum +1;
figure(fnum)
plot(time,yValid,time,yhat_1)

fnum = func_plotacfpacf(fnum, err_1, cf, 0.05, 'error k=1');
fnum = fnum +1;
figure(fnum)
whitenessTest(err_1)
title('Cumulative periodogram for error k=1')
fnum = fnum +1;
figure(fnum)
normplot(err_1)
title('Normplot for error k=1')

%% k = 8 prediction
kk = 8;
yhat_8 = yhat(kk,modelweek*24*7 + 1-kk:end-kk);
err_8 = yValid - yhat_8';
err_8_var = var(err_8);
time = 1:length(yValid);
fnum = fnum +1;
figure(fnum)
plot(time,yValid,time,yhat_8)

fnum = func_plotacfpacf(fnum, err_8, cf, 0.05, 'error k=8');
fnum = fnum +1;
figure(fnum)
normplot(err_8)
title('Normplot for error k=8')