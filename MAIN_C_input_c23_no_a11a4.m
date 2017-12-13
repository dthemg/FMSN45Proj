close all
clearvars
clc

fnum = 0;
cf = 50;
K = 8; % prediction

load model_B
Am = A;
Cm = C;
Bm = B;
S = 24;

load('ptstu94.mat') % Input
load('utempSla_9395.dat')
y = utempSla_9395(:,3);
u = ptstu94; 

y(24:24:end) = nan;
y = fillmissing(y,'linear');

startday = 430;
modelweek = 10;
predWeeks = 10;
hrsInYear = 24 * 365;

yM = y(startday*24+1:startday*24+modelweek*7*24);
uM = u((startday*24+1 - hrsInYear):(startday*24+modelweek*7*24 - hrsInYear));
yM(519) = nan; % taking out the outlier
yM = fillmissing(yM,'linear');

fnum = fnum+1;
figure(fnum)
plot(yM)

yValid = y((startday*24 + modelweek*24*7 + 1): (startday*24 + (modelweek + predWeeks)*24*7));
uValid = u(((startday*24 + modelweek*7*24) + 1 - hrsInYear): ((startday*24+(modelweek + predWeeks)*7*24) + 1 + K - hrsInYear));
yFull = [yM; yValid];
uFull = [uM; uValid];

fnum = fnum+1;
figure(fnum)
plot(yFull); title('Modelling and validation set')
hold on
plot(uFull)

rm = 50;
yFullold = yFull;

% State space equation definition
ord = 12;
A = eye(ord);
Re = diag(10e-4*ones(1,ord)); % Hiden state noise covariance matrix
Re(4,4) = 0; %set seasoning to constant
Rw = 25; % Observation variance
% usually C should be set here to, but in this case C is a function of time

% set initial values
Rxx_1 = 10e-5 * eye(ord); % initial variance
xtt_1 = [Am(2:4) -1 Cm(2) 0 Cm(25:26) Bm(1:4)]'; % initial state

y = yFull;
u = uFull;

N = length(y);
yhat = zeros(K,N);
e = zeros(1,N+K);
% vector to store values in
xsave = zeros(ord,N);

% Kalman filter. Start from k=27, because we need old values of y
for n = 36:N
    % C is, in our case, a function of time
    yt = y(n);
    ut = u(n);
    Ct = [-y(n-1)+y(n-1-S), -y(n-2)+y(n-2-S), -y(n-3)+y(n-3-S), -y(n-S),...
            e(n-1), e(n-23), e(n-24), e(n-25),...
            u(n)-u(n-S), u(n-1)-u(n-1-S), u(n-2)-u(n-2-S), u(n-3)-u(n-3-S)];
    e(n) = yt-Ct*xtt_1;
    
    % Update
    Ryy = Ct*Rxx_1*Ct' + Rw;
    Kt = Rxx_1*Ct'/Ryy;
    xtt = xtt_1+Kt*(yt-Ct*xtt_1);
    Rxx = (eye(ord)-Kt*Ct)*Rxx_1;
    
    
    % Prediction
    for k = 1:K
        if k == 1
            Ck = [-y(n)+y(n-1-S+k), -y(n-1)+y(n-2-S+k), -y(n-2)+y(n-3-S+k), -y(n-S+k),...
                e(n-1+k), e(n-23+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k)];
            yhat(k,n) = Ck*xtt_1;
        elseif k == 2
            Ck = [-yhat(k-1,n)+y(n-1-S+k), -y(n)+y(n-2-S+k), -y(n-1)+y(n-3-S+k), -y(n-S+k),...
                e(n-1+k), e(n-23+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k)];
            yhat(k,n) = Ck*xtt_1;
        elseif k == 3
            Ck = [-yhat(k-1,n)+y(n-1-S+k), -yhat(k-2,n)+y(n-2-S+k), -y(n)+y(n-3-S+k), -y(n-S+k),...
                e(n-1+k), e(n-23+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k)];
            yhat(k,n) = Ck*xtt_1;            
        else
            Ck = [-yhat(k-1,n)+y(n-1-S+k), -yhat(k-2,n)+yFull(n-2-S+k), -yhat(k-2,n)+yFull(n-2-S+k), -y(n-S+k),...
                e(n-1+k), e(n-23+k), e(n-24+k), e(n-25+k),...
                u(n+k)-u(n-S+k), u(n-1+k)-u(n-1-S+k), u(n-2+k)-u(n-2-S+k), u(n-3+k)-u(n-3-S+k)];
            yhat(k,n) = Ck*xtt_1;            
        end
    end
    
    % Save
    xsave(:,n) = xtt_1;
    
    % Predict
    Rxx_1 = A*Rxx*A'+Re;
    xtt_1 = A*xtt;  
end

%% Plot parameter estimates
fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(1:4,:)')
title('a parameters')
legend('a_1 Kalman','a_2 Kalman','a_3 Kalman','a_{24}','location','southeast')

fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(5:8,:)')
title('c parameters')
legend('c_1 Kalman','c_{23} Kalman','c_{24} Kalman','c_{25} Kalman','location','southeast')

fnum = fnum +1;
figure(fnum)
plot([1:N],xsave(9:12,:)')
title('b parameters')
legend('b_0 Kalman','b_1 Kalman','b_2 Kalman','b_3 Kalman','location','southeast')

%% Plot modeling error
resid = e(modelweek*24*7 + 1:end-K);
fnum = func_plotacfpacf(fnum, resid, cf, 0.05, 'resid');
fnum = fnum +1;
figure(fnum)
disp('%%%%%%%% RESID %%%%%%%%')
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
disp('%%%%%%%% 1-STEP %%%%%%%%')
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
%%
fnum = fnum +1;
figure(fnum)
time = 1:length(yFull);
plot(time,yFull,time+8,uFull(K+2:end),time+8,yhat(8,:))
legend('output','input','prediction')


