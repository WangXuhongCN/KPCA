clear all;
n=500;
t=unifrnd(0.01,2,n,1);
e=normrnd(0,0.1,n,3);
x1=t+e(:,1);
x2=t.^2-3*t+e(:,2);
x3=-t.^3+3*t.^2+e(:,3);
% Collect the normal data 
Xtrain = [x1 x2 x3];
% load Xtrain
[axp,mx,stdx] = auto(Xtrain);

% Conduct KPCA
% Kernel matrix
nb_data = size(axp,1);sigma2 = 20;
XXhp = sum(axp.^2,2)*ones(1,nb_data);
omegap = XXhp+XXhp'-2*(axp*axp');
K = exp(-omegap./sigma2); Kp = K;
% C = eye(nb_data)-ones(nb_data)/nb_data;			% Centering matrix.
% K=C*Kp*C;

% centered kernel matrix : Z*omega*Z
%Zc = eye(nb_data) - ones(nb_data)./nb_data;
%omega = Zc*omega*Zc;
Meanvec = mean(K);
MM = mean(Meanvec);
for i=1:nb_data,
    K(i,:) = K(i,:)-Meanvec(i);
end
for i=1:nb_data,
    K(:,i) = K(:,i)-Meanvec(i);
end
K = K+MM;


% numerical stability issues
K = (K+K')./2;

% Eigenvalues are computed with svd
[U,S,V] = svd(K);
PC = 2;
S = diag(S)/(nb_data-1);

% rescaling the eigenvectors
for i=1:PC,
    % eigvec(:,i) = eigvec(:,i)./sqrt(eigvec(:,i)'*eigval(i)*(nb_data-1)*eigvec(:,i));
    U(:,i) = U(:,i)./sqrt(S(i)*(nb_data-1));
end
    
    
% only keep the number of components
P = U(:,1:PC);
scoresp = K'*P;

% %T2 and SPE control limit
S = S(1:PC);
t2std = diag(sqrt(S));
sscore = scoresp*inv(t2std);
T2 = sum(sscore(:,1:PC).^2,2);
I = size(axp,1);
T2cfdLimit = PC*(I^2-1)/(I*(I-PC))*finv(0.99,PC,I-PC);

res = diag(K)-sum(scoresp(:,1:PC).^2, 2);
m = mean(res);
v = var(res);
SPEcfdLimit = v/(2*m)*chi2inv(0.99,2*m.^2/v);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %SPE and T2 monitoring result of Xbad
n2=500;
t=unifrnd(0.01,2,n2,1);
e=normrnd(0,0.1,n2,3);
x1_2=t+e(:,1);
x2_2=t.^2-3*t+e(:,2);
x3_2=-t.^3+3*t.^2+e(:,3);
for i=251:n2
    x1_2(i,1)=x1_2(i,1)+0.005*(i-250);
end
x2_2(101:n2,1)=x2_2(101:n2,1)-0.5;

Xtest1=[x1_2 x2_2 x3_2];
% load Xtest1
% Collect the normal data 

axt_1 = scale(Xtest1,mx,stdx);

% XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
nb_data1 = size(axt_1,1);
XXh2 = sum(axt_1.^2,2)*ones(1,nb_data1);
omega = XXhp +XXh2' - 2*axp*axt_1';
Kt1 = exp(-omega./sigma2);

% centered New kernel
N = nb_data;
I_N   = eye(N);
E_N   = ones(N,N);
one_N = ones(N,1);
A = (I_N-E_N./N)*U(:,1:PC);
temp = Kt1-repmat(Kp*one_N./N,1,size(axt_1,1));    
scorest = temp'*A;

fsnorm = zeros(size(axt_1,1),1);
nb_data = size(axt_1,1);sigma2 = 20;
XXh = sum(axt_1.^2,2)*ones(1,nb_data);
omega = XXh+XXh'-2*axt_1*axt_1';
Ktt = exp(-omega./sigma2);
Meanvec2 = mean(Kt1);
for i = 1:size(axt_1,1),
        fsnorm(i) = Ktt(i,i)-2*Meanvec2(i)+MM;
end

sscore = scorest*inv(t2std);
T2 = sum(sscore(:,1:PC).^2,2);
res = fsnorm-sum(scorest(:,1:PC).^2, 2);
subplot(211);plot(T2);hold on;plot(1:500,T2cfdLimit,'r');title('T^2');
subplot(212);plot(res);hold on;plot(1:500,SPEcfdLimit,'r');title('SPE');
