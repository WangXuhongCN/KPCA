clear;
clc;


%% 相关参数设置，数据准备

h = 165*2^(6); % 高斯核尺度因子
dim =30; % 降维维数
level = 0.99; % 置信水平

% 加载训练数据
X0 = load('d00.dat'); % 52 X 500

% 选取监控变量
X0=[X0(1:22,:); X0(42:end,:)]; % 33 X 500

% 获取数据矩阵的维数，样本数
[X0_M,X0_N]=size(X0);

% 对训练数据进行标准化
[X,X_mean,X_std] = zscore(X0.');
X = X.';

%% 进行KPCA
[alpha, Nlambda,lambda,K,KX_bar] = KPCA_G(X,h);

explained = diag(lambda);
% 确定降维维数
% for dim=1:X0_N
%     if sum(explained(1:dim))/sum(explained)>0.90
%         break;
%     end
% end

% 标准化alpha
alpha_nor = alpha*Nlambda^(-0.5);

% 选取降维矩阵
alpha_dim = alpha_nor(:,1:dim);
lambda_dim = lambda(1:dim,1:dim);

lam = diag(lambda);
lam_sum = sum(lam);
% for i=1:X0_N
%     if lam(i)<10^(-4)*lam_sum
%         break;
%     end
% end
% n0_num = i;
% n0_num = min(sum(lam>10^(-4)*lam_sum)+10,500);
n0_num = 40;

% alpha_r = alpha_nor(:,dim+1:end);
% lambda_r = lambda(dim+1:end,dim+1:end);
alpha_r = alpha_nor(:,dim+1:n0_num);
lambda_r = lambda(dim+1:n0_num,dim+1:n0_num);


% 计算降维后的数据
Xdim = alpha_dim.'*KX_bar;
Xr = alpha_r.'*KX_bar;

% 标准化降维后的数据，便于计算T2统计量
XTdim = lambda_dim^(-0.5)*Xdim;
% XTr = lambda_r^(-0.5)*Xr;
XTr = Xr;



% 计算训练样本的T2和SPE统计量
XT2 = diag(XTdim.'*XTdim);
XSPE = diag(XTr.'*XTr);

% 计算T2控制限
T2_limit = dim*(X0_N-1)/(X0_N-dim)*finv(level,dim,X0_N-dim);

% 计算SPE控制限
XSPE_mean = mean(XSPE);
XSPE_var = var(XSPE);
SPE_limit =XSPE_var/2/XSPE_mean * chi2inv(level,2*XSPE_mean^2/XSPE_var);
% r = X0_N-dim;
% SPE_limit = r*(X0_N-1)/(X0_N-r)*finv(level,r,X0_N-r);
% XSPE_var/2/XSPE_mean*



% plot(XT2,'b')
% hold on;
% plot(T2_limit*ones(1,X0_N),'r');


%% 测试
clc;
% 载入测试数据
Y0 = load('d16_te.dat'); % 960 X 52
Y0 = Y0.';

% 选取监控变量
Y0 = [Y0(1:22,:);Y0(42:end,:)];

% 获取测试矩阵的维数，样本数
[Y0_M,Y0_N] = size(Y0);

% 利用训练数据的均值，标准差进行标准化
Y = Y0 - repmat(X_mean.',1,Y0_N);
Y = Y./(repmat(X_std.',1,Y0_N));

% 计算测试样本的核矩阵
KY = zeros(X0_N,Y0_N);
for i=1:X0_N
    for j=1:Y0_N
        KY(i,j)=exp(-norm(X(:,i)-Y(:,j))^2/h);
    end
end

% 生成测试样本的去中心核矩阵
ONESX = 1/X0_N*ones(X0_N,X0_N);
ONESY = 1/X0_N*ones(X0_N,Y0_N);

KY_bar = KY - ONESX*KY - K*ONESY + ONESX*K*ONESY;
Ydim = alpha_dim.'* KY_bar;
Yr = alpha_r.'*KY_bar;
YTdim = lambda_dim^(-0.5)*Ydim;
% YTr = lambda_r^(-0.5)*Yr;
YTr = Yr;

% 计算YT2
YT2 = diag(YTdim.'*YTdim);
% 计算YSPE
YSPE = diag(YTr.'*YTr);

% 计算故障检测率
temp1 = YT2(161:end);
FD_YT2 = sum(temp1>T2_limit)/800
temp2 = YSPE(161:end);
FD_YSPE = sum(temp2>SPE_limit)/800

% 计算检测延迟
D_YT2 = min(find(temp1>T2_limit))
D_YSPE = min(find(temp2>SPE_limit))

% 结果可视化

subplot(2,1,1)
plot(YT2);
hold on;
plot(T2_limit*ones(1,Y0_N),'r-.')
ylabel('T^2');
legend('Statistics','Control limit')

subplot(2,1,2)
plot(YSPE);
hold on;
plot(SPE_limit*ones(1,Y0_N),'r-.')
xlabel('Samples');
ylabel('SPE')
legend('Statistics','Control limit')

% subplot(2,1,1)
% plot(YT2/T2_limit);
% hold on;
% plot(ones(1,Y0_N),'r')
% axis([0,960,0,2])
% 
% subplot(2,1,2)
% plot(YSPE/SPE_limit);
% hold on;
% plot(ones(1,Y0_N),'r')
% axis([0,960,0,2])



