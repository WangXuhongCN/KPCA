clear;
clc;


%% ��ز������ã�����׼��

h = [300,600,1200,2400,4800,9600]; % ��˹�˳߶�����
h_num = numel(h);
w = ones(1,h_num)/h_num;
dim =30; % ��άά��
level = 0.99; % ����ˮƽ

% ����ѵ������
X0 = load('d00.dat'); % 52 X 500

% ѡȡ��ر���
X0=[X0(1:22,:); X0(42:end,:)]; % 33 X 500

% ��ȡ���ݾ����ά����������
[X0_M,X0_N]=size(X0);

% ��ѵ�����ݽ��б�׼��
[X,X_mean,X_std] = zscore(X0.');
X = X.';

%% ����KPCA
[alpha, Nlambda,lambda,K,KX_bar] = MultiKPCA_G(X,h,w);

explained = diag(lambda);
% ȷ����άά��
% for dim=1:X0_N
%     if sum(explained(1:dim))/sum(explained)>0.99
%         break;
%     end
% end

% ��׼��alpha
alpha_nor = alpha*Nlambda^(-0.5);

% ѡȡ��ά����
alpha_dim = alpha_nor(:,1:dim);
lambda_dim = lambda(1:dim,1:dim);

alpha_r = alpha_nor(:,dim+1:end);
lambda_r = lambda(dim+1:end,dim+1:end);

% ���㽵ά�������
Xdim = alpha_dim.'*KX_bar;
Xr = alpha_r.'*KX_bar;

% ��׼����ά������ݣ����ڼ���T2ͳ����
XTdim = lambda_dim^(-0.5)*Xdim;
XTr = lambda_r^(-0.5)*Xr;

% ����ѵ��������T2��SPEͳ����
XT2 = diag(XTdim.'*XTdim);
XSPE = diag(XTr.'*XTr);

% ����T2������
T2_limit = dim*(X0_N-1)/(X0_N-dim)*finv(level,dim,X0_N-dim);

% ����SPE������
XSPE_mean = mean(XSPE);
XSPE_var = var(XSPE);
% SPE_limit =2* XSPE_var/XSPE_mean * chi2inv(level,2*XSPE_mean^2/XSPE_var);
r = X0_N-dim;
SPE_limit = r*(X0_N-1)/(X0_N-r)*finv(level,r,X0_N-r);
% XSPE_var/2/XSPE_mean*



% plot(XT2,'b')
% hold on;
% plot(T2_limit*ones(1,X0_N),'r');


%% ����
% �����������
Y0 = load('d05_te.dat'); % 960 X 52
Y0 = Y0.';

% ѡȡ��ر���
Y0 = [Y0(1:22,:);Y0(42:end,:)];

% ��ȡ���Ծ����ά����������
[Y0_M,Y0_N] = size(Y0);

% ����ѵ�����ݵľ�ֵ����׼����б�׼��
Y = Y0 - repmat(X_mean.',1,Y0_N);
Y = Y./(repmat(X_std.',1,Y0_N));

% ������������ĺ˾���
KY = zeros(X0_N,Y0_N);
for i=1:X0_N
    for j=1:Y0_N
        for m = 1:h_num
            KY(i,j)= KY(i,j)+exp(-norm(X(:,i)-Y(:,j))^2/h(m))*w(m);
        end
    end
end

% ���ɲ���������ȥ���ĺ˾���
ONESX = 1/X0_N*ones(X0_N,X0_N);
ONESY = 1/X0_N*ones(X0_N,Y0_N);

KY_bar = KY - ONESX*KY - K*ONESY + ONESX*K*ONESY;
Ydim = alpha_dim.'* KY_bar;
Yr = alpha_r.'*KY_bar;
YTdim = lambda_dim^(-0.5)*Ydim;
YTr = lambda_r^(-0.5)*Yr;

% ����YT2
YT2 = diag(YTdim.'*YTdim);
% ����YSPE
YSPE = diag(YTr.'*YTr);

subplot(2,1,1)
plot(YT2);
hold on;
plot(T2_limit*ones(1,Y0_N),'r')

subplot(2,1,2)
plot(YSPE);
hold on;
plot(SPE_limit*ones(1,Y0_N),'r')




