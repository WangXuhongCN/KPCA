clear;
clc;

%% 离线训练

% 加载训练数据
X0 = load('d00.dat');
% X0 = X0; %令X0每一列为一个样本

% 选取监控变量
X0=[X0(1:22,:); X0(42:end,:)]; % 33 X 500

[X0_M,X0_N]=size(X0); %获取训练矩阵尺寸
[X,X_mean,X_std] = zscore(X0.');
X = X.';

% 设定带宽参数和默认主元数
ns = 10; %训练模型个数
c = 2.^(0:ns)*5*X0_M; %带宽参数数组
% c = [15,30,300,1000];
ns = numel(c);
dim = 30; %默认主成分数
level = 0.99; %显著水平
pT2n = level;
pT2f = 1-level;
pSPEn = level;
pSPEf = 1-level;

% 训练EKPCA模型
mout = MKPCA(X,c,dim);

%% 在线测试

% 加载测试数据
clc;
Y0 = load('d16_te.dat'); % 960 X 52
Y0 = Y0.';

% 选取监控变量
Y0 = [Y0(1:22,:);Y0(42:end,:)];


[Y0_M,Y0_N] = size(Y0);

%标注化测试数据
Y = Y0 - repmat(X_mean.',1,Y0_N);
Y = Y./(repmat(X_std.',1,Y0_N));

% 初始化Yfp结构体
% Yfp.pT2fx=0;
% Yfp.pT2fx2=0;
% Yfp.pT2xf=0;
% Yfp.pT2xn=0;
% Yfp.pT2f=1-level;
% Yfp.pT2n=level;
% Yfp.pT2x=0;
% Yfp.pSPEfx=0;
% Yfp.pSPEfx2=0;
% Yfp.pSPExf=0;
% Yfp.pSPExn=0;
% Yfp.pSPEf=1-level;
% Yfp.pSPEn=level;
% Yfp.pSPEx=0;
% Yfp.YT2=0;
% Yfp.YSPE=0;

a.pT2fx=0;
a.pT2fx2=0;
a.pT2xf=0;
a.pT2xn=0;
a.pT2f=1-level;
a.pT2n=level;
a.pT2x=0;
a.pSPEfx=0;
a.pSPEfx2=0;
a.pSPExf=0;
a.pSPExn=0;
a.pSPEf=1-level;
a.pSPEn=level;
a.pSPEx=0;
a.YT2=0;
a.YSPE=0;

Yfp = repmat(a,1,ns);

% 计算每个模型的后验故障概率
EpT2fx = 0;
EpSPEfx = 0;
SpT2fx = 0;
SpSPEfx = 0;
for i=1:ns
    %计算测试样本的核矩阵
    KY = zeros(X0_N,Y0_N);
    for j=1:X0_N
        for k=1:Y0_N
            KY(j,k)=exp(-norm(X(:,j)-Y(:,k))^2/c(i));
        end
    end
    
    %生成中心化核矩阵
    ONESX = 1/X0_N*ones(X0_N,X0_N);
    ONESY = 1/X0_N*ones(X0_N,Y0_N);
    KY_bar = KY - ONESX*KY - mout(i).KX*ONESY + ONESX*mout(i).KX*ONESY;
    
    %对测试数据进行降维
    Ydim = (mout(i).alpha_dim).'* KY_bar;
    Yr = (mout(i).alpha_r).'* KY_bar;
    
    %计算YT2
    YT2 = diag(Ydim.'*(mout(i).lambda_dim)^(-1)*Ydim);
    Yfp(i).YT2 = YT2;
    %计算YSPE
%     YSPE = diag(Yr.'*(mout(i).lambda_r)^(-1)*Yr);
    YSPE = diag(Yr.'*Yr);
    Yfp(i).YSPE = YSPE;
    
    %计算pT2(x|N)
    pT2xn = exp(-YT2/mout(i).T2_limit);
    Yfp(i).pT2xn = pT2xn;
    %计算pT2(x|F)
    pT2xf = exp(-(mout(i).T2_limit)./YT2);
    Yfp(i).pT2xf = pT2xf;
    %计算pT2(x)
    pT2x = pT2xn*pT2n + pT2xf*pT2f;
    Yfp(i).pT2x = pT2x;
    %计算pT2(F|x)
    pT2fx = (pT2xf.*pT2f)./pT2x;
    Yfp(i).pT2fx = pT2fx;
    Yfp(i).pT2fx2 = pT2fx.^2;
    SpT2fx = SpT2fx + pT2fx;
    
    %计算pSPE(x|N)
    pSPExn = exp(-YSPE/mout(i).SPE_limit);
    Yfp(i).pSPExn = pSPExn;
    %计算pSPE(x|F)
    pSPExf = exp(-(mout(i).SPE_limit)./YSPE);
    Yfp(i).pSPExf = pSPExf;
    %计算pSPE(x)
    pSPEx = pSPExn*pSPEn + pSPExf*pSPEf;
    Yfp(i).pSPEx = pSPEx;
    %计算pSPE(F|x)
    pSPEfx = (pSPExf.*pSPEf)./pSPEx;
    Yfp(i).pSPEfx = pSPEfx;
    Yfp(i).pSPEfx2 = pSPEfx.^2;
    SpSPEfx = SpSPEfx + pSPEfx;
end

% 计算最终的故障概率
for i=1:ns
    EpT2fx = EpT2fx + (Yfp(i).pT2fx2)./SpT2fx;
    EpSPEfx = EpSPEfx + (Yfp(i).pSPEfx2)./SpSPEfx;
end

% 计算检测准确率
temp1 = EpT2fx(161:end);
FD_EpT2fx = sum(temp1>0.01)/800
temp2 = EpSPEfx(161:end);
FD_EpSPEfx = sum(temp2>0.01)/800

% 计算检测延迟
D_EpT2fx = min(find(temp1>0.01))
D_EpSPEfx = min(find(temp2>0.01))


% 结果可视化
subplot(2,1,1)
plot(EpT2fx);
% plot(EpT2fx(1:800));
hold on;
plot(ones(1,Y0_N)*0.01,'r-.')
% plot(ones(1,800)*0.01,'r-.')
ylabel('ET^2');
legend('Statistics','Control limit')

subplot(2,1,2)
plot(EpSPEfx)
% plot(EpSPEfx(1:800))
hold on;
plot(ones(1,Y0_N)*0.01,'r-.')
% plot(ones(1,800)*0.01,'r-.')
xlabel('Samples');
ylabel('ESPE')
legend('Statistics','Control limit')