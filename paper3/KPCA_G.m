function [alpha, Nlambda,lambda,K,K_bar] = KPCA_G(X,h)
%[alpha, lambda] = KPCA_G(X,h)
% 利用高斯核函数进行KPCA
% 要求数据矩阵X 每一列代表一个样本点
% 注意，求解过程中，N*lambda 是 N倍的方差，lambda才是方差
% KPCA可以通过广义特征值分解求解，但是通常是转化成一般特征值求解问题，然后对特征
% 向量进行缩放得到同样的效果

[X_M, X_N] = size(X);

% 生成核矩阵
K = Gram(X,h);

% 核矩阵中心化
ONES = 1/X_N*ones(X_N,X_N);
K_bar = K - ONES*K - K*ONES + ONES*K*ONES;

% 对K_bar 进行特征分解
[alpha,Nlambda] = svd(K_bar); % 对于对称矩阵而言，其svd效果与eig相同
% [alpha,Nlambda] = eig(K_bar);
% lambda = Nlambda/X_N;
lambda = Nlambda/X_N;



function K = Gram(X,h)
[X_M,X_N] = size(X);
K = zeros(X_N,X_N);
for i=1:X_N
    for j=1:X_N
        K(i,j) = kfun(X(:,i),X(:,j),h);
    end
end


function y = kfun(x,y,h)
y = exp(-norm(x-y)^2/h);