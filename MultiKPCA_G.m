function [alpha,Nlambda,lambda,K,K_bar] = MultiKPCA_G(X,h,w)
%[alpha,Nlambda,lambda,K,K_bar] = MultiKPCA_G(X,h)用于进行多高斯核PCA
% X 为数据矩阵，每一列代表一个样本点
% h 为每一个高斯核函数的尺度因子
% w 为每一个高斯核函数的加权系数

[X_M,X_N] = size(X);
h_num = numel(h);

if nargin <3
    w = ones(1,h_num)/h_num;
end

% 生成核矩阵
K = Gram(X,h,w);

% 核矩阵中心化
ONES = 1/X_N*ones(X_N,X_N);
K_bar = K - ONES*K - K*ONES + ONES*K*ONES;

% 对K_bar 进行特征分解
[alpha,Nlambda] = svd(K_bar); % 对于对称矩阵而言，其svd效果与eig相同
% [alpha,Nlambda] = eig(K_bar);
% lambda = Nlambda/X_N;
lambda = Nlambda/X_N;

function K = Gram(X,h,w)
[X_M,X_N] = size(X);
h_num = numel(h);
K = zeros(X_N,X_N);
for i=1:X_N
    for j=1:X_N
        for m = 1:h_num
            K(i,j) = K(i,j) + kfun(X(:,i),X(:,j),h(m))*w(m);
        end
    end
end





function y = kfun(x,y,sigma)
y = exp(-norm(x-y)^2/sigma);