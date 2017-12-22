function [alpha, Nlambda,lambda,K,K_bar] = KPCA_P(X,d,c)
%[alpha, lambda] = KPCA_P(X,d,c)
% 利用多项式核函数进行KPCA
% 要求数据矩阵X 每一列代表一个样本点

[X_M,X_N] = size(X);

% 生成核矩阵
K = Gram(X,d,c);

% 核矩阵中心化
ONES = 1/X_N*ones(X_N,X_N);
K_bar = K - ONES*K - K*ONES + ONES*K*ONES;


% 对K_bar进行特征分解（SVD）
[alpha, Nlambda] = svd(K_bar);
lambda = Nlambda/X_N;



function K = Gram(X,d,c)
[X_M,X_N] = size(X);
K = zeros(X_N,X_N);
for i=1:X_N
    for j=1:X_N
        K(i,j) = kfun(X(:,i),X(:,j),d,c);
    end
end


function y = kfun(x,y,d,c)
y = (x.'*y+c)^d;