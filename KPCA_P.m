function [alpha, Nlambda,lambda,K,K_bar] = KPCA_P(X,d,c)
%[alpha, lambda] = KPCA_P(X,d,c)
% ���ö���ʽ�˺�������KPCA
% Ҫ�����ݾ���X ÿһ�д���һ��������

[X_M,X_N] = size(X);

% ���ɺ˾���
K = Gram(X,d,c);

% �˾������Ļ�
ONES = 1/X_N*ones(X_N,X_N);
K_bar = K - ONES*K - K*ONES + ONES*K*ONES;


% ��K_bar���������ֽ⣨SVD��
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