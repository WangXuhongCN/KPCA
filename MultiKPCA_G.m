function [alpha,Nlambda,lambda,K,K_bar] = MultiKPCA_G(X,h,w)
%[alpha,Nlambda,lambda,K,K_bar] = MultiKPCA_G(X,h)���ڽ��ж��˹��PCA
% X Ϊ���ݾ���ÿһ�д���һ��������
% h Ϊÿһ����˹�˺����ĳ߶�����
% w Ϊÿһ����˹�˺����ļ�Ȩϵ��

[X_M,X_N] = size(X);
h_num = numel(h);

if nargin <3
    w = ones(1,h_num)/h_num;
end

% ���ɺ˾���
K = Gram(X,h,w);

% �˾������Ļ�
ONES = 1/X_N*ones(X_N,X_N);
K_bar = K - ONES*K - K*ONES + ONES*K*ONES;

% ��K_bar ���������ֽ�
[alpha,Nlambda] = svd(K_bar); % ���ڶԳƾ�����ԣ���svdЧ����eig��ͬ
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