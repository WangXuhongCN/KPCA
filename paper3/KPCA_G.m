function [alpha, Nlambda,lambda,K,K_bar] = KPCA_G(X,h)
%[alpha, lambda] = KPCA_G(X,h)
% ���ø�˹�˺�������KPCA
% Ҫ�����ݾ���X ÿһ�д���һ��������
% ע�⣬�������У�N*lambda �� N���ķ��lambda���Ƿ���
% KPCA����ͨ����������ֵ�ֽ���⣬����ͨ����ת����һ������ֵ������⣬Ȼ�������
% �����������ŵõ�ͬ����Ч��

[X_M, X_N] = size(X);

% ���ɺ˾���
K = Gram(X,h);

% �˾������Ļ�
ONES = 1/X_N*ones(X_N,X_N);
K_bar = K - ONES*K - K*ONES + ONES*K*ONES;

% ��K_bar ���������ֽ�
[alpha,Nlambda] = svd(K_bar); % ���ڶԳƾ�����ԣ���svdЧ����eig��ͬ
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