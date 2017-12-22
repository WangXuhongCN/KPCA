function mout = MKPCA(X,c,dim,level,f1)
if nargin < 3
    error('Not enough input arguments.');
end
if nargin == 3
    f1 = 0;
    level = 0.99;
end
if nargin == 4
    f1 = 0;
end

ns = numel(c);
[X_M,X_N]=size(X);

% 初始化mout结构体
a.alpha = zeros(X_N,X_N);
a.Nlambda = zeros(X_N,X_N);
a.lambda = zeros(X_N,X_N);
a.KX = zeros(X_N,X_N);
a.KX_bar = zeros(X_N,X_N);
a.T2_limit = 0;
a.SPE_limit=0;
a.dim = dim;
a.n0_num=0;
a.alpha_nor = 0;
a.alpha_dim=0;
a.lambda_dim=0;
a.alpha_r = 0;
a.lambda_r=0;

mout = repmat(a,1,ns);

% 进行KPCA
for i=1:ns
    [alpha, Nlambda,lambda,KX,KX_bar] = KPCA_G(X,c(i));
    mout(i).alpha = alpha;
    mout(i).Nlambda = Nlambda;
    mout(i).lambda = lambda;
    mout(i).KX = KX;
    mout(i).KX_bar = KX_bar;
    
    %确定主元数
    eigval = diag(lambda);
    if f1==1
        dim = dim_num(eigval);%eigval为lambda的对角元素
        mout(i).dim = dim;
    end
    
    %标准化alpha
    alpha_nor = alpha*Nlambda^(-0.5);
    mout(i).alpha_nor = alpha_nor;
    
    %选取降维矩阵
    alpha_dim = alpha_nor(:,1:dim);
    lambda_dim = lambda(1:dim,1:dim);
    mout(i).alpha_dim = alpha_dim;
    mout(i).lambda_dim = lambda_dim;
    
    %确定非零特征值数
%     n0_num = nonzeronum(eigval);%eigval为lambda的对角元素
    n0_num = 40;
    mout(i).n0_num = n0_num;
    
    %确定残差空间矩阵
    alpha_r = alpha_nor(:,dim+1:n0_num);
    lambda_r = lambda(dim+1:n0_num,dim+1:n0_num);
    mout(i).alpha_r = alpha_r;
    mout(i).lambda_r = lambda_r;
    
    %计算降维后的训练数据
    Xdim = alpha_dim.'*KX_bar;
    Xr = alpha_r.'*KX_bar;
    
    %计算训练样本的T2和SPE统计量
    XT2 = diag(Xdim.'*lambda_dim^(-1)*Xdim);
%     XSPE = diag(Xr.'*lambda_r^(-1)*Xr);
    XSPE = diag(Xr.'*Xr);
    
    %计算T2的控制限
    T2_limit = dim*(X_N-1)/(X_N-dim)*finv(level,dim,X_N-dim);
    mout(i).T2_limit = T2_limit;
    
    %计算SPE的控制限
    XSPE_mean = mean(XSPE);
    XSPE_var = var(XSPE);
    SPE_limit = XSPE_var/2/XSPE_mean * chi2inv(level,2*XSPE_mean^2/XSPE_var);
    mout(i).SPE_limit = SPE_limit;    
end



function dim = dim_num(eigval)
% 采用平均特征值的方法
m = mean(eigval);
dim = sum(eigval>m);

function n0_num = nonzeronum(eigval)
eig_sum = sum(eigval);
n0_num = min(sum(eigval>10^(-4)*eig_sum)+10,500);
% m = mean(eigval);
% n0_num = sum(eigval>m)*2;