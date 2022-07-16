function fitness=fitnessfun(population, params)
% 子程序：计算适应度函数, 函数名称存储为fitnessfun.m

popsize=size(population,1);   %有popsize个个体

% 开启并行计算
parfor i=1:popsize
     
    ve=population(i,1);
    p=population(i,2);
    q=population(i,3);
    
    fitness(i)=targetfun(ve,p,q);  %计算函数值，即适应度
    
end
fitness = fitness';
end
%% 子程序：将2进制数转换为10进制数,函数名称存储为BintoDec.m

function x=BintoDec(Population)

BitLength=size(Population,2);
x=Population(BitLength);

for i=1:BitLength-1
    x=x+Population(BitLength-i)*power(2,i);
end
end

%% 子程序：将2进制数分割为三个基因并转换为十进制,函数名称存储为BintoThr.m

function vepq=BintoThr(Population, params)

% global BitLength
% global num_ve
% global num_p
% global num_q
% global ve_max
% global ve_min
% global p_max
% global p_min
% global q_max
% global q_min

BitLength = params.BitLength;
num_ve = params.num_ve;
num_p = params.num_p;
num_q = params.num_q;
ve_max = params.ve_max;
ve_min = params.ve_min;
p_max = params.p_max;
p_min = params.p_min;
q_max = params.q_max;
q_min = params.q_min;

    ve_1=BintoDec(Population(1:num_ve));  %将二进制转换为十进制
    p_1=BintoDec(Population(num_ve+1:num_ve+num_p));  %将二进制转换为十进制
    q_1=BintoDec(Population(num_ve+num_p+1:BitLength));  %将二进制转换为十进制
    
    vepq(1)=ve_min+ve_1*(ve_max-ve_min)/(2^num_ve-1);
    vepq(2)=p_min+p_1*(p_max-p_min)/(2^num_p-1);
    vepq(3)=q_min+q_1*(q_max-q_min)/(2^num_q-1);
    
end
%子程序：对于优化最大值或极大值函数问题，目标函数可以作为适应度函数
%函数名称存储为targetfun.m

%% 计算对比度

function y=targetfun(ve,p,q) %目标函数

%% 聚焦
persistent Kr;
persistent Rc;
persistent T_A;
persistent T_R;
persistent Ta;
persistent Tp;
persistent c;
persistent contrast0;
persistent fc;
persistent lambda;
persistent p0;
persistent q0;
persistent s_fr_fa;
persistent tac;
persistent ve0;
if isempty(Kr)
 load Target1DataNoNoise
end 
%load Target1Data
%load Target1DataNoNoise

r_ref=sqrt(Rc^2+ve^2*T_A.^2+p*T_A)+q*T_A;

s_ref=(abs(T_R-2*r_ref/c)<Tp/2).*(abs(T_A-tac)<Ta/2).*exp(-1i*2*pi*fc*2*r_ref/c).*exp(1i*pi*Kr*(T_R-2*r_ref/c).^2);

s_fr_fa_ref=fft2(s_ref); % clear s_ref

s_tr_ta=fftshift(ifft2(fftshift(s_fr_fa.*conj(s_fr_fa_ref)))); % clear s_fr_fa s_fr_fa_ref


%% 计算对比度
AveragePower=mean(mean(abs(s_tr_ta).^2));
%AveragePower=mean(abs(s_tr_ta).^2);
% TotalPower=sum(sum(abs(s_tr_ta).^2));

Contrast=sqrt(mean(mean(abs(s_tr_ta).^2-AveragePower).^2))/AveragePower;
%Contrast=sqrt(mean((abs(s_tr_ta).^2-AveragePower).^2))/AveragePower;
%Contrast=sqrt(mean((abs(s_tr_ta).^2-repmat(AveragePower, size(s_tr_ta,1), 1)).^2))/AveragePower;
% Entropy=-sum(sum(abs(s_tr_ta).^2/TotalPower.*log(abs(s_tr_ta).^2/TotalPower)));

y=Contrast;  % 遗传算法是求最大值

end