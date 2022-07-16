function [ ] = bga( )
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
%% 分析遗传算法结果的统计特性，通过随机独立实验

tic

clc;
clear all;
close all;

load Target1DataNoNoise

clear s_fr_fa Rc lambda c fc Kr T_R T_A Tp tac Ta

%%

% global BitLength
% global num_ve
% global num_p
% global num_q
% global k1
% global k2
% global k3
% global k4
% global ve_max
% global ve_min
% global p_max
% global p_min
% global q_max
% global q_min

k1=1;
k2=0.5;
k3=1;
k4=0.5;

%计算染色体长度

ve_max=round(ve0)+0.9;
ve_min=round(ve0)-0.9;
p_max=(round(p0/1e3)+180)*1e3;
p_min=(round(p0/1e3)-180)*1e3;
q_max=round(q0)+18;
q_min=round(q0)-18;

num_ve=ceil(log2((ve_max-ve_min)/0.03));
num_p=ceil(log2((p_max-p_min)/9000));
num_q=ceil(log2((q_max-q_min)/2.25));
BitLength=num_ve+num_p+num_q;%


params.BitLength = BitLength;
params.num_ve = num_ve;
params.num_p = num_p;
params.num_q = num_q;
params.ve_max = ve_max;
params.ve_min = ve_min;
params.p_max = p_max;
params.p_min = p_min;
params.q_max = q_max;
params.q_min = q_min;

popsize=32; %初始种群大小
Generationnmax=100;  %最大代数

pc=0.8;
pm=0.1;

%% 通过随机实验，获得结果的统计信息

ExperimentNumber=1;

for en=1:ExperimentNumber
    
    en
    
    %% 遗传算法
    
    %产生初始种群
    population=round(rand(popsize,BitLength));
    
    %计算适应度,返回适应度fitness，即图像对比度
    fitness=fitnessfun(population, params);  % 每一个个体一个适应度，因此fitness是一个向量
    
    % figure,plot(fitness),hold on
    
    %计算累积概率cumsump，中间变量
    cumsump=cumsumpfun(fitness);      % 各个个体的适应度在所有个体适应度之和的比例 的累加和
    
    [fmax,nmax]=max(fitness);
    % fmean=mean(fitness);
    
    %最优个体保存
    bestvepq=population(nmax,:);   % 每一行一个个体
    
    Generation=1;
    
    while Generation<=Generationnmax
        
        %     [pcrossover,pmutation]=adaptive(popsize,fitness,fmax,fmean); %计算自适应概率，输出pcrossover交叉与pmutation变异概率，每个个体都有
        
        for j=1:2:popsize
            
            sel=selection(population,cumsump); %选择操作，输出被选择的个体的序号，轮盘赌方法；选两个个体
            
            %交叉操作
            cro=crossover(population,sel,pc,fitness); % 利用选择出来的个体进行交叉操作，形成两个新个体，新种群所有个体，均由选择的个体经交叉、变异得到
            cronew(j,:)=cro(1,:);
            cronew(j+1,:)=cro(2,:);
            
            %变异操作，对交叉后的个体进行变异
            mutmnew(j,:)=mutation(cronew(j,:),pm);
            mutmnew(j+1,:)=mutation(cronew(j+1,:),pm);
            
        end
        
        Offspring = mutmnew;
        OffFitness=fitnessfun(Offspring, params);  %计算新种群的适应度
        
        selection_process = 'elitist';
        if strcmp(selection_process,'elitist')
            intpopulation = [population;Offspring];
            intFitness = [fitness;OffFitness];
            
            [xxx,y]=sort(intFitness,1,'descend' );
            population = intpopulation(y(1:popsize),:);
            fitness = intFitness(y(1:popsize),:);
        elseif strcmp(selection_process,'sus')
            intpopulation = [population;Offspring];
            intFitness = [fitness;OffFitness];
            
            y = sus(intFitness, popsize);
            population = intpopulation(y(1:popsize),:);
            fitness = intFitness(y(1:popsize),:);
            
            % keep elites
            [fmax,nmax]=max(intFitness);
            [fmax1,nmax1]=max(fitness);
            if fmax>fmax1
                population(1, :) = intpopulation(nmax, :);
                fitness(1, 1) = intFitness(nmax, :);
            end
            
        elseif strcmp(selection_process,'rws')
            intpopulation = [population;Offspring];
            intFitness = [fitness;OffFitness];
            
            num = 0;y=[];
            while num < popsize,
                y = [y, RouletteWheelSelection(intFitness)];
                num = num + 1;
            end
            
            population = intpopulation(y(1:popsize),:);
            fitness = intFitness(y(1:popsize),:);
            
            % keep elites
            [fmax,nmax]=max(intFitness);
            [fmax1,nmax1]=max(fitness);
            if fmax>fmax1
                population(1, :) = intpopulation(nmax, :);
                fitness(1, 1) = intFitness(nmax, :);
            end
        else
            [fmin,nmin]=min(fitness);
            [fmax,nmax]=max(fitness);
            bestvepq = population(nmax, :);

            population = Offspring;
            fitness = OffFitness;

            %最优个体保存
            population(nmin,:)=bestvepq;  % 用上一代的最优个体，替换新一代的最差个体
            fitness(nmin)=fmax;
        end
        
        %     plot(fitness)
        
        cumsump=cumsumpfun(fitness);  %计算累计概率
        
        %记录当前代最大的适应度和平均适应度
        [fmax,nmax]=max(fitness);
        %     fmean=mean(fitness);
        ymax(Generation)=fmax;
        %     ymean(Generation)=fmean;
        
        disp([num2str(Generation),' ', num2str(fmax)])
        
        %记录当前代的最佳染色体个体
        bestvepq=population(nmax,:);
        vepq=BintoThr(population(nmax,:), params);
        ve=vepq(1);
        p=vepq(2);
        q=vepq(3);
        
        vemax(Generation)=ve;
        pmax(Generation)=p;
        qmax(Generation)=q;
        
        %     if fmax>40.05
        %         break
        %     end
        
        if Generation==Generationnmax % && abs(ymax(Generation)-ymax(Generation-4))<0.01  % 终止条件，多代不变？
            break
        end
        
        Generation=Generation+1;
        
    end
    
    % [x,Generation]=size(ymax);
    
    Contrast(en)=fmax;
    
    Ve(en)=ve;
    P(en)=p;
    Q(en)=q;
    
    
    %%
    
    % h=figure;
    % aa=get(h,'Position');
    % set(h,'Position',[aa(1)     aa(2)   aa(3)*0.8   aa(4)*0.8])
    % plot(1:Generation,ymax)
    % xlabel('Genration'),ylabel('Contrast')
    %
    % figure2= figure('Color',[1 1 1]);
    % axes1 = axes('Parent',figure2);
    % plot(1:Generation,vemax)
    % set(axes1,'FontName','Times New Roman','Layer','top');
    % xlabel('迭代代数','FontName','宋体','FontSize',10.5),ylabel('Ve','FontName','Times New Roman','FontSize',10.5)
    % figure3= figure('Color',[1 1 1]);
    % axes1 = axes('Parent',figure3);
    % plot(1:Generation,pmax)
    % set(axes1,'FontName','Times New Roman','Layer','top');
    % xlabel('迭代代数','FontName','宋体','FontSize',10.5),ylabel('p','FontName','Times New Roman','FontSize',10.5)
    % figure4= figure('Color',[1 1 1]);
    % axes1 = axes('Parent',figure4);
    % plot(1:Generation,qmax)
    % set(axes1,'FontName','Times New Roman','Layer','top');
    % xlabel('迭代代数','FontName','宋体','FontSize',10.5),ylabel('q','FontName','Times New Roman','FontSize',10.5)
    
    %%
    disp([vepq,' ', num2str(fmax)])
    eval(sprintf('save T1GaResults%d',en))
    
end

%绘制经过遗传运算后的适应度曲线。一般地，如果进化过程中种群的平均适应度与最大适
%应度在曲线上有相互趋同的形态，表示算法收敛进行得很顺利，没有出现震荡；在这种前
%提下，最大适应度个体连续若干代都没有发生进化表明种群已经成熟。

% h=figure;
% aa=get(h,'Position');
% set(h,'Position',[aa(1)     aa(2)   aa(3)*0.8   aa(4)*0.8])
% plot(1:Generation,ymax)
% xlabel('Genration'),ylabel('Contrast')
%
% figure2= figure('Color',[1 1 1]);
% axes1 = axes('Parent',figure2);
% plot(1:Generation,vemax)
% set(axes1,'FontName','Times New Roman','Layer','top');
% xlabel('迭代代数','FontName','宋体','FontSize',10.5),ylabel('Ve','FontName','Times New Roman','FontSize',10.5)
% figure3= figure('Color',[1 1 1]);
% axes1 = axes('Parent',figure3);
% plot(1:Generation,pmax)
% set(axes1,'FontName','Times New Roman','Layer','top');
% xlabel('迭代代数','FontName','宋体','FontSize',10.5),ylabel('p','FontName','Times New Roman','FontSize',10.5)
% figure4= figure('Color',[1 1 1]);
% axes1 = axes('Parent',figure4);
% plot(1:Generation,qmax)
% set(axes1,'FontName','Times New Roman','Layer','top');
% xlabel('迭代代数','FontName','宋体','FontSize',10.5),ylabel('q','FontName','Times New Roman','FontSize',10.5)

toc

end
%% 以下均为子程序



%% 子程序： 自适应概率计算

function [pcrossover,pmutation]=adaptive(popsize,fitness,fmax,fmean)

global k1
global k2
global k3
global k4

for i=1:popsize
    
    if fitness(i)==fmax
        pcrossover(i)=k1*(fmax-fitness(i))/(fmax-fmean);
        pmutation(i)=0.01;
    elseif fitness(i)>fmean
        pcrossover(i)=k1*(fmax-fitness(i))/(fmax-fmean);
        pmutation(i)=k2*(fmax-fitness(i))/(fmax-fmean);
    else
        pcrossover(i)=k3;
        pmutation(i)=k4;
    end
    
end
end

%% 子程序：新种群交叉操作,函数名称存储为crossover.m

% function cro=crossover(population,sel,pc,fitness)
%
% BitLength=size(population,2);
%
% if fitness(sel(1))<fitness(sel(2))   % 谁的适应度高就用谁的的交叉概率
%             pc=pcrossover(sel(1));
% else
%             pc=pcrossover(sel(2));
% end
%
% pcc=IfCroIfMut(pc);  %根据交叉概率决定是否进行交叉操作，1则是，0则否，概率越高，交叉的可能性越大
%
% if pcc==1
%     chb=round(rand*(BitLength-2))+1;  %在[1,BitLength-1]范围内随机产生一个交叉位
%     cro(1,:)=[population(sel(1),1:chb) population(sel(2),chb+1:BitLength)];
%     cro(2,:)=[population(sel(2),1:chb) population(sel(1),chb+1:BitLength)];
% else
%     cro(1,:)=population(sel(1),:);
%     cro(2,:)=population(sel(2),:);
% end
% end

% 均匀交叉
% function cro=crossover(population,sel,pc,fitness)
%
% BitLength=size(population,2);
% cro(1,:) = population(sel(1),:);
% cro(2,:) = population(sel(2),:);
% for i=1:BitLength
%     if(rand(1)<=pc)
%         chb=i;
%         cro(1,chb)=population(sel(2),chb);
%         cro(2,chb)=population(sel(1),chb);
%     end
% end
% end

% MultiPointCross
function cro=crossover(population,sel,pc,fitness)
k = 10;

BitLength=size(population,2);
Parent1=population(sel(1),:);
Parent2=population(sel(2),:);
Children1=Parent1;
Children2=Parent2;
Points=sort(unidrnd(BitLength,1,2*k));
for i=1:k
    if(rand(1)<=pc)
        Children1(Points(2*i-1):Points(2*i))=Parent2(Points(2*i-1):Points(2*i));
        Children2(Points(2*i-1):Points(2*i))=Parent1(Points(2*i-1):Points(2*i));
    end
end
cro = [Children1;Children2];
end

%% 子程序：计算适应度函数, 函数名称存储为fitnessfun.m

function fitness=fitnessfun(population, params)

popsize=size(population,1);   %有popsize个个体

% 开启并行计算
parfor i=1:popsize
    
    %     i
    vepq=BintoThr(population(i,:), params);   % （二进制）解码
    ve=vepq(1);
    p=vepq(2);
    q=vepq(3);
    
    fitness(i)=targetfun(ve,p,q);  %计算函数值，即适应度
    
end
fitness = fitness';
end

%% 计算累计概率

function cumsump=cumsumpfun(fitness)

%计算选择概率
fsum=sum(fitness);              % 种群所有个体的适应度之和
Pperpopulation=fitness/fsum;    % 各个个体的适应度在所有个体适应度之和的比例

%计算累积概率
cumsump = cumsum(Pperpopulation);   % 累加和，
cumsump=cumsump';

end

%% 子程序：新种群变异操作，函数名称存储为mutation.m，基本位变异，选取一位进行变异

% function snnew=mutation(snew,pm)
%
% BitLength=size(snew,2);
% snnew=snew;
% pmm=IfCroIfMut(pm);  %根据变异概率决定是否进行变异操作，1则是，0则否
%
% if pmm==1
%     chb=round(rand*(BitLength-1))+1;  %在[1,BitLength]范围内随机产生一个变异位
%     snnew(chb)=abs(snew(chb)-1);   %进行一个0-1,1-0的反转
% end
% end

% 多点变异
function snnew=mutation(snew,pm)
BitLength=size(snew,2);
snnew=snew;
for i=1:BitLength
    if(rand(1)<pm)
        chb=i;
        snnew(chb)=abs(snew(chb)-1);   %进行一个0-1,1-0的反转
    end
end
end

%% 子程序：判断遗传运算是否需要进行交叉或变异, 函数名称存储为IfCroIfMut.m；随机确定，交叉概率越高，变异的概率也越高

function pcc=IfCroIfMut(pcorpm)   % 输入为交叉概率

if rand<pcorpm
    pcc=1;
else pcc=0;
end

end

%% 子程序：新种群选择操作, 函数名称存储为selection.m；轮盘赌方法

function seln=selection(population,cumsump)

%从种群中选择两个个体

for i=1:2
    r=rand;  %产生一个随机数
    prand=cumsump-r;
    j=1;
    while prand(j)<0
        j=j+1;
    end
    seln(i)=j; %选中个体的序号,随机两选个个体
end
end

%% 子程序：将2进制数转换为10进制数,函数名称存储为BintoDec.m

function x=BintoDec(population)

BitLength=size(population,2);
x=population(BitLength);

for i=1:BitLength-1
    x=x+population(BitLength-i)*power(2,i);
end
end

%% 子程序：将2进制数分割为三个基因并转换为十进制,函数名称存储为BintoThr.m

function vepq=BintoThr(population, params)

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

ve_1=BintoDec(population(1:num_ve));  %将二进制转换为十进制
p_1=BintoDec(population(num_ve+1:num_ve+num_p));  %将二进制转换为十进制
q_1=BintoDec(population(num_ve+num_p+1:BitLength));  %将二进制转换为十进制

vepq(1)=ve_min+ve_1*(ve_max-ve_min)/(2^num_ve-1);
vepq(2)=p_min+p_1*(p_max-p_min)/(2^num_p-1);
vepq(3)=q_min+q_1*(q_max-q_min)/(2^num_q-1);

end
%子程序：对于优化最大值或极大值函数问题，目标函数可以作为适应度函数
%函数名称存储为targetfun.m

%% 计算对比度

function y=targetfun(ve,p,q) %目标函数

%% 聚焦
% 定义静态变量，避免每次调用该函数，都需要重新加载数据
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


