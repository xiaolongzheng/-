function [ ] = bga( )
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%% �����Ŵ��㷨�����ͳ�����ԣ�ͨ���������ʵ��

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

%����Ⱦɫ�峤��

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

popsize=32; %��ʼ��Ⱥ��С
Generationnmax=100;  %������

pc=0.8;
pm=0.1;

%% ͨ�����ʵ�飬��ý����ͳ����Ϣ

ExperimentNumber=1;

for en=1:ExperimentNumber
    
    en
    
    %% �Ŵ��㷨
    
    %������ʼ��Ⱥ
    population=round(rand(popsize,BitLength));
    
    %������Ӧ��,������Ӧ��fitness����ͼ��Աȶ�
    fitness=fitnessfun(population, params);  % ÿһ������һ����Ӧ�ȣ����fitness��һ������
    
    % figure,plot(fitness),hold on
    
    %�����ۻ�����cumsump���м����
    cumsump=cumsumpfun(fitness);      % �����������Ӧ�������и�����Ӧ��֮�͵ı��� ���ۼӺ�
    
    [fmax,nmax]=max(fitness);
    % fmean=mean(fitness);
    
    %���Ÿ��屣��
    bestvepq=population(nmax,:);   % ÿһ��һ������
    
    Generation=1;
    
    while Generation<=Generationnmax
        
        %     [pcrossover,pmutation]=adaptive(popsize,fitness,fmax,fmean); %��������Ӧ���ʣ����pcrossover������pmutation������ʣ�ÿ�����嶼��
        
        for j=1:2:popsize
            
            sel=selection(population,cumsump); %ѡ������������ѡ��ĸ������ţ����̶ķ�����ѡ��������
            
            %�������
            cro=crossover(population,sel,pc,fitness); % ����ѡ������ĸ�����н���������γ������¸��壬����Ⱥ���и��壬����ѡ��ĸ��徭���桢����õ�
            cronew(j,:)=cro(1,:);
            cronew(j+1,:)=cro(2,:);
            
            %����������Խ����ĸ�����б���
            mutmnew(j,:)=mutation(cronew(j,:),pm);
            mutmnew(j+1,:)=mutation(cronew(j+1,:),pm);
            
        end
        
        Offspring = mutmnew;
        OffFitness=fitnessfun(Offspring, params);  %��������Ⱥ����Ӧ��
        
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

            %���Ÿ��屣��
            population(nmin,:)=bestvepq;  % ����һ�������Ÿ��壬�滻��һ����������
            fitness(nmin)=fmax;
        end
        
        %     plot(fitness)
        
        cumsump=cumsumpfun(fitness);  %�����ۼƸ���
        
        %��¼��ǰ��������Ӧ�Ⱥ�ƽ����Ӧ��
        [fmax,nmax]=max(fitness);
        %     fmean=mean(fitness);
        ymax(Generation)=fmax;
        %     ymean(Generation)=fmean;
        
        disp([num2str(Generation),' ', num2str(fmax)])
        
        %��¼��ǰ�������Ⱦɫ�����
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
        
        if Generation==Generationnmax % && abs(ymax(Generation)-ymax(Generation-4))<0.01  % ��ֹ������������䣿
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
    % xlabel('��������','FontName','����','FontSize',10.5),ylabel('Ve','FontName','Times New Roman','FontSize',10.5)
    % figure3= figure('Color',[1 1 1]);
    % axes1 = axes('Parent',figure3);
    % plot(1:Generation,pmax)
    % set(axes1,'FontName','Times New Roman','Layer','top');
    % xlabel('��������','FontName','����','FontSize',10.5),ylabel('p','FontName','Times New Roman','FontSize',10.5)
    % figure4= figure('Color',[1 1 1]);
    % axes1 = axes('Parent',figure4);
    % plot(1:Generation,qmax)
    % set(axes1,'FontName','Times New Roman','Layer','top');
    % xlabel('��������','FontName','����','FontSize',10.5),ylabel('q','FontName','Times New Roman','FontSize',10.5)
    
    %%
    disp([vepq,' ', num2str(fmax)])
    eval(sprintf('save T1GaResults%d',en))
    
end

%���ƾ����Ŵ���������Ӧ�����ߡ�һ��أ����������������Ⱥ��ƽ����Ӧ���������
%Ӧ�������������໥��ͬ����̬����ʾ�㷨�������еú�˳����û�г����𵴣�������ǰ
%���£������Ӧ�ȸ����������ɴ���û�з�������������Ⱥ�Ѿ����졣

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
% xlabel('��������','FontName','����','FontSize',10.5),ylabel('Ve','FontName','Times New Roman','FontSize',10.5)
% figure3= figure('Color',[1 1 1]);
% axes1 = axes('Parent',figure3);
% plot(1:Generation,pmax)
% set(axes1,'FontName','Times New Roman','Layer','top');
% xlabel('��������','FontName','����','FontSize',10.5),ylabel('p','FontName','Times New Roman','FontSize',10.5)
% figure4= figure('Color',[1 1 1]);
% axes1 = axes('Parent',figure4);
% plot(1:Generation,qmax)
% set(axes1,'FontName','Times New Roman','Layer','top');
% xlabel('��������','FontName','����','FontSize',10.5),ylabel('q','FontName','Times New Roman','FontSize',10.5)

toc

end
%% ���¾�Ϊ�ӳ���



%% �ӳ��� ����Ӧ���ʼ���

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

%% �ӳ�������Ⱥ�������,�������ƴ洢Ϊcrossover.m

% function cro=crossover(population,sel,pc,fitness)
%
% BitLength=size(population,2);
%
% if fitness(sel(1))<fitness(sel(2))   % ˭����Ӧ�ȸ߾���˭�ĵĽ������
%             pc=pcrossover(sel(1));
% else
%             pc=pcrossover(sel(2));
% end
%
% pcc=IfCroIfMut(pc);  %���ݽ�����ʾ����Ƿ���н��������1���ǣ�0��񣬸���Խ�ߣ�����Ŀ�����Խ��
%
% if pcc==1
%     chb=round(rand*(BitLength-2))+1;  %��[1,BitLength-1]��Χ���������һ������λ
%     cro(1,:)=[population(sel(1),1:chb) population(sel(2),chb+1:BitLength)];
%     cro(2,:)=[population(sel(2),1:chb) population(sel(1),chb+1:BitLength)];
% else
%     cro(1,:)=population(sel(1),:);
%     cro(2,:)=population(sel(2),:);
% end
% end

% ���Ƚ���
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

%% �ӳ��򣺼�����Ӧ�Ⱥ���, �������ƴ洢Ϊfitnessfun.m

function fitness=fitnessfun(population, params)

popsize=size(population,1);   %��popsize������

% �������м���
parfor i=1:popsize
    
    %     i
    vepq=BintoThr(population(i,:), params);   % �������ƣ�����
    ve=vepq(1);
    p=vepq(2);
    q=vepq(3);
    
    fitness(i)=targetfun(ve,p,q);  %���㺯��ֵ������Ӧ��
    
end
fitness = fitness';
end

%% �����ۼƸ���

function cumsump=cumsumpfun(fitness)

%����ѡ�����
fsum=sum(fitness);              % ��Ⱥ���и������Ӧ��֮��
Pperpopulation=fitness/fsum;    % �����������Ӧ�������и�����Ӧ��֮�͵ı���

%�����ۻ�����
cumsump = cumsum(Pperpopulation);   % �ۼӺͣ�
cumsump=cumsump';

end

%% �ӳ�������Ⱥ����������������ƴ洢Ϊmutation.m������λ���죬ѡȡһλ���б���

% function snnew=mutation(snew,pm)
%
% BitLength=size(snew,2);
% snnew=snew;
% pmm=IfCroIfMut(pm);  %���ݱ�����ʾ����Ƿ���б��������1���ǣ�0���
%
% if pmm==1
%     chb=round(rand*(BitLength-1))+1;  %��[1,BitLength]��Χ���������һ������λ
%     snnew(chb)=abs(snew(chb)-1);   %����һ��0-1,1-0�ķ�ת
% end
% end

% ������
function snnew=mutation(snew,pm)
BitLength=size(snew,2);
snnew=snew;
for i=1:BitLength
    if(rand(1)<pm)
        chb=i;
        snnew(chb)=abs(snew(chb)-1);   %����һ��0-1,1-0�ķ�ת
    end
end
end

%% �ӳ����ж��Ŵ������Ƿ���Ҫ���н�������, �������ƴ洢ΪIfCroIfMut.m�����ȷ�����������Խ�ߣ�����ĸ���ҲԽ��

function pcc=IfCroIfMut(pcorpm)   % ����Ϊ�������

if rand<pcorpm
    pcc=1;
else pcc=0;
end

end

%% �ӳ�������Ⱥѡ�����, �������ƴ洢Ϊselection.m�����̶ķ���

function seln=selection(population,cumsump)

%����Ⱥ��ѡ����������

for i=1:2
    r=rand;  %����һ�������
    prand=cumsump-r;
    j=1;
    while prand(j)<0
        j=j+1;
    end
    seln(i)=j; %ѡ�и�������,�����ѡ������
end
end

%% �ӳ��򣺽�2������ת��Ϊ10������,�������ƴ洢ΪBintoDec.m

function x=BintoDec(population)

BitLength=size(population,2);
x=population(BitLength);

for i=1:BitLength-1
    x=x+population(BitLength-i)*power(2,i);
end
end

%% �ӳ��򣺽�2�������ָ�Ϊ��������ת��Ϊʮ����,�������ƴ洢ΪBintoThr.m

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

ve_1=BintoDec(population(1:num_ve));  %��������ת��Ϊʮ����
p_1=BintoDec(population(num_ve+1:num_ve+num_p));  %��������ת��Ϊʮ����
q_1=BintoDec(population(num_ve+num_p+1:BitLength));  %��������ת��Ϊʮ����

vepq(1)=ve_min+ve_1*(ve_max-ve_min)/(2^num_ve-1);
vepq(2)=p_min+p_1*(p_max-p_min)/(2^num_p-1);
vepq(3)=q_min+q_1*(q_max-q_min)/(2^num_q-1);

end
%�ӳ��򣺶����Ż����ֵ�򼫴�ֵ�������⣬Ŀ�꺯��������Ϊ��Ӧ�Ⱥ���
%�������ƴ洢Ϊtargetfun.m

%% ����Աȶ�

function y=targetfun(ve,p,q) %Ŀ�꺯��

%% �۽�
% ���徲̬����������ÿ�ε��øú���������Ҫ���¼�������
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


%% ����Աȶ�
AveragePower=mean(mean(abs(s_tr_ta).^2));
%AveragePower=mean(abs(s_tr_ta).^2);
% TotalPower=sum(sum(abs(s_tr_ta).^2));

Contrast=sqrt(mean(mean(abs(s_tr_ta).^2-AveragePower).^2))/AveragePower;
%Contrast=sqrt(mean((abs(s_tr_ta).^2-AveragePower).^2))/AveragePower;
%Contrast=sqrt(mean((abs(s_tr_ta).^2-repmat(AveragePower, size(s_tr_ta,1), 1)).^2))/AveragePower;
% Entropy=-sum(sum(abs(s_tr_ta).^2/TotalPower.*log(abs(s_tr_ta).^2/TotalPower)));

y=Contrast;  % �Ŵ��㷨�������ֵ

end


