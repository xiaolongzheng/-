function [ solution, obj ] = GA(func,pop_size,gens,pm,pc, params)
%UNTITLED10 此处显示有关此函数的摘要
%   此处显示详细说明

% func.dims = 3
% func.ubounds = [];  %
% func.dbounds = [];

Population = rand(func.dims,pop_size)'.*repmat(func.ubounds-func.dbounds, pop_size, 1)+repmat(func.dbounds, pop_size, 1);%
Fitness = inf*ones(pop_size,1);% the fitness 

Gbest_position = zeros(1,func.dims); 
Gbest_fitness = inf*ones(1,1);%the fitness
Fitness = fitnessfun(Population, params);
%rank and calculate the Pbest_fitness 
[ y, index ] = sort( Fitness,1,'descend' );

%save the current optimal value
Gbest_fitness(1) = y(1);
Gbest_position(1,:) = Population(index(1),:);

Generation=1;

while Generation<=gens
    Generation = Generation + 1;
    
    % Select individuals from population
%     iSeParents = sus(Fitness,pop_size);
%     iSeParents = randperm(pop_size,pop_size);
    
    num = 0;iSeParents=[];
    while num < pop_size,
        iSeParents = [iSeParents, RouletteWheelSelection(Fitness)];
        num = num + 1;
    end
    
    Offspring = zeros(pop_size, func.dims);
    OffFitness = [];
    indorder = 1:1:pop_size;
    for i=1:2:pop_size
        p1 = iSeParents(indorder(i));
        p2 = iSeParents(indorder(i+1));
        
        mu = 2;
        mum = 5;
        if rand(1)<pc
            [Offspring(i,:),Offspring(i+1,:)] = SBC(Population(p1,:),Population(p2,:),mu);
            if rand(1)<pm
                Offspring(i,:) = PolynomialMutation( Offspring(i,:),func.ubounds,func.dbounds,mum);
                Offspring(i+1,:) = PolynomialMutation( Offspring(i+1,:),func.ubounds,func.dbounds,mum);
            end
        else
            Offspring(i,:) = PolynomialMutation( Population(p1,:),func.ubounds,func.dbounds,mum);
            Offspring(i+1,:) = PolynomialMutation( Population(p2,:),func.ubounds,func.dbounds,mum);
        end
    end
    %set limitation to the Offspring
    illcounts = 0;
    for i=1:pop_size
        for j=1:func.dims
            if Offspring(i,j)>func.ubounds(j)
    %             Offspring(i,j)=func.ubounds(j);
                Offspring(i,j)=rand(1)*(func.ubounds(j)-func.dbounds(j))+func.dbounds(j);
                illcounts = illcounts + 1;
            end
            if Offspring(i,j)<func.dbounds(j)
    %             Offspring(i,j)=func.dbounds(j);
                Offspring(i,j)=rand(1)*(func.ubounds(j)-func.dbounds(j))+func.dbounds(j);
                illcounts = illcounts + 1;
            end
        end
    end
    OffFitness = fitnessfun(Offspring, params);
    disp(illcounts)
    
    selection_process = 'elitist';
    if strcmp(selection_process,'elitist')
        intpopulation = [Population;Offspring];
        intFitness = [Fitness;OffFitness];
        
        [xxx,y]=sort(intFitness,1,'descend' );
        Population = intpopulation(y(1:pop_size),:);
        Fitness = intFitness(y(1:pop_size),:);
    elseif strcmp(selection_process,'sus')
        intpopulation = [Population;Offspring];
        intFitness = [Fitness;OffFitness];

        y = sus(intFitness, pop_size);
        Population = intpopulation(y(1:pop_size),:);
        Fitness = intFitness(y(1:pop_size),:);
        
        % keep elites
        [fmax,nmax]=max(intFitness);
        [fmax1,nmax1]=max(Fitness);
        if fmax>fmax1
            Population(1, :) = intpopulation(nmax, :);
            Fitness(1, 1) = intFitness(nmax, :);
        end

    elseif strcmp(selection_process,'rws')
        intpopulation = [Population;Offspring];
        intFitness = [Fitness;OffFitness];
         
        num = 0;y=[];
        while num < pop_size,
            y = [y, RouletteWheelSelection(intFitness)];
            num = num + 1;
        end
        
        Population = intpopulation(y(1:pop_size),:);
        Fitness = intFitness(y(1:pop_size),:);
        
        % keep elites
        [fmax,nmax]=max(intFitness);
        [fmax1,nmax1]=max(Fitness);
        if fmax>fmax1
            Population(1, :) = intpopulation(nmax, :);
            Fitness(1, 1) = intFitness(nmax, :);
        end
    else
        [fmin,nmin]=min(Fitness);
        [fmax,nmax]=max(Fitness);
        bestvepq = Population(nmax, :);
        
        Population = Offspring;
        Fitness = OffFitness;
        
        %最优个体保存
        Population(nmin,:)=bestvepq;  % 用上一代的最优个体，替换新一代的最差个体
        Fitness(nmin)=fmax;
    end
    [fmax,nmax]=max(Fitness);
    disp([num2str(Generation),' ', num2str(fmax)])
end
[fmax,nmax]=max(Fitness);
solution = Population(nmax,:);
obj = fmax;
end

