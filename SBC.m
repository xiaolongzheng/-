function [ Child1,Child2 ] = SBC( Parent1,Parent2,eta )
%SBC Simulate Binary Crossover.
%   Parent1/2: parent for crossover.
%   eta :distribution index.

dim = size(Parent1,2);
           
u = rand(1,dim);
cf = zeros(1,dim);
cf(u<=0.5)=(2*u(u<=0.5)).^(1/(eta+1));
cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(eta+1));
Child1 = 0.5*((1+cf) .* Parent1 + (1-cf) .* Parent2);
Child2 = 0.5 *((1-cf) .* Parent1 +(1+cf) .* Parent2);
% Child1(Child1>1) = 1;
% Child1(Child1<0) = 0;
% Child2(Child2>1) = 1;
% Child2(Child2<0) = 0;
end

