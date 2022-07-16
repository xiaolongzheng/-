function [N_Individual] = PolynomialMutation(Individual,bu,bd,eta_m)
%PolynomialMutation Polynomial Mutation Operator.
%   Individual:[dim1,dim2,dim3,...,dim k];
%   N_Individual:new individual.
%   bu:upbound of individual.
%   bd:lowbound of individual.
%   eta_m:distribution index.
%--------------------------------------------------------------------------
[N,Dim] = size(Individual);
N_Individual = zeros(N,Dim);
for j = 1:Dim
    y = Individual(1,j);
    yd = bd(1,j);
    yu = bu(1,j);
    if y > yd
       if (y-yd)<(yu-y)
           delta = (y-yd)/(yu-yd);
       else
           delta = (yu-y)/(yu-yd);
       end
       r2 = rand;
       indi = 1/(eta_m+1);
       if r2 <= 0.5
          xy = 1-delta;
          val = 2*r2 + (1-2*r2)*(xy^(eta_m+1));
          deltaq = val^indi - 1;
        else
           xy = 1-delta;
           val = 2*(1-r2) + 2*(r2-0.5)*(xy^(eta_m+1));
           deltaq = 1 - val^indi;
        end
        y = y + deltaq*(yu-yd);
        N_Individual(1,j) = min(y,yu);
        N_Individual(1,j) = max(y,yd);
   else%y <= yd
        N_Individual(1,j) = rand(1) * (yu-yd) + yd;
   end
end
end

