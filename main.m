load Target1DataNoNoise 

clear s_fr_fa Rc lambda c fc Kr T_R T_A Tp tac Ta
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
func.dims = 3;
func.ubounds = [ve_max, p_max, q_max];  % 
func.dbounds = [ve_min, p_min, q_min];  % 

pc=0.8;
pm=0.1;
gens = 100;
pop_size = 32;
[ solution, obj ] = GA(func,pop_size,gens,pm,pc, params);
disp(solution)
disp(obj)