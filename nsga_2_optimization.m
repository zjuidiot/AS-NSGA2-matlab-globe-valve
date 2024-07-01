function nsga_2_optimization

np=200;%种群数量
gen=5;%迭代次数
M=2;%优化目标数
V=3;%参数量
min_range=zeros(1,V);%参数下限
max_range=ones(1,V);%参数上限

min=min_range;
max=max_range;
K=M+V;%参数量+目标量，即数据总量

for i=1:N

chromosome = non_domination_sort_mod(chromosome, M, V);%对初始化种群进行非支配快速排序和拥挤度计算
