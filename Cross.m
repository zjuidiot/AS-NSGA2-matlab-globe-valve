function [popc1 popc2]=Cross(pop1,pop2)

nx=numel(pop1);%读取参数个数
popc1=pop1;
popc2=pop2;

exchange=randperm(nx,1);%生成交叉项

popc1(1,exchange)=pop2(1,exchange);
popc2(1,exchange)=pop1(1,exchange);
end