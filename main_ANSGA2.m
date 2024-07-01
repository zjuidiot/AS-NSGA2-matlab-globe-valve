%% 遗传算法主程序
% 这是1.1版本，主要增加了拉丁超立方抽样
clear
clc
close all

npop=200;%种群个数

nvar=3;%参数量
maxit=200;%迭代次数

pc1=0.8;%交叉率
pc2=0.6;
pm1=0.2;%变异率
pm2=0.1;

rng(0);

template.x=[];
template.y=[];
template.dominationset=[];%支配集，指个体能够支配的其他个体所对应的下标
template.dominated=[];%被支配数，指个体在自然选择中被支配的次数
template.rank=[];%等级，指个体在自然选择中所占据的生态位
template.cd=[];%拥挤度
 
pop=repmat(template,npop,1);
%% 二阶回归拟合
data = readmatrix('excel.xlsx','Range','A2:E18');
x= readmatrix('excel.xlsx','Range','A2:C18');
Y1=data(:,4);
Y2=data(:,5);
[y1,ps1] = mapminmax(Y1',0,1);%归一化
[y2,ps2] = mapminmax(Y2',0,1);
lb = [min(x(:,1)),min(x(:,2)),min(x(:,3))];%下界
ub = [max(x(:,1)),max(x(:,2)),max(x(:,3))];%上界

%开始拟合
y1_b=@(b,x) b(1)*x(:,1).^2+b(2)*x(:,2).^2+b(3)*x(:,3).^2+b(4)*x(:,1).*x(:,2)+b(5)*x(:,1).*x(:,3)+...
b(6)*x(:,2).*x(:,3)+b(7)*x(:,1)+b(8)*x(:,2)+b(9)*x(:,3)+b(10);%给出拟合公式
c=ones(1,10);%初始化系数向量
y1_bbde=fitnlm(x,y1,y1_b,c);%非线性函数拟合
c_1=y1_bbde.Coefficients{:,{'Estimate'}};%获取拟合系数

y2_b=@(b1,x) b1(1)*x(:,1).^2+b1(2)*x(:,2).^2+b1(3)*x(:,3).^2+b1(4)*x(:,1).*x(:,2)+b1(5)*x(:,1).*x(:,3)+...
b1(6)*x(:,2).*x(:,3)+b1(7)*x(:,1)+b1(8)*x(:,2)+b1(9)*x(:,3)+b1(10);%给出拟合公式
c1 = ones(1,10);
y2_bbde=fitnlm(x,y2,y2_b,c1);
c_2=y2_bbde.Coefficients{:,{'Estimate'}};
%% 交叉变异迭代
lhs_rand = lhsdesign(npop,3);
for i=1:npop
    %随机生成参数组合
    pop(i).x = lb+lhs_rand(i,:).*(ub-lb);
    pop(i).y=respond(pop(i).x,y1_b,c_1,y2_b,c_2);%根据参数组合得到对应的y值
end

mean_iteration = [];%迭代过程记录
iteration = [];
n_best = 0;%记录最优值是同一个的次数
y_best = zeros(1,maxit+1);%记录最优解
%for it=1:maxit
it = 1;
while n_best <= maxit/4 & length(iteration) <= maxit
    npc=1;
    popc=repmat(template,npop/2,2);
    fall = [pop.y];%fall存储所有的y值
    fall = reshape(fall,[2, npop]);%第一行存储所有的y1，第二行存储所有的y2
    fmax =2; %这一代中综合y值最高的个体，以其作为归一化的上限
    fave =mean(fall(1,:))./max(fall(1,:))+mean(fall(2,:))./max(fall(2,:)); %这一代中综合y值的平均数
    %% 交叉
    for i=1:npop/2
        ind=randperm(npop,2);%选择交叉个体

        fcross1=pop(ind(1)).y(1)./max(fall(1,:))+pop(ind(1)).y(2)./max(fall(2,:));
        fcross2=pop(ind(2)).y(1)./max(fall(1,:))+pop(ind(2)).y(2)./max(fall(2,:));
        fcross = max([fcross1 fcross2]);
        pc=pc2;
        if(fcross>fave)
            pc = pc1-(pc1-pc2).*(fcross-fave)./(fmax-fave);
        end

        value = rand();
        if(value<=pc)
        [popc(npc,1).x,popc(npc,2).x]=Cross(pop(ind(1)).x,pop(ind(2)).x);%交叉得到新的参数组合
        popc(npc,1).y=respond(popc(npc,1).x,y1_b,c_1,y2_b,c_2);  
        popc(npc,2).y=respond(popc(npc,2).x,y1_b,c_1,y2_b,c_2);
        npc=npc+1;
        end
    end

    npc = npc-1;
    popc(npc+1:npop/2,:)=[];

    %% 变异
    npm=1;
    popm=repmat(template,npop,1);
    i_mutate = [];%存储变异个体的编号
    for j=1:npop%
        ind=randperm(npop,1);%选择变异个体
        fmutate=pop(ind(1)).y(1)./max(fall(1,:))+pop(ind(1)).y(2)./max(fall(2,:));
        pm=pm2;
        if(fmutate>fave)
            pm = pm1-(pm1-pm2).*(fmutate-fave)./(fmax-fave);
        end
        value=rand();
        if(value<=pm)
            i_mutate = [i_mutate ind];%存储变异个体
        end
    end
    mutate_rand = lhsdesign(length(i_mutate),1);%根据变异个体数量进行拉丁超立方抽样
    for j = i_mutate
        popm(npm,1).x=Mutate(pop(j).x,lb,ub,mutate_rand(npm));
        popm(npm,1).y=respond(popm(npm).x,y1_b,c_1,y2_b,c_2);
        npm=npm+1;
    end

    npm=npm-1;
    popm(npm+1:npop)=[];
    popc=popc(:);

    newpop=[pop;popc;popm];%新种群
    [newpop,F]=Non_dominate_sort(newpop);
    newpop=Crowd(newpop,F);%计算拥挤度
    newpop=nsga2Sort(newpop);%根据生态等级与拥挤度进行排序
    pop=newpop(1:npop);%只取前npop个个体存活至下一次迭代

    y1=zeros(1,npop);
    y2=zeros(1,npop);
    ys=[pop.y];
    for j=1:npop
        y1(j)=ys(2*j-1);
        y2(j)=ys(2*j);
    end

    %% 绘图
    mean_iteration = [mean_iteration,mean(sum(fall,1))];%这一带的综合值
    iteration = [iteration,it];

    subplot(2,1,1)
    plot(y1,y2,'r*');
    numtitle=num2str(it);
    title('迭代次数=',numtitle);
    xlabel('y1');
    ylabel('y2');
    subplot(2,1,2)
    plot(iteration,mean_iteration);
    ylim([1 2.17]);
    xlabel('iteration');
    ylabel('y1+y2');
    set(gcf,'color','white');
    pause(0.001);
    %frame=getframe(gcf);
    %writeVideo(v,frame);%记录迭代过程

    %% 判断最优
    y_sum = 1*fall(1,:)+1*fall(2,:);%
    y_best(it+1) = max(y_sum);
    if y_best(it+1) == y_best(it)
        n_best = n_best+1;
    else
        n_best = 0;%最优解不一样，则最优解重复次数归0
    end
    it = it+1;
end
disp(['第',num2str(find(y_sum==y_best(it))),...
            '个个体最优，y_sum=',num2str(y_best(it))]);
%% 绘制响应面
n1=31;%网格数量
n2=11;
n3=51;
A1=max(x(:,1))-min(x(:,1));
A2=max(x(:,2))-min(x(:,2));
A3=max(x(:,3))-min(x(:,3));%各项参数的幅值
k1=A1./(n1-1);
k2=A2./(n2-1);
k3=A3./(n3-1);

AB=ones(1,3);
BC=ones(1,3);
AC=ones(1,3);

vAB=zeros(n2,n1);
qAB=zeros(n2,n1);
AB(1,3)=(max(x(:,3))+min(x(:,3)))./2;
for i=1:n1
    for j=1:n2
        AB(1,1)=min(x(:,1))+k1.*(i-1);
        AB(1,2)=min(x(:,2))+k2.*(j-1);
        vAB(j,i)=y2_b(c_2,AB);
        qAB(j,i)=y1_b(c_1,AB);
    end
end

vBC=zeros(n3,n2);
qBC=zeros(n3,n2);
BC(1,1)=(max(x(:,1))+min(x(:,1)))./2;
for i=1:n2
    for j=1:n3
        BC(1,2)=min(x(:,2))+k2.*(i-1);
        BC(1,3)=min(x(:,3))+k3.*(j-1);
        vBC(j,i)=y2_b(c_2,BC);
        qBC(j,i)=y1_b(c_1,BC);
    end
end

vAC=zeros(n3,n1);
qAC=zeros(n3,n1);
AC(1,2)=(max(x(:,2))+min(x(:,2)))./2;
for i=1:n1
    for j=1:n3
        AC(1,1)=min(x(:,1))+k1.*(i-1);
        AC(1,3)=min(x(:,3))+k3.*(j-1);
        vAC(j,i)=y2_b(c_2,AC);
        qAC(j,i)=y1_b(c_1,AC);
    end
end
ABC=ones(1,3);
y1_all=zeros(n1,n2,n3);
y2_all=zeros(n1,n2,n3);
for i=1:n1
    for j=1:n2
        for k=1:n3
            ABC(1)=min(x(:,1))+k1.*(i-1);
            ABC(2)=min(x(:,2))+k2.*(j-1);
            ABC(3)=min(x(:,3))+k3.*(k-1);
            y1_all(i,j,k)=y1_b(c_1,ABC);
            y2_all(i,j,k)=y2_b(c_2,ABC);
        end
    end
end
figure('name','响应面分析3D图')

subplot(3,2,1)
surf(linspace(2,6,n1),linspace(1,3,n2),vAB);
shading interp; 
colorbar; colormap(jet);
title('x1-x2-y1');
xlabel('x1');
ylabel('x2');
zlabel('y1');
zlim([min(y2_all,[],'all') max(y2_all,[],'all')]);
clim([min(y2_all,[],'all') max(y2_all,[],'all')]);
set(gcf,'unit','centimeters','position',[10 5 20 20]);%设置图片的位置与大小
set(gcf, 'Color', 'w');

subplot(3,2,2)
surf(linspace(2,6,n1),linspace(1,3,n2),qAB);
shading interp; 
colorbar; colormap(jet);
title('x1-x2-y2');
xlabel('x1');
ylabel('x2');
zlabel('y2');
zlim([min(y1_all,[],'all') max(y1_all,[],'all')]);
clim([min(y1_all,[],'all') max(y1_all,[],'all')]);

subplot(3,2,3)
surf(linspace(1,3,n2),linspace(15,30,n3),vBC);
shading interp; 
colorbar; colormap(jet);
title('x2-x3-y1');
xlabel('x2');
ylabel('x3');
zlabel('y1');
zlim([min(y2_all,[],'all') max(y2_all,[],'all')]);
clim([min(y2_all,[],'all') max(y2_all,[],'all')]);

subplot(3,2,4)
surf(linspace(1,3,n2),linspace(15,30,n3),qBC);
shading interp; 
colorbar; colormap(jet);
title('x2-x3-y2');
xlabel('x2');
ylabel('y3');
zlabel('y2');
zlim([min(y1_all,[],'all') max(y1_all,[],'all')]);
clim([min(y1_all,[],'all') max(y1_all,[],'all')]);

subplot(3,2,5)
surf(linspace(2,6,n1),linspace(15,30,n3),vAC);
shading interp; 
colorbar; colormap(jet);
title('x1-x3-y1');
xlabel('x1');
ylabel('x3');
zlabel('y1');
zlim([min(y2_all,[],'all') max(y2_all,[],'all')]);
clim([min(y2_all,[],'all') max(y2_all,[],'all')]);

subplot(3,2,6)
surf(linspace(2,6,n1),linspace(15,30,n3),qAC);
shading interp; 
colorbar; colormap(jet);
title('x1-x3-y2');
xlabel('x1');
ylabel('x3');
zlabel('y2');
zlim([min(y1_all,[],'all') max(y1_all,[],'all')]);
clim([min(y1_all,[],'all') max(y1_all,[],'all')]);
%% 参数敏感性分析
M = nvar*2;
ns = 100;%采样数
pointset= sobolset(M);
R = net(pointset,ns);%生成样本集
A = R(:,1:nvar);
B = R(:,nvar+1:end);
SAB = zeros(ns,nvar,nvar);
for i=1:nvar
    A(:,i) = min(x(:,i))+A(:,i).*(max(x(:,i))-min(x(:,i)));%将0~1之间的样本值映射到实际值的范围内
    B(:,i) = min(x(:,i))+B(:,i).*(max(x(:,i))-min(x(:,i)));
end
for i=1:1:nvar
    tempA = A;
    tempA(:,i) = B(:,i);
    SAB(:,:,i) = tempA;
end
Y1A = zeros(nvar,1);Y1B = zeros(nvar,1);
Y1AB = zeros(ns,nvar);
for i=1:1:nvar
Y1A = y2_b(c_2,A);
Y1B = y2_b(c_2,B);
end
for i=1:nvar
Y1AB(:,i) = y2_b(c_2,SAB(:,:,i));
end

VarEX = zeros(nvar,1);%一阶影响指数分子
VarY = var([Y1A;Y1B],1);%分母
S1 = zeros(nvar,1);%一阶影响指数
EVarX = zeros(nvar,1);%全局影响指数分子
ST = zeros(nvar,1);%全局影响指数
for i=1:nvar
    for j =1:ns
        VarEX(i) = VarEX(i)+Y1B(j).*(Y1AB(j,i)-Y1A(j))./ns;
        EVarX(i) = EVarX(i)+((Y1A(j)-Y1AB(j,i)).^2)./(2.*ns);
    end
end
S1 = VarEX./VarY;
ST = EVarX./VarY;
S = [S1 ST];
figure
bar(S);

%% 取最优
% y1_all=zeros(1,npop);
% y2_all=zeros(1,npop);
% for i=1:npop
%     y1_all(i)=pop(i).y(1);
%     y2_all(i)=pop(i).y(2);
% end
% y_sum = y1_all +y2_all;
% for i=1:npop
%     if(y_sum(i)==max(y_sum))
%         disp(['第',num2str(i),'个个体最优，y_sum=',num2str(y_sum(i))]);
%         break;
%     end
% end
