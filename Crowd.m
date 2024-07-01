function pop=Crowd(pop,F);
    n=numel(F);%生态等级的数量
    for i=1:n
        ys=[];
        for k=F{i}
            ys=[ys;pop(k).y];%把该等级中的个体所对应的y值取出
        end
        nofy=size(ys,2);%优化目标数量
        nk=numel(F{i}); %该等级中的个体数
        crow=zeros(nk,nofy);

        for j=1:nofy
            [y,ind]=sort(ys(:,j),'ascend');
            crow(ind(1),j)=inf;
            crow(ind(nk),j)=inf;%边界点的拥挤度为无穷大
            for k=2:nk-1
                crow(ind(k),j)=abs((y(k+1)-y(k-1))./(y(1)-y(nk)));
            end
        end
        for in=1:nk
            pop(F{i}(in)).cd=sum(crow(in,:));%总的拥挤度等于各项目标值计算得到的拥挤度的总和
        end
    end
end