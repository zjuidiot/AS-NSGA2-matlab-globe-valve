function [pop,F]=Non_dominate_sort(pop)
    n=size(pop);
    num=n(1);%参与比较的个体数
    for i=1:num
    pop(i).dominationset=[];
    pop(i).dominated=0;
    end
    F{1}=[];

    for i=1:num
        for j=i+1:num
            if all(pop(i).y<=pop(j).y) && any(pop(i).y<pop(j).y)
                pop(j).dominationset=[pop(j).dominationset, i];%加上被支配个体的下标
                pop(i).dominated=pop(i).dominated+1;%j被支配了一次
            elseif all(pop(j).y<=pop(i).y) && any(pop(j).y<pop(i).y)
                pop(i).dominationset=[pop(i).dominationset, j];
                pop(j).dominated=pop(j).dominated+1;
            end
        end
        
        if pop(i).dominated==0
            F{1}=[F{1}, i];
            pop(i).rank=1;%没有被支配过，生态位最高
        end
    end
    %划分生态位
        k=1;
        while true
            result=[];
            for i=F{k}%遍历生态位排序为k的所有个体
                for j =pop(i).dominationset%遍历所有被pop(i)所支配的个体
                    pop(j).dominated=pop(j).dominated-1;
                    if pop(j).dominated==0
                        result=[result,j];
                        pop(j).rank=k+1;
                    end
                end
            end
            if isempty(result)
                break;
            else
                k=k+1;
                F{k}=result;
            end
        end

end