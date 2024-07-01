function pop=nsga2Sort(pop)
    [~, ind_cd] = sort([pop.cd], 'descend');%先根据拥挤度进行排序
    pop = pop(ind_cd);
    [~,ind_rank] = sort([pop.rank],'ascend');%再根据支配等级进行排序
    pop = pop(ind_rank);
end