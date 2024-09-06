# AS-NSGA2-matlab

my first repo on github

[TOC]

## 中文

 介绍

Matlab实现响应面与遗传算法的多目标优化。代码是3个参数对应2个优化目标，采用二阶回归拟合，然后用自适应非支配排序遗传算法优化一下。感觉代码里面注释已经写得挺明确了。 讲一下主要功能吧，首先是用fitnlm进行了一个拟合，这个拆出来其实就是响应面。 然后是自适应非支配排序遗传算法。自适应就是交叉率和变异率会根据个体的y值（或者说适应度）变化，避免陷入局部最优，非支配就是由个体间多个目标都进行一下对比，形成支配关系进而排序。



 更新记录

更新1：增加了绘图与参数敏感性分析的内容

更新2：初始的随机生成种群与变异部分的随机都换成了拉丁超立方抽样，这样会更均匀一些，与原先的对比，确实收敛速度肉眼可见得更快。 x1min、x2min这些乱七八糟的变量未免太丑了，去掉，换成了lb和ub。 增加了归一化，取最优的时候直接加起来就行。 迭代的时候增加了一张均值随着迭代次数增加的曲线图，更直观地看到迭代是否收敛。



使用说明

把excel里面的数据换成自己的数据然后运行ANSGA2就行了，这里面的试验表是用matlab的bbdesign生成的，换成别的应该也行。



 声明

望轻喷

[1]马哲辉,吴斌彬,李文庆,等.高温高压Y型截止阀组内流特性及流道结构改进研究[J/OL].机电工程,1-11[2024-09-06].http://kns.cnki.net/kcms/detail/33.1088.TH.20240810.1224.002.html.

## English

Introduction

​	This code uses Matlab to achieve muti-objective optimization of response surface and NSGA2.Three parameters correspond to two optimization objectives, using second-order regression fitting, and then optimizing using adaptive non dominated sorting genetic algorithm. 

​	 Let's talk about the main functions. Firstly, we used a function called fitnlm to perform a fitting, which is actually the response surface when disassembled. Then comes the adaptive non dominated sorting genetic algorithm. Adaptation means that the crossover rate and mutation rate will change based on the individual's y-value (or fitness) to avoid getting stuck in local optima. Non dominance means comparing multiple objectives between individuals to form a dominance relationship and then ranking them.

​	The original data is in the Excel



Update

update1: 

Added content on drawing and parameter sensitivity anslysis.

update2:
The initial randomly generated population and the random variation part were replaced with Latin hypercube sampling, which would be more uniform. Compared with the original, the convergence speed is indeed faster and visible to the naked eye.



statement

The author’s code ability is weak. He hopes everyone will not criticize him harshly.
