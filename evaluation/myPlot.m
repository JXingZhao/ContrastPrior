close all; clear; clc;
load('Result.mat');

ss_sm = sort(Smeasure);

[sizePlot,countPlot] = HistPlot(Smeasure,10);
plot(sizePlot,countPlot,'b','LineWidth',1);
hold on;
[sizePlot2,countPlot2] = HistPlot(Fmeasure,10);
plot(sizePlot2,countPlot2,'y','LineWidth',1);


function [sizePlot,countPlot] = HistPlot(str, number)
[count,sizeRatio] = histcounts(str,number);

count = count/sum(count) * 100;
sizeRatio(:,1) = [];
%plot(sizeRatio,count);

sizePlot = sizeRatio(1,1):0.01:sizeRatio(length(sizeRatio));
countPlot = spline(sizeRatio,count,sizePlot);

end
