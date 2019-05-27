clear all; close all; clc;

%set your dataset path and saliency map result path.
%Models = {'PDNet','ACSD','DES','LHM','GP','SE','CTMF','DF','MDSF','CDCP','PCF'};
Models = {'CDB'};
%Models = {'ACSD','DES','LHM','GP','MC','LBE','SE','DF','MDSF','CDCP','NLDF','AMU','PCF','CDB','PDNet'};
%method_col = { 'r', 'g', 'b', 'm', 'c', '--m', '--g', 'k', 'c--' };
modelNum = length(Models);

%Thresholds = 1:-1/255:0;
Thresholds = 1:1:255;
for m = 1:modelNum
    
    model = Models{m};
    %Datasets = {'GIT','NJU2000','LFSD','RGBD135','NLPR','SSD100'};
    %Datasets = {'SSB','GIT','LFSD','RGBD135','SSD100'};
    %Datasets = {'NLPR'};
    Datasets = {'rgbd135/'};
    datasetNum = length(Datasets);
    
    fprintf('Processing %s model\n',model);
    for d = 1:datasetNum
        
        dataset = Datasets{d};
        
        fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
        gtPath = ['./Dataset/' dataset '/'];
        salPath = ['./Model Result/' dataset '/'];

        



        %obtain the total number of image (ground-truth)
        imgFiles = dir([salPath '*.png']);
        imgNUM = length(imgFiles)
        
        %evaluation score initilization.
        Smeasure=zeros(1,imgNUM);
        Fmeasure=zeros(1,imgNUM);
        threshold_Fmeasure  = zeros(imgNUM,length(Thresholds));
        MAE=zeros(1,imgNUM);
        
        tic;
        for i = 1:imgNUM
            i
            %fprintf('Evaluating: %d/%d\n',i,imgNUM);
            
            name =  imgFiles(i).name;
            
            %load gt
            gt = imread([gtPath name]);
            
            
            if numel(size(gt))>2
                gt = rgb2gray(gt);
            end
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            %load salency
            sal  = imread([salPath name]);
            
            %check size
            %if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
            %    sal = imresize(sal,size(gt));
            %    imwrite(sal,[salPath name]);
            %    fprintf('Error occurs in the path: %s!!!\n', [salPath name]);
                
            %end
            
            sal = double(sal(:,:,1));
            %max(max(sal))
            %normalize sal to [0, 1]
            %sal = reshape(mapminmax(sal(:)',0,1),size(sal));
%             Smeasure(i) = StructureMeasure(sal,logical(gt));
%             
%             % Using the 2 times of average of sal map as the threshold. 
%             threshold =  2* mean(sal(:)) ;
%             temp = Fmeasure_calu(sal,double(gt),size(gt),threshold);
%             Fmeasure(i) = temp(3);
            
            for t = 1:length(Thresholds)
                threshold = Thresholds(t);
                temp = Fmeasure_calu(sal,double(gt),size(gt),threshold);
                threshold_Fmeasure(i,t) = temp(3);
            end
            
            MAE(i) = mean2(abs(double(logical(gt)) - sal));
            
        end
        
        toc;
        
%         Sm = mean2(Smeasure);
%         Fm = mean2(Fmeasure);
        column_F = mean(threshold_Fmeasure,1);
        meanF = mean(column_F);
        maxF = max(column_F);
        mae = mean2(MAE);
        
        %fprintf('(%s Dataset)Smeasure: %.3f; fixFm %.3f; meanFmeasure:%.3f; maxFmeasure: %.3f; MAE: %.3f.\n',dataset,Sm, Fm, meanF, maxF, mae);
        fprintf('(%s Dataset)meanFmeasure:%.3f; maxFmeasure: %.3f; MAE: %.3f.\n',dataset,meanF, maxF, mae);
        
        
    end
    
    %     [sizePlot,countPlot] = HistPlot(Smeasure,10);
    %     plot(sizePlot,countPlot,method_col{m},'LineWidth',1);
    %     grid on;
    %     hold on;
end

% function [sizePlot,countPlot] = HistPlot(str, number)
% [count,sizeRatio] = histcounts(str,number);
% 
% count = count/sum(count) * 100;
% sizeRatio(:,1) = [];
% %plot(sizeRatio,count);
% 
% sizePlot = sizeRatio(1,1):0.01:sizeRatio(length(sizeRatio));
% countPlot = spline(sizeRatio,count,sizePlot);
% 
% end


