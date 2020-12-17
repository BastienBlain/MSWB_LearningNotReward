function  [] = MoodTrackLearning_choice_doPlots(allChoiceData)
% Bastien Blain, bastien.blain@gmail.com
% February, 2019

% set visual parameters
names_models  = {'Multiplicative, PPE','Multiplicative, RPE','Multiplicative, RPE Power'...
    'Additive, PPE','Additive, RPE','Additive, RPE Power',...
    'Additive relative, PPE','Additive relative, RPE','Additive relative, RPE Power'
    };
names_context = {'Stable','Volatile'};
lcolor        = [0 0.45 0.75;
                 1 0.45  0];

%% Figure 2
%% 2A choice accuracy
bestEVChoiceall{1} = [allChoiceData{1}.bestEVChoice{1};allChoiceData{2}.bestEVChoice{2}];
bestEVChoiceall{2} = [allChoiceData{1}.bestEVChoice{2};allChoiceData{2}.bestEVChoice{1}];
for iModel=1:4
bestEVPredall{iModel,1} = [allChoiceData{1}.bestEVPred{1,iModel};allChoiceData{2}.bestEVPred{2,iModel}];
bestEVPredall{iModel,2} = [allChoiceData{1}.bestEVPred{2,iModel};allChoiceData{2}.bestEVPred{1,iModel}];
end
% data source 1
xlswrite('eLife-fig2-dataSource1',double(bestEVChoiceall{1}),'ChosenHighEVStable')
xlswrite('eLife-fig2-dataSource1',double(bestEVChoiceall{2}),'ChosenHighEVVolatile')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{1,1}),'AdditivePredStable')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{1,2}),'AdditivePredVolatile')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{2,1}),'MultiplicativePredStable')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{2,2}),'MultiplicativePredVolatile')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{3,1}),'ProbabilityPredStable')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{3,2}),'ProbabilityPredVolatile')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{4,1}),'RewardPredStable')
xlswrite('eLife-fig2-dataSource1',double(bestEVPredall{4,2}),'RewardPredVolatile')
for icontext=1:2    
    figure('rend','painters','pos',[10 10 550 500]);hold on;
    % get the choice data and the model prediction
    X = bestEVChoiceall{icontext};
    Y1 = bestEVPredall{1,icontext};
    Y2 = bestEVPredall{2,icontext};
    Y3 = bestEVPredall{3,icontext};
    Y4 = bestEVPredall{4,icontext};
    % plot both data & pred
    plot(mean(X),'Color', lcolor(icontext,:),'LineWIdth',3)
    plot(mean(Y1),':','Color',[0.5 0 0.5],'LineWIdth',3)
    plot(mean(Y2),':','Color',[0.4 0.4 0.4],'LineWIdth',3)
    plot(mean(Y3),':','Color',[0.6 0.6 0.6],'LineWIdth',3)
    plot(mean(Y4),':','Color',[0.8 0.8 0.8],'LineWIdth',3)
    % add the relevant legend
    if icontext==1
        legend({'Choice (stable)','Additive model','Multiplicative model','Probability only model','Reward magnitude only model'},...
            'Interpreter','latex','location','South','AutoUpdate','off')
    else
        legend({'Choice (volatile)','Additive model','Multiplicative model','Probability only model','Reward magnitude only model'},...
            'Interpreter','latex','location','South','AutoUpdate','off')
    end
    legend boxoff
    % add SE shaded area
    jbfill(1:80,mean(X)+std((X))./sqrt(size(X,1)),mean(X)-std((X))./sqrt(size(X,1)),...
        lcolor(icontext,:),lcolor(icontext,:),0,0.3)
    % add the switch lines in the volatile environment
    if icontext==2
        line([20.5 20.5],[0 1],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
        line([40.5 40.5],[0 1],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
        line([60.5 60.5],[0 1],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
    end
    % set limits, labels & title
    ylim([0 1])
    xlim([1 80])
    ylabel('Proportion of chosen highest EV')
    xlabel('Trial number')
    set(gca,'Ytick',0:0.25:1,'FontSize',18,'LineWidth',3)
    axis square
end
%% 2B Model agnostic signature of learning rate
meanchoice(:,[1:4]) = [allChoiceData{1}.meanchoice{1};allChoiceData{2}.meanchoice{2}];
meanchoice(:,[5:8]) = [allChoiceData{1}.meanchoice{2};allChoiceData{2}.meanchoice{1}];
for imodel=1:4
    meanprob{imodel}(:,1:4) = [allChoiceData{1}.meanprob{1,imodel};allChoiceData{2}.meanprob{2,imodel}];
    meanprob{imodel}(:,5:8) = [allChoiceData{1}.meanprob{2,imodel};allChoiceData{2}.meanprob{1,imodel}];
end
% data source 2
header = {'Stable_HighPwin_winstay','Stable_HighPwin_losestay','Stable_LowPwin_winstay','Stable_LowPwin_losestay',...
          'Volatile_HighPwin_winstay','Volatile_HighPwin_losestay','Volatile_LowPwin_winstay)','Volatile_LowPwin_losestay'};
xlswrite('eLife-fig2-dataSource1',[header;num2cell(meanchoice)],'choiceProportion')
xlswrite('eLife-fig2-dataSource1',[header;num2cell(meanprob{1})],'AdditivePredProportion')
xlswrite('eLife-fig2-dataSource1',[header;num2cell(meanprob{2})],'MultiplicativePredProportion')
xlswrite('eLife-fig2-dataSource1',[header;num2cell(meanprob{3})],'ProbabilityPredProportion')
xlswrite('eLife-fig2-dataSource1',[header;num2cell(meanprob{4})],'RewardPredProportion')
lcolors2 = [0 0.65 0.95;
    0 0.25 0.55;
    1 0.6 0
    1 0.2 0];
figure('rend','painters','pos',[10 10 550 500]);hold on;
X = meanchoice;
idX = [1 3]; % gain;
bar(idX,nanmean(X(:,idX)),0.4,'FaceColor',lcolors2(1,:),'EdgeColor','none')
idX = [2 4]; % gain;
bar(idX,nanmean(X(:,idX)),0.4,'FaceColor',lcolors2(2,:),'EdgeColor','none')
idX = [5 7]; % gain;
bar(idX,nanmean(X(:,idX)),0.4,'FaceColor',lcolors2(3,:),'EdgeColor','none')
idX = [6 8]; % gain;
bar(idX,nanmean(X(:,idX)),0.4,'FaceColor',lcolors2(4,:),'EdgeColor','none')
set(gca,'XTick',1.5:2:7.5,'XTickLabel',{'High P(win)','Low P(win)','High P(win)','Low P(win)'},'LineWidth',3)
mcolor = colormap(jet(8));
mcolor([2:7],:)=[];
c=0;
Y = meanprob{1};
idY = [1:8];
plot([1:8]-0.35,nanmean(Y(:,idY)),'o','MarkerFaceColor',[0.5 0 0.5],'MarkerEdgeColor',[0.5 0 0.5]) %[0.5 0 0.5]
Y = meanprob{2};
plot([1:8]-0.15,nanmean(Y(:,idY)),'o','MarkerFaceColor',[0.4 0.4 0.4],'MarkerEdgeColor',[0.4 0.4 0.4])
Y = meanprob{3};
plot([1:8]+0.15,nanmean(Y(:,idY)),'o','MarkerFaceColor',[0.6 0.6 0.6],'MarkerEdgeColor',[0.6 0.6 0.6])
Y = meanprob{4};
plot([1:8]+0.35,nanmean(Y(:,idY)),'o','MarkerFaceColor',[0.8 0.8 0.8],'MarkerEdgeColor',[0.8 0.8 0.8])
l = legend({'Stable Win Stay','Stable Lose Stay','Volatile Win Stay','Volatile Lose Stay',...
    'Additive','Multiplicative','Probability only','Reward magnitude only'},...
    'location','North','NumColumns',2,'AutoUpdate','off')
title(l,'Data                         Model prediction')
legend boxoff
xlim([0.2 8.5])
errorbar(nanmean(X), nanstd(X)./sqrt(size(X,1)),'k.')
idY = [1:8];
Y = meanprob{1};
errorbar(idY-0.35,nanmean(Y(:,idY)), nanstd(Y(:,idY))./sqrt(size(Y(:,idY),1)),'.','Color',[0.5 0 0.5])
Y = meanprob{2};
errorbar(idY-0.15,nanmean(Y(:,idY)), nanstd(Y(:,idY))./sqrt(size(Y(:,idY),1)),'.','Color',[0.4 0.4 0.4])
Y = meanprob{3};
errorbar(idY+0.15,nanmean(Y(:,idY)), nanstd(Y(:,idY))./sqrt(size(Y(:,idY),1)),'.','Color',[0.6 0.6 0.6])
Y = meanprob{4};
errorbar(idY+0.35,nanmean(Y(:,idY)), nanstd(Y(:,idY))./sqrt(size(Y(:,idY),1)),'.','Color',[0.8 0.8 0.8])
ylim([0 1.2])
line(0.2:0.1:8.5,zeros(1,84),'color',[0 0 0],'linewidth',2)
set(gca,'FontSize',14,'YTick',[0:0.2:1])
ylabel('Choice proportion','FontSize',18)
xlabel('Stable                   Volatile','FontSize',18)
axis square
%% 2C Learning rate
figure('rend','painters','pos',[10 10 550 500]);hold on;
for iparam = 1;
    for imodel=1 
        % get the parameter
        X = [allChoiceData{1}.param_est{1,imodel}(:,iparam),allChoiceData{1}.param_est{2,imodel}(:,iparam)];
        Y = [allChoiceData{2}.param_est{1,imodel}(:,iparam),allChoiceData{2}.param_est{2,imodel}(:,iparam)];
        % plot individual data
        for i=1:size(X,1)
        p = plot(1:2,(X(i,1:2)),'k-','MarkerFaceColor',[0.3 0.3 0.3],'MarkerEdgeColor',[0.3 0.3 0.3],'LineWidth',1);
        p.Color(4) = 0.4;
        end
        for i=1:size(Y,1)
        p = plot(3:4,(Y(i,1:2)),'k-','MarkerFaceColor',[0.3 0.3 0.3],'MarkerEdgeColor',[0.3 0.3 0.3],'LineWidth',1);
        p.Color(4) = 0.4;
        end
                % plot the parameter SE      
        idx = [1];
        errorbar(1,mean(X(:,idx)),std(X(:,idx))./sqrt(length(X(:,idx))),'color',[0 0.45 0.75],'linewidth',4)
        idx = [2];
        errorbar(2,mean(X(:,idx)),std(X(:,idx))./sqrt(length(X(:,idx))),'color',[1 0.4 0],'linewidth',4)
        idx = [2];
        errorbar(4,mean(Y(:,idx)),std(Y(:,idx))./sqrt(length(Y(:,idx))),'color',[0 0.45 0.75],'linewidth',4)
        idx = [1];
        errorbar(3,mean(Y(:,idx)),std(Y(:,idx))./sqrt(length(Y(:,idx))),'color',[1 0.4 0],'linewidth',4)
        % plot the parameter average
        line(0.9:0.1:1.1,ones(1,3).*mean(X(:,1)),'color',[0 0.45 0.75],'linewidth',3)
        line([0.9:0.1:1.1]+3,ones(1,3).*mean(Y(:,2)),'color',[0 0.45 0.75],'linewidth',3)
        line([0.9:0.1:1.1]+1,ones(1,3).*mean(X(:,2)),'color',[1 0.4 0],'linewidth',3)
        line([0.9:0.1:1.1]+2,ones(1,3).*mean(Y(:,1)),'color',[1 0.4 0],'linewidth',3)
        % set labels & limits
        ylabel('Learning rate (\alpha)')
        ylim([0 0.95])
        xlim([0.5 4.5])
        set(gca,'YTick',[0:0.2:0.8],'Xtick',[1:4],'XTickLabel',{'Stable 1','Volatile 1','Volatile 2','Stable 2'},'Fontsize',18,'LineWidth',3)
    end
end
axis square
header = {'StableGroup1Alpha','VolatileGroup1Alpha','VolatileGroup2Alpha','StableGroup2Alpha',};
dataMat = [[X;nan(1,2)] Y];
xlswrite('eLife-fig2-dataSource1',[header;num2cell(dataMat)],'LearningRate')

