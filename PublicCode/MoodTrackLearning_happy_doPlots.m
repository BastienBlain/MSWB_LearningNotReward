function  [] = MoodTrackLearning_happy_doPlots(happyData)
% Bastien Blain, bastien.blain@gmail.com
% February, 2019
%% Figure 3. Happiness is explained by probability and probability prediction error
% Happiness after winning/not winning
lcontext ={'Stable','Volatile'};
lcolors = [0 0.45 0.75
    1 0.55 0];
happy_gainall{1} = [happyData{1}.happy_gain(1,:),happyData{2}.happy_gain(2,:)];
happy_gainall{2} = [happyData{1}.happy_gain(2,:),happyData{2}.happy_gain(1,:)];
happy_lossall{1} = [happyData{1}.happy_loss(1,:),happyData{2}.happy_loss(2,:)];
happy_lossall{2} = [happyData{1}.happy_loss(2,:),happyData{2}.happy_loss(1,:)];
for icontext=1:2
    %     subplot(1,2,icontext);
    figure('rend','painters','pos',[100 100 525 500]);
    hold on;%subplot(1,2,1)
    scatter(happy_lossall{icontext}', happy_gainall{icontext}',90,'filled','MarkerFaceColor',lcolors(icontext,:),'MarkerEdgeColor',[0 0 0])
    ylabel('Happiness after winning')
    xlabel('Happiness after losing')
    title(lcontext{icontext})
    xlim([0 100])
    ylim([0 100])
    
    plot(0:10:100,0:10:100,'k--','LineWidth',3)
    set(gca,'XTick',[0 50 100],'FontSize',18,'LineWidth',3)
    axis square
end
header = {'StableHappinnessGain','StableHappinnessLoss','StableHappinnessGain','StableHappinnessLoss'};
dataMat = [happy_gainall{1}' happy_lossall{1}' happy_gainall{2}' happy_lossall{2}'];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure3aHappinessGainVsLoss')
% Happiness time series
model_names = {'PPEhat','RPEhat','PPE','RPE','Phat+PPEhat','Phat+RPEhat','EVhatmult+PPEhat','EVhatmult+RPEhat',...
    'EVhatadd+RPEhat','EVhatadd+PPEhat','CNreward','rewardFreeRef','P+PPE','PPEhatmult',...
    'Phatmult+PPEhatmult','SignedOutcome','PPEhat+Counterfactual','Interaction1','Interaction2','WinLoss'}

matrawhappyall{1} = [happyData{1}.mathappyalltrials{1};happyData{2}.mathappyalltrials{2}];
matrawhappyall{2} = [happyData{1}.mathappyalltrials{2};happyData{2}.mathappyalltrials{1}];
matZhappyall{1} = [happyData{1}.matZhappyalltrials{1};happyData{2}.matZhappyalltrials{2}];
matZhappyall{2} = [happyData{1}.matZhappyalltrials{2};happyData{2}.matZhappyalltrials{1}];
lModel=[5];%[5 1 2 6 19]
for iModel=1:length(lModel)
matpredhappyall{1,iModel} = [happyData{1}.matpredhappyalltrials{1,lModel(iModel)};happyData{2}.matpredhappyalltrials{2,lModel(iModel)}];
matpredhappyall{2,iModel} = [happyData{1}.matpredhappyalltrials{2,lModel(iModel)};happyData{2}.matpredhappyalltrials{1,lModel(iModel)}];
end
lColorPred = [0 0 0;
             .3 .3 .3;
             .6 .6 .6];
% RAW
for icontext=1:2
    figure('rend','painters','pos',[100 100 525 500]);
    hold on;
    data  = matrawhappyall{icontext}.*100;%mat_backfilled_happyall{icontext}.*100;%matrawhappyalltrialsall{icontext}.*100;
    pred  = matpredhappyall{icontext}.*100;
    plot(backfilling(nanmean(data)),'Color', max(lcolors(icontext,:),0),'LineWIdth',3)
    for iModel=1:length(lModel)
    plot(backfilling(nanmean(matpredhappyall{icontext,iModel}.*100)),':','Color',lColorPred(iModel,:),'LineWIdth',3)
    end
    if icontext==1
        legend({'Happiness (stable)','$\hat{P}$ + $\hat{PPE}$','$\hat{RPE}$','Win+Loss'},...
            'Interpreter','latex','location','North','AutoUpdate','off')
    else
        legend({'Happiness (volatile)','$\hat{P}$ + $\hat{PPE}$','$\hat{RPE}$','Win+Loss'},...
            'Interpreter','latex','location','North','AutoUpdate','off')
    end
    %plot((nanmean(data)),'Color', max(lcolors(icontext,:),0),'LineWIdth',3)
    legend boxoff
    jbfill(1:80,backfilling(nanmean(data)+nanstd((data))./sqrt(sum(~isnan(data)))),backfilling(nanmean(data)-nanstd((data))./sqrt(size(data,1))),...
        lcolors(icontext,:),lcolors(icontext,:),0,0.3)
    if icontext==2
        line([20.5 20.5],[0 100],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
        line([40.5 40.5],[0 100],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
        line([60.5 60.5],[0 100],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
    end
    ylim([20 80])
    xlim([1 80])
    ylabel('Happiness')
    xlabel('Trial number')
    set(gca,'Ytick',[20 50 80],'FontSize',18,'LineWidth',3)
    axis square
end

% z-score
for icontext=1:2
    figure('rend','painters','pos',[100 100 525 500]);
    hold on;
    data  = matZhappyall{icontext};%mat_backfilled_happyall{icontext}.*100;%matrawhappyalltrialsall{icontext}.*100;
    pred  = matpredhappyall{icontext};
    plot(backfilling(nanmean(data)),'Color', max(lcolors(icontext,:),0),'LineWIdth',3)
    for iModel=1:length(lModel)
    plot(backfilling(nanmean(matpredhappyall{icontext,iModel})),':','Color',lColorPred(iModel,:),'LineWIdth',3)
    end
    if icontext==1
        legend({'Happiness (stable)','$\hat{P}$ + $\hat{PPE}$','$\hat{RPE}$','Win+Loss'},...
            'Interpreter','latex','location','North','AutoUpdate','off')
    else
        legend({'Happiness (volatile)','$\hat{P}$ + $\hat{PPE}$','$\hat{RPE}$','Win+Loss'},...
            'Interpreter','latex','location','North','AutoUpdate','off')
    end
    %plot((nanmean(data)),'Color', max(lcolors(icontext,:),0),'LineWIdth',3)
    legend boxoff
    jbfill(1:80,backfilling(nanmean(data)+nanstd((data))./sqrt(sum(~isnan(data)))),backfilling(nanmean(data)-nanstd((data))./sqrt(size(data,1))),...
        lcolors(icontext,:),lcolors(icontext,:),0,0.3)
    if icontext==2
        line([20.5 20.5],[-1 1],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
        line([40.5 40.5],[-1 1],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
        line([60.5 60.5],[-1 1],'LineWidth',2,'LineStyle','--','Color',[.5 .5 .5])
    end
    ylim([-1 1])
    xlim([1 80])
    ylabel('Happiness')
    xlabel('Trial number')
    set(gca,'Ytick',[-1 0 1],'FontSize',18,'LineWidth',3)
    axis square
end
xlswrite('eLife-fig3-dataSource2',[matZhappyall{1}],'Figure3BHappinessDataStable')
xlswrite('eLife-fig3-dataSource2',[matZhappyall{2}],'Figure3BHappinessDataVolatile')
xlswrite('eLife-fig3-dataSource2',[matpredhappyall{1}],'Figure3BHappinessDataStable')
xlswrite('eLife-fig3-dataSource2',[matpredhappyall{2}],'Figure3BHappinessDataVolatile')

% P+PPE model parameters
for imodel=1:11
    paramall{1,imodel}(:,:) = [happyData{1}.param_est{1,imodel};happyData{2}.param_est{2,imodel}];
    paramall{2,imodel}(:,:) = [happyData{1}.param_est{2,imodel};happyData{2}.param_est{1,imodel}];
end
header = {'StableP+PPEWp','StableP+PPEWppe','VolatileP+PPEWp','VolatileP+PPEWppe'};
dataMat = [paramall{1,5}(:,[1:2]) paramall{2,5}(:,[1:2])];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure3cWPPEvsWP')
% imodel = 11;
% nparam = size(paramall{1,imodel},2);
% for icontext=1:2;
%     figure('rend','painters','pos',[100 100 550 500])
%     hold on;
%     X = paramall{icontext,imodel}(:,1);
%     bar(1,mean(X),0.75,'FaceColor',lcolors(icontext,:));
%     errorbar(1,mean(X),std(X)./sqrt(size(X,1)),'k.')
%     Y = paramall{icontext,imodel}(:,2);
%     bar(2,mean(Y),0.75,'FaceColor',lcolors(icontext,:));
%     errorbar(2,mean(Y),std(Y)./sqrt(size(Y,1)),'k.')
%     if icontext==1
%         ylabel('Estimate in stable')
%     else
%         ylabel('Estimate in volatile')
%     end
%     % plot individual data
%     Z = [X Y];
%     for i=1:size(Z,1)
%         p = plot(1:2,(Z(i,1:2)),'k-','MarkerFaceColor',[0.4 0.4 0.4],'MarkerEdgeColor',[0.4 0.4 0.4]);
%         p.Color(4) = 0.2;
%     end
%     set(gca,'XTick',1:2,'XTickLabel',{'W_{win}','W_{loss}'},'FontSize',22,'LineWidth',3)
%     ylim([-0.2 1])
%     axis square
%     [h p] = ttest(X,Y);
% end

imodel = 11;
nparam = size(paramall{1,imodel},2);
for icontext=1:2;
    figure('rend','painters','pos',[100 100 550 500])
    hold on;
    X = paramall{icontext,imodel}(:,1);
    Y = paramall{icontext,imodel}(:,2);
    scatter(X,Y,90,'filled','MarkerFaceColor',lcolors(icontext,:),'MarkerEdgeColor',[0 0 0])
    xlabel('W_{win} estimate')
    ylabel('W_{loss} estimate')
    xlim([-.5 2.5])
    ylim([-.5 2.5])    
    title([lcontext{icontext}])
    plot(-0.5:.1:2.5,-0.5:.1:2.5,'k--','LineWidth',3)    
    set(gca,'FontSize',18,'LineWidth',3)
    axis square    
end

imodel = 5;
nparam = size(paramall{1,imodel},2);
for icontext=1:2;
    figure('rend','painters','pos',[100 100 550 500])
    hold on;
    X = paramall{icontext,imodel}(:,1);
    Y = paramall{icontext,imodel}(:,2);
    scatter(X, Y,90,'filled','MarkerFaceColor',lcolors(icontext,:),'MarkerEdgeColor',[0 0 0])
    xlabel('$\hat{P}$ weight estimate','Interpreter','latex')
    ylabel('$\hat{PPE}$ weight estimate','Interpreter','latex')
    %title(lcontext{icontext})
    xlim([-.5 2.5])
    ylim([-0.5 2.5])
    plot(-0.5:.1:2.5,-0.5:.1:2.5,'k--','LineWidth',3)
    axis square
    [h p] = ttest(X,Y);
    set(gca,'XTick',[0:1:2],'YTick',[0:1:2],'FontSize',18,'LineWidth',3)
end


%% Figure 4. Mood dynamics are more sensitive to learning- than choice-relevant variables
r2all{1}  = [happyData{1}.matr2{1};happyData{2}.matr2{2}];
r2all{2}  = [happyData{1}.matr2{2};happyData{2}.matr2{1}];
% A PPE versus RPE
for icontext=1:2
    figure('rend','painters','pos',[100 100 525 500])
    hold on;
    X = r2all{icontext}(:,2);%rpe
    Y = r2all{icontext}(:,1);%ppe
    scatter(X,Y,100,'MarkerFaceColor',lcolors(icontext,:),'MarkerEdgeColor',[0 0 0])
    title([lcontext{icontext}])
    plot(0:0.1:1,0:0.1:1,'--k','LineWidth',3)
    xlabel('$r^2 for  \hat{RPE}$','Interpreter','latex')
    ylabel('$r^2 for \hat{PPE}$','Interpreter','latex')
    set(gca,'FontSize',18)
    axis square
end
header = {'StableR2PPEhat','StableR2RPEhat','VolatileR2PPEhat','VolatileR2RPEhat'};
dataMat = [r2all{1}(:,1) r2all{1}(:,2) r2all{2}(:,1) r2all{2}(:,2)];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure4aR2PPEvsR2RPE')
% B P + PPE versus EV + PPE
for icontext=1:2
    figure('rend','painters','pos',[100 100 525 500])
    hold on;
    X = r2all{icontext}(:,7);%rv + ppe
    Y = r2all{icontext}(:,5);% p + ppe
    scatter(X,Y,100,'MarkerFaceColor',lcolors(icontext,:),'MarkerEdgeColor',[0 0 0])
    title([lcontext{icontext}])
    plot(0:0.1:1,0:0.1:1,'--k','LineWidth',3)
    ylabel('$r^2 for \hat{P} + \hat{PPE}$','Interpreter','latex')
    xlabel('$r^2 for \hat{EV} + \hat{PPE} $','Interpreter','latex')
    set(gca,'FontSize',18)
    axis square
end
header = {'StableR2P+PPEhat','StableR2P+RPEhat','VolatileR2P+PPEhat','VolatileR2P+RPEhat'};
dataMat = [r2all{1}(:,5) r2all{1}(:,7) r2all{2}(:,5) r2all{2}(:,7)];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure4bR2PPEvsR2RPE')
%% Figure 5. Forgetting factors are consistent across stable and volatile environments
lcolors(3,:) = [0.5 0.5 0.5];
% scatter plot
for iparam=1:2
    figure('rend','painters','pos',[300 100 525 500]);
    hold on;
    Y = paramall{2,5}(:,iparam+1);
    X = paramall{1,5}(:,iparam+1);
    if iparam==1
        ylabel('Volatile happiness $$\hat{PPE}$$','Interpreter','Latex')
        xlabel('Volatile happiness $$\hat{PPE}$$','Interpreter','Latex')
    else
        ylabel('Volatile happiness forgetting factor','Interpreter','Latex')
        xlabel('Stable happiness forgetting factor','Interpreter','Latex')
    end
    if iparam==1
        xlim([-0.05 2])
        ylim([-0.05 2])
    else
        xlim([0 1])
        ylim([0 1])
    end
    scatter(X,Y,100,'MarkerFaceColor',lcolors(3,:),'MarkerEdgeColor',[0 0 0])
    % scatter(X,Y,50,'MarkerFaceColor',[.5 .5 .5],'MarkerEdgeColor',[0 0 0])
    [r p] = corr(X,Y,'type','Spearman')
    allstats.happyParamcorr(iparam,1) = r;
    allstats.happyParamcorr(iparam,2) = p;
    if iparam==1
        plot(-0.05:0.1:2,-0.05:0.1:2,'k--')
    else
        plot(0:0.1:1,0:0.1:1,'k--')
    end
    %title([names_context{icontext},', \rho = ',num2str(round(r,3)),', p = ',num2str(round(p,3))])
    bb = glmfit(X,Y)
    plot(X,bb(1) + bb(2).*[X],'k','LineWidth',3)
    set(gca,'FontSize',18,'LineWidth',3)
    axis square
end
colNames = {'SpearmanRho','Pvalue'};
rowNames = {'PPEweight','ForgettingFactor'};
allstats.happyParamcorr = array2table(allstats.happyParamcorr,'RowNames',rowNames,'VariableNames',colNames);

header = {'StableP+PPE_wPPE','VolatileP+PPE_wPPE'};
dataMat = [paramall{1,5}(:,2) paramall{2,5}(:,2)];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure5a_wPPE')

%  gamma error bars
figure('rend','painters','pos',[300 100 525 500])
ii=0;
for iparam = 1;%[0 2 4 6]
    for imodel=[5]%,3]
        ii=ii+1;
        subplot(1,1,ii);hold on;
        X = [happyData{1}.param_est{1,5}(:,3),happyData{1}.param_est{2,5}(:,3)];
        Y = [happyData{2}.param_est{1,5}(:,3),happyData{2}.param_est{2,5}(:,3)];
        ylabel('Happiness forgetting factor (\gamma)','Interpreter','Latex')
        %ylim([0 0.8])
        xlim([0.5 4.5]);
        % plot individual data
        for i=1:size(X,1)
        p = plot(1:2,(X(i,1:2)),'k-','MarkerFaceColor',[0.3 0.3 0.3],'MarkerEdgeColor',[0.3 0.3 0.3]);
        p.Color(4) = 0.4;
        end
        for i=1:size(Y,1)
        p = plot(3:4,(Y(i,1:2)),'k-','MarkerFaceColor',[0.3 0.3 0.3],'MarkerEdgeColor',[0.3 0.3 0.3]);
        p.Color(4) = 0.4;
        end
        idx = [1];
        errorbar(idx,mean(X(:,idx)),std(X(:,idx))./sqrt(length(X(:,idx))),'color',[0 0.45 0.75],'linewidth',4)
        idx = [2];
        errorbar(2,mean(X(:,idx)),std(X(:,idx))./sqrt(length(X(:,idx))),'color',[1 0.4 0],'linewidth',4)
        idx = [2];
        errorbar(4,mean(Y(:,idx)),std(Y(:,idx))./sqrt(length(Y(:,idx))),'color',[0 0.45 0.75],'linewidth',4)
        idx = [1];
        errorbar(3,mean(Y(:,idx)),std(Y(:,idx))./sqrt(length(Y(:,idx))),'color',[1 0.4 0],'linewidth',4)
        line(0.9:0.1:1.1,ones(1,3).*mean(X(:,1)),'color',[0 0.45 0.75],'linewidth',3)
        line([0.9:0.1:1.1]+3,ones(1,3).*mean(Y(:,2)),'color',[0 0.45 0.75],'linewidth',3)
        line([0.9:0.1:1.1]+1,ones(1,3).*mean(X(:,2)),'color',[1 0.4 0],'linewidth',3)
        line([0.9:0.1:1.1]+2,ones(1,3).*mean(Y(:,1)),'color',[1 0.4 0],'linewidth',3)
        set(gca,'Xtick',[1:4],'XTickLabel',{'Stable 1','Volatile 1','Volatile 2','Stable 2'},'Fontsize',18,'LineWidth',3)
        axis square
    end
end
header = {'StableGroup1Gamma','VolatileGroup1Gamma','VolatileGroup2Gamma','StableGroup2Gamma'};
dataMat = [[X;nan(1,2)] Y];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure5bGamma')
header = {'StableP+PPE_gamma','VolatileP+PPE_gamma'};
dataMat = [paramall{1,5}(:,3) paramall{2,5}(:,3)];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure5c_gamma')
%% Figure 5. Adding scatter plots instead of errrobars for gammas
for iparam=2
    figure('rend','painters','pos',[300 100 525 500]);
    hold on;
    Y = paramall{2,5}(:,iparam+1);
    X = paramall{1,5}(:,iparam+1);
    if iparam==1
        ylabel('Volatile happiness PPE weight')
        xlabel('Stable happiness PPE weight')
    else
        ylabel('Volatile happiness forgetting factor')
        xlabel('Stable happiness forgetting factor')
    end
    xlim([0 1])
    ylim([0 1])
    plot(0:0.1:1,0:0.1:1,'k--')
    scatter(X,Y,100,'MarkerFaceColor',lcolors(3,:),'MarkerEdgeColor',[0 0 0])
    % scatter(X,Y,50,'MarkerFaceColor',[.5 .5 .5],'MarkerEdgeColor',[0 0 0])
    [r p] = corr(X,Y,'type','Spearman')
    %title([names_context{icontext},', \rho = ',num2str(round(r,3)),', p = ',num2str(round(p,3))])
    bb = glmfit(X,Y)
    plot(X,bb(1) + bb(2).*[X],'k','LineWidth',3)
    set(gca,'FontSize',18,'LineWidth',3)
    axis square
end
[h p c stats]= ttest(X,Y)
mean(X-Y)
std(X-Y)./sqrt(75)
colNames = {'SpearmanRho','Pvalue'};
rowNames = {'PPEweight','ForgettingFactor'};

%% Figure 6. Baseline mood decreases with depressive symptoms in volatile environments
scorePHQvall  = [happyData{1}.scorePHQ';happyData{2}.scorePHQ'];
scoreSTAIvall = [happyData{1}.scoreSTAI';happyData{2}.scoreSTAI'];
matmeanrawhappyall{1} = [happyData{1}.matmean{1}';happyData{2}.matmean{2}'];
matmeanrawhappyall{2} = [happyData{1}.matmean{2}';happyData{2}.matmean{1}'];

cd Figures
f = 0;
% A Mood average
for i=1:3
    figure('rend','painters','pos',[100 100 525 500]);hold on;
    X = scorePHQvall;
    if i==1
        Y = matmeanrawhappyall{1};
        ylabel('Average happiness (stable)')
        ylim([0 100])
    elseif i==2
        Y = matmeanrawhappyall{2};
        ylabel('Average happiness (volatile)')
        ylim([0 100])
    elseif i==3
        Y = matmeanrawhappyall{2}-matmeanrawhappyall{1};
        ylabel('Average happiness (volatile - stable)')
        ylim([-60 60])
    end
    % get the stats
    
    scatter(X,Y,100,'MarkerFaceColor',lcolors(i,:),'MarkerEdgeColor',[0 0 0])
    [r p] = corr(X,Y,'type','Spearman')
    allstats.happyMeanClinicalRho(1,i+3) = r;
    allstats.happyMeanClinicalP(1,i+3) = p;
    % fit & plot the slope
    bb = glmfit(X,Y)
    plot(X,bb(1) + bb(2).*X,'k','LineWidth',3)
    xlabel('Depressive symptoms (PHQ)')
    xlim([0 30])
    set(gca,'FontSize',18)
    axis square
    f=f+1;
    saveas(gcf,['Mood',num2str(f),'.tif'])
end
header = {'StableHappyMean','VolatileHappyMean','Stable-VolatileHappyMean'};
dataMat = [matmeanrawhappyall{1} matmeanrawhappyall{2} matmeanrawhappyall{1}-matmeanrawhappyall{2}];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure6a_HappyMean')

% B Happiness constant
for i=1:3
    figure('rend','painters','pos',[100 100 525 500]);hold on;
    X = scorePHQvall;
    if i==1
        Y = paramall{1,imodel}(:,4).*100;
        ylabel('Happiness constant (stable)')
        ylim([0 100])
    elseif i==2
        Y = paramall{2,imodel}(:,4).*100;
        ylabel('Happiness constant (volatile)')
        ylim([0 100])
    elseif i==3
        Y = paramall{2,imodel}(:,4).*100-paramall{1,imodel}(:,4).*100;
        ylabel('Happiness constant (volatile - stable)')
        ylim([-80 80])
    end
    scatter(X,Y,100,'MarkerFaceColor',lcolors(i,:),'MarkerEdgeColor',[0 0 0])
    [r p] = corr(X,Y,'type','Spearman')
    allstats.happyCstClinicalRho(1,i+3) = r;
    allstats.happyCstClinicalP(1,i+3) = p;
    % fit & plot the slope
    bb = glmfit(X,Y)
    plot(X,bb(1) + bb(2).*X,'k','LineWidth',3)
    xlabel('Depressive symptoms (PHQ)')
    xlim([0 30])
    set(gca,'FontSize',18)
    axis square
    f=f+1;
    saveas(gcf,['Mood',num2str(f),'.tif'])
end

header = {'StableP+PPE_cst','VolatileP+PPE_cst','Stable-VolatileP+PPE_cst'};
dataMat = [paramall{1,5}(:,4) paramall{2,5}(:,4) paramall{1,5}(:,4)-paramall{2,5}(:,4)];
xlswrite('eLife-fig3-dataSource2',[header;num2cell(dataMat)],'Figure6b_HappyCst')

%% supplementary figure 1 Win-Loss parameter
imodel=11;    
figure('rend','painters','pos',[100 100 950 700])
for icontext=1:2
    X = [paramall{icontext,imodel}(:,1) paramall{icontext,imodel}(:,2)];
    subplot(1,2,icontext);hold on;
    bar(mean(X),'FaceColor',lcolors(icontext,:))
    errorbar(mean(X),std(X)./sqrt(size(X,1)),'k.')
    for i=1:size(X,1)
        p = plot(1:2,(X(i,1:2)),'k-','MarkerFaceColor',[0.3 0.3 0.3],'MarkerEdgeColor',[0.3 0.3 0.3]);
        p.Color(4) = 0.2;
    end
    set(gca,'XTick',1:2,'XTickLabel',{'Win','Loss'},'FontSize',18)
    title(lcontext{icontext})
    ylim([-0.2 0.8])
    axis square
end
%% supplementary figure 2: model agnostic description of forgetting factor
kernel_PPEhat_all{1} = [happyData{1}.kernel_est{1,1};happyData{2}.kernel_est{2,1}];
kernel_PPEhat_all{2} = [happyData{1}.kernel_est{2,1};happyData{2}.kernel_est{1,1}];
kernel_PPE_all{1} = [happyData{1}.kernel_est{1,2};happyData{2}.kernel_est{2,2}];
kernel_PPE_all{2} = [happyData{1}.kernel_est{2,2};happyData{2}.kernel_est{1,2}];

figure('rend','painters','pos',[100 100 450 400]);hold on;
for icontext=1:2
    errorbar(mean(kernel_PPEhat_all{icontext}),std(kernel_PPEhat_all{icontext})./sqrt(size(kernel_PPEhat_all{icontext},1)),'Color',lcolors(icontext,:),'LineWidth',3)
    [h p] = ttest(kernel_PPEhat_all{icontext});
end
xlabel(['Previous ','$$\hat{PPE}$$'],'Interpreter','Latex')
ylabel('Weight on current happiness','Interpreter','Latex')
set(gca,'XTick',[1:10],'XTickLabel',-[1:10],'FontSize',18,'LineWidth',3)
% xlabel('Lag')
ylim([0 0.2])
xlim([0.8 10.2])

figure('rend','painters','pos',[100 100 450 400]);hold on;
for icontext=1:2
errorbar(mean(kernel_PPE_all{icontext}),std(kernel_PPE_all{icontext})./sqrt(size(kernel_PPE_all{icontext},1)),'Color',lcolors(icontext,:),'LineWidth',3)
end
xlabel(['Previous ','PPE'],'Interpreter','Latex')
ylabel('Weight on current happiness','Interpreter','Latex')
set(gca,'XTick',[1:10],'XTickLabel',-[1:10],'FontSize',18,'LineWidth',3)
% xlabel('Lag')
ylim([0 0.2])
xlim([0.8 10.2])

%% supplementary figure 3
% histogram model frequency
% plot histograms
BIC{1} = [happyData{1}.matBIC{1};happyData{2}.matBIC{2}];
BIC{2} = [happyData{1}.matBIC{2};happyData{2}.matBIC{1}];
AIC{1} = [happyData{1}.matAIC{1};happyData{2}.matAIC{2}];
AIC{2} = [happyData{1}.matAIC{2};happyData{2}.matAIC{1}];
matr2{1} = [happyData{1}.matr2{1};happyData{2}.matr2{2}];
matr2{2} = [happyData{1}.matr2{2};happyData{2}.matr2{1}];

model_names = {'$\hat{PPE}$','$\hat{RPE}$','PPE','RPE','$\hat{P}$ + $\hat{PPE}$','$\hat{P}$ + $\hat{RPE}$',...
    '$\hat{EV}$ + $\hat{RPE}$','$\hat{EV}$ + $\hat{PPE}$','R - $\bar{R}$','R - RP','Win - Loss'}

ioi = 1:4;
options.families = {[1,3], [2,4]} ;
options.DisplayWin = 0;
[posterior,outS] = VBA_groupBMC(-BIC{1}(:,ioi)',options) 
[posterior,outV] = VBA_groupBMC(-BIC{2}(:,ioi)',options) 

figure;
subplot(1,2,1);hold on;
bar(outS.Ef,'FaceColor',lcolors(1,:))
errorbar(outS.Ef,sqrt(diag(outS.Vf)),'.k')
title('Stable')
ylabel('Estimated frequency')
xlabel('Model')
set(gca,'Xtick',1:length(ioi),'XTickLabel',{model_names{ioi}},'XTickLabelRotation',90,'TickLabelInterpreter','latex','FontSize',15)
xlim([0.5 length(ioi)+0.5])
ylim([0 0.8])
axis square

subplot(1,2,2);hold on;
bar(outV.Ef,'FaceColor',lcolors(2,:))
errorbar(outV.Ef,sqrt(diag(outV.Vf)),'.k')
title('Volatile')
ylabel('Estimated frequency')
xlabel('Model')
set(gca,'Xtick',1:length(ioi),'XTickLabel',{model_names{ioi}},'XTickLabelRotation',90,'TickLabelInterpreter','latex','FontSize',15)
xlim([0.5 length(ioi)+0.5])
ylim([0 0.8])
axis square

options =  struct;
options.DisplayWin = 0;
ioi = 1:10;
[posterior,outS] = VBA_groupBMC(-BIC{1}(:,ioi)',options) 
[posterior,outV] = VBA_groupBMC(-BIC{2}(:,ioi)',options) 

figure;
subplot(1,2,1);hold on;
bar(outS.Ef,'FaceColor',lcolors(1,:))
errorbar(outS.Ef,sqrt(diag(outS.Vf)),'.k')
title('Stable')
ylabel('Estimated frequency')
xlabel('Model')
set(gca,'Xtick',1:length(ioi),'XTickLabel',{model_names{ioi}},'XTickLabelRotation',90,'TickLabelInterpreter','latex','FontSize',15)
xlim([0.5 length(ioi)+0.5])
ylim([0 0.8])
axis square

subplot(1,2,2);hold on;
bar(outV.Ef,'FaceColor',lcolors(2,:))
errorbar(outV.Ef,sqrt(diag(outV.Vf)),'.k')
title('Volatile')
ylabel('Estimated frequency')
xlabel('Model')
set(gca,'Xtick',1:length(ioi),'XTickLabel',{model_names{ioi}},'XTickLabelRotation',90,'TickLabelInterpreter','latex','FontSize',15)
xlim([0.5 length(ioi)+0.5])
ylim([0 0.8])
axis square

options.DisplayWin = 0;
ioi = [5 11];
[posterior,outS] = VBA_groupBMC(-BIC{1}(:,ioi)',options) 
[posterior,outV] = VBA_groupBMC(-BIC{2}(:,ioi)',options) 

figure;
subplot(1,2,1);hold on;
bar(outS.Ef,'FaceColor',lcolors(1,:))
errorbar(outS.Ef,sqrt(diag(outS.Vf)),'.k')
title('Stable')
ylabel('Estimated frequency')
xlabel('Model')
set(gca,'Xtick',1:length(ioi),'XTickLabel',{model_names{ioi}},'XTickLabelRotation',90,'TickLabelInterpreter','latex','FontSize',15)
xlim([0.5 length(ioi)+0.5])
ylim([0 0.8])
axis square

subplot(1,2,2);hold on;
bar(outV.Ef,'FaceColor',lcolors(2,:))
errorbar(outV.Ef,sqrt(diag(outV.Vf)),'.k')
title('Volatile')
ylabel('Estimated frequency')
xlabel('Model')
set(gca,'Xtick',1:length(ioi),'XTickLabel',{model_names{ioi}},'XTickLabelRotation',90,'TickLabelInterpreter','latex','FontSize',15)
xlim([0.5 length(ioi)+0.5])
ylim([0 0.8])
axis square

cd ..