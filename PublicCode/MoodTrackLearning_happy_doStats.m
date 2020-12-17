 function  [happy] = MoodTrackLearning_happy_doStats(happyData,choiceData,nmodel)
% Bastien Blain, bastien.blain@gmail.com
% February, 2019

model_names = {'PPEhat','RPEhat','PPE','RPE','Phat+PPEhat','Phat+RPEhat','EVhat+PPEhat','EVhat+RPEhat',...
    'CNreward','rewardFreeRef','WinLoss','Interaction1','boostDecay'}%% SD
SDall{1} = [happyData{1}.matSD{1},happyData{2}.matSD{2}]';
SDall{2} = [happyData{1}.matSD{2},happyData{2}.matSD{1}]';
for icontext=1:2
   happy.SD(icontext,1) = mean(SDall{icontext});
   happy.SD(icontext,2) = std(SDall{icontext})./sqrt(size(SDall{icontext},1));
end
colNames = {'mean','SEM'};
rowNames = {'Stable','Volatile'};
happy.SD = array2table(happy.SD,'RowNames',rowNames,'VariableNames',colNames);

%% Winning versus losing
happy_gainall{1} = [happyData{1}.happy_gain(1,:),happyData{2}.happy_gain(2,:)];
happy_gainall{2} = [happyData{1}.happy_gain(2,:),happyData{2}.happy_gain(1,:)];
happy_lossall{1} = [happyData{1}.happy_loss(1,:),happyData{2}.happy_loss(2,:)];
happy_lossall{2} = [happyData{1}.happy_loss(2,:),happyData{2}.happy_loss(1,:)];

for icontext=1:2
    happy.winLoss(icontext,1) = mean(happy_gainall{icontext});
    happy.winLoss(icontext,2) = std(happy_gainall{icontext})./sqrt(size(happy_gainall{icontext},2));
    happy.winLoss(icontext,3) = mean(happy_lossall{icontext});
    happy.winLoss(icontext,4) = std(happy_lossall{icontext})./sqrt(size(happy_lossall{icontext},2));
    [p,h,stats] = signrank(happy_gainall{icontext},happy_lossall{icontext});
    happy.winLoss(icontext,5) = stats.signedrank;
    happy.winLoss(icontext,6) = stats.zval;
    happy.winLoss(icontext,7) = p;
end
colNames = {'WinMean','WinSE','LossMean','LossSE','Tvalue','Zvalue','Pvalue'};
rowNames = {'Stable','Volatile'};
happy.winLoss = array2table(happy.winLoss,'RowNames',rowNames,'VariableNames',colNames);

%% Mean difference between stable and volatile
meanall{1} = [happyData{1}.matmean{1},happyData{2}.matmean{2}]';
meanall{2} = [happyData{1}.matmean{2},happyData{2}.matmean{1}]';

happy.mean(1) = mean(meanall{1});
happy.mean(2) = std(meanall{1})./sqrt(size(meanall{1},1));
happy.mean(3) = mean(meanall{2});
happy.mean(4) = std(meanall{2})./sqrt(size(meanall{2},1));
[p,h,stats] = signrank(meanall{1},meanall{2})
happy.mean(5) = stats.signedrank;
happy.mean(6) = stats.zval;
happy.mean(7) = p;

colNames = {'StableMean','StableSE','VolatileMean','VolatileSE','Tvalue','Zvalue','Pvalue'};
happy.mean = array2table(happy.mean,'VariableNames',colNames);

matProbaAccuracy{1} = [choiceData{1}.matperfProba(:,1);choiceData{2}.matperfProba(:,2)];
matProbaAccuracy{2} = [choiceData{1}.matperfProba(:,2);choiceData{2}.matperfProba(:,1)];
[r, p]= corr(matProbaAccuracy{1}-matProbaAccuracy{2},meanall{1}-meanall{2});

matEVAccuracy{1} = [choiceData{1}.matperfEV(:,1);choiceData{2}.matperfEV(:,2)];
matEVAccuracy{2} = [choiceData{1}.matperfEV(:,2);choiceData{2}.matperfEV(:,1)];
[r2, p2]= corr(matEVAccuracy{1}-matEVAccuracy{2},meanall{1}-meanall{2});

happy.meanDiffVSAccuray(1,1) = r;
happy.meanDiffVSAccuray(1,2) = p;
happy.meanDiffVSAccuray(2,1) = r2;
happy.meanDiffVSAccuray(2,2) = p2;
colNames = {'rho','Pvalue'};
rowNames = {'ProbaMax','EVmax'};
happy.meanDiffVSAccuray = array2table(happy.meanDiffVSAccuray,'VariableNames',colNames);
%% Table 2.Happiness model comparison results
BICall{1} = [happyData{1}.matBIC{1};happyData{2}.matBIC{2}];
BICall{2} = [happyData{1}.matBIC{2};happyData{2}.matBIC{1}];
r2all{1}  = [happyData{1}.matr2{1};happyData{2}.matr2{2}]; 
r2all{2}  = [happyData{1}.matr2{2};happyData{2}.matr2{1}];
for imodel=1:nmodel
   happy.modelComparison(imodel,1) =  happyData{1}.dof{1}(1,imodel);
   happy.modelComparison(imodel,2) = mean(r2all{1}(:,imodel));
   happy.modelComparison(imodel,3) = std(r2all{1}(:,imodel))./sqrt(size(r2all{1}(:,imodel),1));
   happy.modelComparison(imodel,4) = mean(r2all{2}(:,imodel));
   happy.modelComparison(imodel,5) = std(r2all{2}(:,imodel))./sqrt(size(r2all{2}(:,imodel),1));
   happy.modelComparison(imodel,6) = sum(BICall{1}(:,imodel));
   happy.modelComparison(imodel,7) = sum(BICall{2}(:,imodel));
   happy.modelComparison(imodel,8) = sum(BICall{1}(:,imodel))-sum(BICall{1}(:,5));
   happy.modelComparison(imodel,9) = sum(BICall{2}(:,imodel))-sum(BICall{2}(:,5));
end
colNames = {'NumberOfParameters','MeanStablePseudor2','SEstablePseudor2','MeanVolatilePseudor2','SEvolatilePseudor2',...
    'StableBIC','VolatileBIC','StableDeltaBIC','VolatileDeltaBIC'};
rowNames = model_names;
happy.modelComparison = array2table(happy.modelComparison,'RowNames',rowNames,'VariableNames',colNames);

%% stats for figure S3: estimated frequency
BIC{1} = [happyData{1}.matBIC{1};happyData{2}.matBIC{2}];
BIC{2} = [happyData{1}.matBIC{2};happyData{2}.matBIC{1}];
model_names2 = {'$\hat{PPE}$','$\hat{RPE}$','PPE','RPE','$\hat{P}$ + $\hat{PPE}$','$\hat{P}$ + $\hat{RPE}$',...
    '$\hat{EV}$ + $\hat{RPE}$','$\hat{EV}$ + $\hat{PPE}$','R - $\bar{R}$','R - RP','Win - Loss'}

ioi = 1:4;
options.families = {[1,3], [2,4]} ;
options.DisplayWin = 0;
[posterior,outS] = VBA_groupBMC(-BIC{1}(:,ioi)',options) 
[posterior,outV] = VBA_groupBMC(-BIC{2}(:,ioi)',options) 

happy.modelFrequencyPE(1,ioi) = outS.Ef(ioi);
happy.modelFrequencyPE(2,ioi) = outS.ep(ioi);
happy.modelFrequencyPE(3,ioi) = outV.Ef(ioi);
happy.modelFrequencyPE(4,ioi) = outV.ep(ioi);
colNames = {model_names2{ioi}};
rowNames = {'StableEF','StableEP','VolatileEF','VolatileEP'};
happy.modelFrequencyPE = array2table(happy.modelFrequencyPE,'RowNames',rowNames,'VariableNames',colNames);

happy.modelFrequencyPEfamilies(1,1:2) = outS.families.Ef(1:2);
happy.modelFrequencyPEfamilies(2,1:2) = outS.families.ep(1:2);
happy.modelFrequencyPEfamilies(3,1:2) = outV.families.Ef(1:2);
happy.modelFrequencyPEfamilies(4,1:2) = outV.families.ep(1:2);
colNames = {'PPE family','RPE family'};
rowNames = {'StableEF','StableEP','VolatileEF','VolatileEP'};
happy.modelFrequencyPEfamilies = array2table(happy.modelFrequencyPEfamilies,'RowNames',rowNames,'VariableNames',colNames);

options =  struct;
options.DisplayWin = 0;
ioi = 1:10;
[posterior,outS] = VBA_groupBMC(-BIC{1}(:,ioi)',options) 
[posterior,outV] = VBA_groupBMC(-BIC{2}(:,ioi)',options) 

happy.modelFrequencyAll(1,:) = outS.Ef(ioi);
happy.modelFrequencyAll(2,:) = outS.ep(ioi);
happy.modelFrequencyAll(3,:) = outV.Ef(ioi);
happy.modelFrequencyAll(4,:) = outV.ep(ioi);
colNames = {model_names2{ioi}};
rowNames = {'StableEF','StableEP','VolatileEF','VolatileEP'};
happy.modelFrequencyAll = array2table(happy.modelFrequencyAll,'RowNames',rowNames,'VariableNames',colNames);


%% Happiness parameter values
for imodel=1:nmodel
    paramall{1,imodel}(:,:) = [happyData{1}.param_est{1,imodel};happyData{2}.param_est{2,imodel}];
    paramall{2,imodel}(:,:) = [happyData{1}.param_est{2,imodel};happyData{2}.param_est{1,imodel}];
end
for imodel=1:nmodel
    if ismember(imodel,[1:4 9])
        lparam = 1:3;
    elseif ismember(imodel,[10])
        lparam = 1:4;
    elseif ismember(imodel,[5 6 7 8 11 13 12])
        lparam = [1 3 2 4];% makes sure gammas are always is second position
%     elseif ismember(imodel,12)
%         lparam = [1 4 2 3 5];
    end
    for iparam=1:length(lparam)
       % stable
       happy.param(imodel,iparam,1) = mean(paramall{1,imodel}(:,lparam(iparam)));
       happy.param(imodel,iparam,2) = std(paramall{1,imodel}(:,lparam(iparam)))./sqrt(size(paramall{1,imodel}(:,:),1));
       [p,h,stats] = signrank(paramall{1,imodel}(:,lparam(iparam))-1*double(imodel==12 & iparam==4))
       happy.param(imodel,iparam,3) = stats.signedrank;
       happy.param(imodel,iparam,4) = stats.zval;
       happy.param(imodel,iparam,5) = p;
       % volatile
       happy.param(imodel,iparam,6) = mean(paramall{2,imodel}(:,lparam(iparam)));
       happy.param(imodel,iparam,7) = std(paramall{2,imodel}(:,lparam(iparam)))./sqrt(size(paramall{1,imodel}(:,:),1));
       [p,h,stats] = signrank(paramall{2,imodel}(:,lparam(iparam))-1*double(imodel==12 & iparam==4))
       happy.param(imodel,iparam,8) = stats.signedrank;
       happy.param(imodel,iparam,9) = stats.zval;
       happy.param(imodel,iparam,10) = p;
       % difference
       paramdiff = paramall{2,imodel}(:,lparam(iparam))-paramall{1,imodel}(:,lparam(iparam));
       happy.param(imodel,iparam,11) = mean(paramdiff);
       happy.param(imodel,iparam,12) = std(paramdiff)./sqrt(size(paramdiff,1));
       [p,h,stats] = signrank(paramdiff)
       happy.param(imodel,iparam,13) = stats.signedrank;
       happy.param(imodel,iparam,14) = stats.zval;
       happy.param(imodel,iparam,15) = p;
    end
end
colNames = {'StableMean','StableSE','StableTvalue','StableZvalue','StablePvalue','VolatileMean','VolatileSE','VolatileTvalue','VolatileZvalue','VolatilePvalue','DifferenceMean','DifferenceSE','Tvalue','Zvalue','Pvalue'};
rowNames = model_names(1:nmodel);
happy.paramW1 = array2table(squeeze(happy.param(:,1,:)),'RowNames',rowNames,'VariableNames',colNames);
happy.paramW2 = array2table([nan(4,15);squeeze(happy.param(5:8,3,:));nan(1,15);squeeze(happy.param(10:13,3,:));],'RowNames',rowNames,'VariableNames',colNames);
happy.paramW3 = array2table([nan(11,15);squeeze(happy.param(12,3,:))';nan(1,15)],'RowNames',rowNames,'VariableNames',colNames);
%happy.paramW4 = array2table([nan(11,15);squeeze(happy.param(12,4,:))';nan(1,15)],'RowNames',rowNames,'VariableNames',colNames);
happy.paramGamma = array2table(squeeze(happy.param(:,2,:)),'RowNames',rowNames,'VariableNames',colNames);

%%% gamma
for imodel=1:4
   happy.PE_gamma(imodel,1) = mean(paramall{1,imodel}(:,2));
   happy.PE_gamma(imodel,2) = std(paramall{1,imodel}(:,2))./sqrt(size(paramall{1,imodel}(:,:),2));
   happy.PE_gamma(imodel,3) = mean(paramall{2,imodel}(:,2));
   happy.PE_gamma(imodel,4) = std(paramall{2,imodel}(:,2))./sqrt(size(paramall{1,imodel}(:,:),2));
    paramdiff = paramall{2,imodel}(:,2)-paramall{1,imodel}(:,2);
   happy.PE_gamma(imodel,5) = mean(paramdiff);
   happy.PE_gamma(imodel,6) = std(paramdiff)./sqrt(size(paramdiff,2));
   [p,h,stats] = signrank(paramdiff)
   happy.PE_gamma(imodel,7) = stats.signedrank;
   happy.PE_gamma(imodel,8) = stats.zval;
   happy.PE_gamma(imodel,9) = p;
end
colNames = {'StableMean','StableSE','VolatileMean','VolatileSE','DifferenceMean','DifferenceSE','Tvalue','Zvalue','Pvalue'};
rowNames = model_names(1:4);
happy.PE_gamma = array2table(happy.PE_gamma,'RowNames',rowNames,'VariableNames',colNames);

%% Is there an expectation effect on happiness? 
% differece between P and PPE
happy.PPEvsP(1) = mean(paramall{1,5}(:,2) - paramall{1,5}(:,1));
happy.PPEvsP(2) = std(paramall{1,5}(:,2) - paramall{1,5}(:,1))./sqrt(size(paramall{1,5}(:,1),1));
[p,h,stats] = signrank(paramall{1,5}(:,2) - paramall{1,5}(:,1))
happy.PPEvsP(3) = stats.signedrank;
happy.PPEvsP(4) = stats.zval;
happy.PPEvsP(5) = p;
happy.PPEvsP(6) = mean(paramall{2,5}(:,2) - paramall{2,5}(:,1));
happy.PPEvsP(7) = std(paramall{2,5}(:,2) - paramall{2,5}(:,1))./sqrt(size(paramall{2,5}(:,1),1));
[p,h,stats] = signrank(paramall{2,5}(:,2) - paramall{2,5}(:,1))
happy.PPEvsP(8) = stats.signedrank;
happy.PPEvsP(9) = stats.zval;
happy.PPEvsP(10) = p;
happy.PPEvsP(11) = mean((paramall{2,5}(:,2) - paramall{2,5}(:,1)) - (paramall{1,5}(:,2) - paramall{1,5}(:,1)));
happy.PPEvsP(12) = std((paramall{2,5}(:,2) - paramall{2,5}(:,1)) - (paramall{1,5}(:,2) - paramall{1,5}(:,1)))./sqrt(size(paramall{2,5}(:,1),1));
[p,h,stats] = signrank((paramall{2,5}(:,2) - paramall{2,5}(:,1)) - (paramall{1,5}(:,2) - paramall{1,5}(:,1)))
happy.PPEvsP(13) = stats.signedrank;
happy.PPEvsP(14) = stats.zval;
happy.PPEvsP(15) = p;
colNames = {'StableMean','StableSE','StableTvalue','StableZvalue','StablePvalue','VolatileMean','VolatileSE','VolatileTvalue','VolatileZvalue','VolatilePvalue','DifferenceMean','DifferenceSE','Tvalue','Zvalue','Pvalue'};
rowNames = model_names(5);
happy.PPEvsP = array2table(happy.PPEvsP,'RowNames',rowNames,'VariableNames',colNames);

%%% model frequency
BIC{1} = [happyData{1}.matBIC{1};happyData{2}.matBIC{2}];
BIC{2} = [happyData{1}.matBIC{2};happyData{2}.matBIC{1}];
ioi = [5 11];
options.DisplayWin = 0;
[posterior,outS] = VBA_groupBMC(-BIC{1}(:,ioi)',options); 
[posterior,outV] = VBA_groupBMC(-BIC{2}(:,ioi)',options) ;
happy.PPPEvsWinLoss(1,1) = outS.Ef(1);
happy.PPPEvsWinLoss(1,2) = sqrt(outS.Vf(1,1));
happy.PPPEvsWinLoss(1,3) = outS.Ef(2);
happy.PPPEvsWinLoss(1,4) = sqrt(outS.Vf(2,2));
happy.PPPEvsWinLoss(1,5) = outS.ep(1);
happy.PPPEvsWinLoss(2,1) = outV.Ef(1);
happy.PPPEvsWinLoss(2,2) = sqrt(outV.Vf(1,1));
happy.PPPEvsWinLoss(2,3) = outV.Ef(2);
happy.PPPEvsWinLoss(2,4) = sqrt(outV.Vf(2,2));
happy.PPPEvsWinLoss(2,5) = outV.ep(1);
colNames = {'pppeEF','pppeSD','wlEF','wlSD','EP'};
rowNames = {'Stable','Volatile'};
happy.PPPEvsWinLoss = array2table(happy.PPPEvsWinLoss,'RowNames',rowNames,'VariableNames',colNames);

%%% residuals analysis
%%%%% #1 residuals of Win-Loss model
imodel=11;
matchosen_estprobaall{1} =  [happyData{1}.matExpectation{1}; happyData{2}.matExpectation{2}];
matchosen_estprobaall{2} =  [happyData{1}.matExpectation{2}; happyData{2}.matExpectation{1}];
mat_happyresall{1} = [happyData{1}.mathappyres{1,imodel};happyData{2}.mathappyres{2,imodel}];
mat_happyresall{2} = [happyData{1}.mathappyres{2,imodel};happyData{2}.mathappyres{1,imodel}];
mat_happyall{1} = [happyData{1}.mathappyalltrials{1};happyData{2}.mathappyalltrials{2}];
mat_happyall{2} = [happyData{1}.mathappyalltrials{2};happyData{2}.mathappyalltrials{1}];
happyInd=[];
rhoProbaAll = [];
for icontext=1:2
    for isub=1:size(matchosen_estprobaall{1},1)
        happyInd{isub}(icontext,:) = ~isnan(mat_happyall{icontext}(isub,:));       
        [rhoProbaAll{icontext}(isub) p] = corr(matchosen_estprobaall{icontext}(isub,happyInd{isub}(icontext,:))',mat_happyresall{icontext}(isub,:)','type','Spearman');
    end
    [p,h,stats] = signrank(rhoProbaAll{icontext})
    happy.ExpectationOnWinLossResiduals(icontext,1) = nanmean(rhoProbaAll{icontext});
    happy.ExpectationOnWinLossResiduals(icontext,2) = nanstd(rhoProbaAll{icontext})./sqrt(size(rhoProbaAll{icontext},2));
    happy.ExpectationOnWinLossResiduals(icontext,3) = stats.signedrank;
    happy.ExpectationOnWinLossResiduals(icontext,4) = stats.zval;
    happy.ExpectationOnWinLossResiduals(icontext,5) = p;  
end
colNames = {'ExpectationM','ExpectationSE','t','z','p'};
rowNames = {'Stable','Volatile'};
happy.ExpectationOnWinLossResiduals = array2table(happy.ExpectationOnWinLossResiduals,'RowNames',rowNames,'VariableNames',colNames);

%%%%% # residuals of Win-Loss models, split by the ten trials before a
%%%%% shift(when uncertainy is low) and after a shift; when uncertainty is
%%%%% high
% get backfilled data to get more power
matbackfilled_happyresall{1} = [happyData{1}.matbackfilled_happyres{1,imodel};happyData{2}.matbackfilled_happyres{2,imodel}];
matbackfilled_happyresall{2} = [happyData{1}.matbackfilled_happyres{2,imodel};happyData{2}.matbackfilled_happyres{1,imodel}];

mat_happyresallTrials{1} = [happyData{1}.mathappyresall{1,imodel};happyData{2}.mathappyresall{2,imodel}];
mat_happyresallTrials{2} = [happyData{1}.mathappyresall{2,imodel};happyData{2}.mathappyresall{1,imodel}];

happyInd=[];
rhoProbaAll = [];
happy.ExpectationOnWinLossResidualsBeforeAfterShift =[];
for icontext=2
    for itrialtype = 1:2
        if itrialtype==1% beginning or after a shift, uncertainty is high
            ltrialtype = [1:10 21:30 41:50 61:70];
        else
            ltrialtype = [11:20 31:40 51:60 71:80];
        end
    for isub=1:size(matchosen_estprobaall{1},1)
        happyInd{isub}(icontext,:) = ~isnan(mat_happyall{icontext}(isub,ltrialtype));
        resTmp = mat_happyresallTrials{icontext}(isub,ltrialtype);%matbackfilled_happyresall{icontext}(isub,ltrialtype);
        [rhoProbaAll{icontext}(isub) p] = corr(matchosen_estprobaall{icontext}(isub,happyInd{isub}(icontext,:))',resTmp (happyInd{isub}(icontext,:))','type','Spearman');
    end
    [p,h,stats] = signrank(rhoProbaAll{icontext});
    happy.ExpectationOnWinLossResidualsBeforeAfterShift(itrialtype,1) = nanmean(rhoProbaAll{icontext});
    happy.ExpectationOnWinLossResidualsBeforeAfterShift(itrialtype,2) = nanstd(rhoProbaAll{icontext})./sqrt(size(rhoProbaAll{icontext},2));
    happy.ExpectationOnWinLossResidualsBeforeAfterShift(itrialtype,3) = stats.signedrank;
    happy.ExpectationOnWinLossResidualsBeforeAfterShift(itrialtype,4) = stats.zval;
    happy.ExpectationOnWinLossResidualsBeforeAfterShift(itrialtype,5) = p;
    end
end
colNames = {'ExpectationM','ExpectationSE','t','z','p'};
rowNames = {'AfterShift','BeforeShift'};
happy.ExpectationOnWinLossResidualsBeforeAfterShift = array2table(happy.ExpectationOnWinLossResidualsBeforeAfterShift,'RowNames',rowNames,'VariableNames',colNames);

%% win-loss model, within environment (between subject) parameter correlation
happy.WinLossCorr = [];
for icontext=1:2
    X = paramall{icontext,11}(:,1);%win
    Y = paramall{icontext,11}(:,2);%loss
    [rho,p] = corr(X,Y,'type','Spearman')
    happy.WinLossCorr(icontext,1) = rho;
    happy.WinLossCorr(icontext,2) = p;
end
colNames = {'rho','p'};
rowNames = {'Stable','Volatile'};
happy.WinLossCorr = array2table(happy.WinLossCorr,'RowNames',rowNames,'VariableNames',colNames);

%% comparing win and loss weight for both environments
happy.WinLossContrast = [];
for icontext=1:2
    X = paramall{icontext,11}(:,1);%win
    Y = paramall{icontext,11}(:,2);%loss
    [p,h,stats] = signrank(X-Y);
    happy.WinLossContrast(icontext,1) = nanmean(X-Y);
    happy.WinLossContrast(icontext,2) = nanstd(X-Y)./sqrt(size(X-Y,1));
    happy.WinLossContrast(icontext,3) = stats.signedrank;
    happy.WinLossContrast(icontext,4) = stats.zval;
    happy.WinLossContrast(icontext,5) = p;    
end
colNames = {'MeanDiff','SEdiff','t','z','p'};
rowNames = {'Stable','Volatile'};
happy.WinLossContrast = array2table(happy.WinLossContrast,'RowNames',rowNames,'VariableNames',colNames);

%% Win-loss does not correlate with task accuracy
happy.WinLossAccuracyPmaxCorr = [];
for icontext=1:2
    X = paramall{icontext,11}(:,1);%win
    Y = paramall{icontext,11}(:,2);%loss    
    accuracyPMax = matProbaAccuracy{icontext};
    [rho,p] = corr(X-Y,accuracyPMax,'type','Spearman')
    happy.WinLossAccuracyPmaxCorr(icontext,1) = rho;
    happy.WinLossAccuracyPmaxCorr(icontext,2) = p;
end
colNames = {'rho','p'};
rowNames = {'Stable','Volatile'};
happy.WinLossAccuracyPmaxCorr = array2table(happy.WinLossAccuracyPmaxCorr,'RowNames',rowNames,'VariableNames',colNames);

happy.WinLossAccuracyEVmaxCorr = [];
for icontext=1:2
    X = paramall{icontext,11}(:,1);%win
    Y = paramall{icontext,11}(:,2);%loss    
    accuracyEVMax = matEVAccuracy{icontext};
    [rho,p] = corr(X-Y,accuracyEVMax,'type','Spearman')
    happy.WinLossAccuracyEVmaxCorr(icontext,1) = rho;
    happy.WinLossAccuracyEVmaxCorr(icontext,2) = p;
end
colNames = {'rho','p'};
rowNames = {'Stable','Volatile'};
happy.WinLossAccuracyEVmaxCorr = array2table(happy.WinLossAccuracyEVmaxCorr,'RowNames',rowNames,'VariableNames',colNames);
%% correlation between choice learning rate & happiness P+PPE parameters
% learning rates
alphaAll{1} = [choiceData{1}.param_est{1,1}(:,1);choiceData{2}.param_est{2,1}(:,1)];
alphaAll{2} = [choiceData{1}.param_est{2,1}(:,1);choiceData{2}.param_est{1,1}(:,1)];
for icontext=1:3
    if icontext<=2 % stable & volatile
        X = paramall{icontext,5}(:,2);%wPPE
        Y = paramall{icontext,5}(:,3);%Gamma
        Z = alphaAll{icontext};
        [rx px] = corr(X,Z,'type','Spearman')
        happy.moodLearningCorr(icontext,1) = rx;
        happy.moodLearningCorr(icontext,2) = px;
        [ry py] = corr(Y,Z,'type','Spearman')
        happy.moodLearningCorr(icontext,3) = ry;
        happy.moodLearningCorr(icontext,4) = py;
    else % volatile - stable
        X = paramall{2,5}(:,2)-paramall{1,5}(:,2);%wPPE
        Y = paramall{2,5}(:,3)-paramall{1,5}(:,3);%Gamma
        Z = alphaAll{2}-alphaAll{1};
        [rx px] = corr(X,Z,'type','Spearman')
        happy.moodLearningCorr(icontext,1) = rx;
        happy.moodLearningCorr(icontext,2) = px;
        [ry py] = corr(Y,Z,'type','Spearman')
        happy.moodLearningCorr(icontext,3) = ry;
        happy.moodLearningCorr(icontext,4) = py;
    end
end
colNames = {'SpearmanRhoPPE','PvaluePPE','SpearmanRhoGamma','PvalueGamma'};
rowNames = {'Stable','Volatile','Difference'};
happy.moodLearningCorr = array2table(happy.moodLearningCorr,'RowNames',rowNames,'VariableNames',colNames);

%% correlation between choice accuracy & happiness P+PPE parameters
% proba max
for icontext=1:3
    if icontext<=2 % stable & volatile
        X = paramall{icontext,5}(:,2);%wPPE
        Y = paramall{icontext,5}(:,3);%Gamma
        Z = matProbaAccuracy{icontext};
        [rx px] = corr(X,Z,'type','Spearman')
        happy.moodProbaMaxCorr(icontext,1) = rx;
        happy.moodProbaMaxCorr(icontext,2) = px;
        [ry py] = corr(Y,Z,'type','Spearman')
        happy.moodProbaMaxCorr(icontext,3) = ry;
        happy.moodProbaMaxCorr(icontext,4) = py;
    else % volatile - stable
        X = paramall{2,5}(:,2)-paramall{1,5}(:,2);%wPPE
        Y = paramall{2,5}(:,3)-paramall{1,5}(:,3);%Gamma
        Z = matProbaAccuracy{2}-matProbaAccuracy{1};
        [rx px] = corr(X,Z,'type','Spearman')
        happy.moodProbaMaxCorr(icontext,1) = rx;
        happy.moodProbaMaxCorr(icontext,2) = px;
        [ry py] = corr(Y,Z,'type','Spearman')
        happy.moodProbaMaxCorr(icontext,3) = ry;
        happy.moodProbaMaxCorr(icontext,4) = py;
    end
end
colNames = {'SpearmanRhoPPE','PvaluePPE','SpearmanRhoGamma','PvalueGamma'};
rowNames = {'Stable','Volatile','Difference'};
happy.moodProbaMaxCorr = array2table(happy.moodProbaMaxCorr,'RowNames',rowNames,'VariableNames',colNames);

% EV max
for icontext=1:3
    if icontext<=2 % stable & volatile
        X = paramall{icontext,5}(:,2);%wPPE
        Y = paramall{icontext,5}(:,3);%Gamma
        Z = matEVAccuracy{icontext};
        [rx px] = corr(X,Z,'type','Spearman')
        happy.moodEVMaxCorr(icontext,1) = rx;
        happy.moodEVMaxCorr(icontext,2) = px;
        [ry py] = corr(Y,Z,'type','Spearman')
        happy.moodEVMaxCorr(icontext,3) = ry;
        happy.moodEVMaxCorr(icontext,4) = py;
    else % volatile - stable
        X = paramall{2,5}(:,2)-paramall{1,5}(:,2);%wPPE
        Y = paramall{2,5}(:,3)-paramall{1,5}(:,3);%Gamma
        Z = matEVAccuracy{2}-matEVAccuracy{1};
        [rx px] = corr(X,Z,'type','Spearman')
        happy.moodEVMaxCorr(icontext,1) = rx;
        happy.moodEVMaxCorr(icontext,2) = px;
        [ry py] = corr(Y,Z,'type','Spearman')
        happy.moodEVMaxCorr(icontext,3) = ry;
        happy.moodEVMaxCorr(icontext,4) = py;
    end
end
colNames = {'SpearmanRhoPPE','PvaluePPE','SpearmanRhoGamma','PvalueGamma'};
rowNames = {'Stable','Volatile','Difference'};
happy.moodEVMaxCorr = array2table(happy.moodEVMaxCorr,'RowNames',rowNames,'VariableNames',colNames);

%% between session parameter correlation
for iparam=1:2    
    Y = paramall{2,5}(:,iparam+1);
    X = paramall{1,5}(:,iparam+1);
    [r p] = corr(X,Y,'type','Spearman')
   happy.paramCorr(iparam,1) = r;
   happy.paramCorr(iparam,2) = p;
end
colNames = {'SpearmanRho','Pvalue'};
rowNames = {'PPEweight','ForgettingFactor'};
happy.paramCorr = array2table(happy.paramCorr,'RowNames',rowNames,'VariableNames',colNames);
%% clinical parameters
scorePHQvall  = [happyData{1}.scorePHQ';happyData{2}.scorePHQ'];
scoreSTAIvall = [happyData{1}.scoreSTAI';happyData{2}.scoreSTAI'];

%% average happiness (stats for figure 6A)
matmeanrawhappyall{1} = [happyData{1}.matmean{1}';happyData{2}.matmean{2}']; 
matmeanrawhappyall{2} = [happyData{1}.matmean{2}';happyData{2}.matmean{1}'];
for i=1:3
    X    = scorePHQvall;
    if i<3
        Y = matmeanrawhappyall{i}
    elseif i==3
        Y = matmeanrawhappyall{2}-matmeanrawhappyall{1};
    end
    % get the stats
    [r p] = corr(X,Y,'type','Spearman')
    happy.meanClinicalRho(1,i+3) = r;
    happy.meanClinicalP(1,i+3) = p;
    % add the stats for STAI
    [r p] = corr(scoreSTAIvall,Y,'type','Spearman')
    happy.meanClinicalRho(1,i) = r;
    happy.meanClinicalP(1,i) = p;
end
colNames = {'STAIstable','STAIvolatile','STAIvolatilestable','PHQstable','PHQvolatile','PHQvolatilestable'};
rowNames = {'Rho'};
happy.meanClinicalRho = array2table(happy.meanClinicalRho,'RowNames',rowNames,'VariableNames',colNames);
colNames = {'STAIstable','STAIvolatile','STAIvolatilestable','PHQstable','PHQvolatile','PHQvolatilestable'};
rowNames = {'Pvalue'};
happy.meanClinicalP   = array2table(happy.meanClinicalP,'RowNames',rowNames,'VariableNames',colNames);
Y  = scorePHQvall;
X1 = matmeanrawhappyall{1};
X2 = matmeanrawhappyall{2};
Z1 = (X1-mean(X1))./std(X1);
Z2 = (X2-mean(X2))./std(X2);
D  = Z2-Z1;
[r p] = corr(D,Y,'type','Spearman')
happy.meanClinicalRhoDiff = r;
happy.meanClinicalPDiff   = p;

%% constant term (stats for figure 6B) and other parameters 
imodel=5;% get the best model
for itest=1:3
    for iparam=1:4
         X = zscore(scorePHQvall);
         Z = zscore(scoreSTAIvall);
         if itest<3
             Y = (paramall{itest,imodel}(:,iparam));
         else
             Y = (paramall{2,imodel}(:,iparam))-(paramall{1,imodel}(:,iparam));
         end
         [r p] = corr(Z,Y,'type','Spearman')
        happy.paramClinicalRho(iparam,itest)   = r;
        happy.paramClinicalP(iparam,itest)     = p;
         [r p] = corr(X,Y,'type','Spearman')
        happy.paramClinicalRho(iparam,itest+3) = r;
        happy.paramClinicalP(iparam,itest+3)   = p;
    end
end
Y  = scorePHQvall;
X1 = paramall{1,5}(:,4);
X2 = paramall{2,5}(:,4);
Z1 = (X1-mean(X1))./std(X1);
Z2 = (X2-mean(X2))./std(X2);
D  = Z2-Z1;
[r p] = corr(D,Y,'type','Spearman')
happy.paramClinicalRhoDiff = r;
happy.paramClinicalPDiff   = p;

% t = (r1-r2)/(sqrt([n-1)(1+r12)]/[2*((n-1)/n-3)*det(R)-(r)^2(1-r12)^3))]))
colNames = {'STAIstable','STAIvolatile','STAIvolatilestable','PHQstable','PHQvolatile','PHQvolatilestable'};
rowNames = {'PhatWeight','PPEhatWeight','ForgettingFactor','Constant'};
happy.paramClinicalRho = array2table(happy.paramClinicalRho,'RowNames',rowNames,'VariableNames',colNames);
happy.paramClinicalP   = array2table(happy.paramClinicalP,'RowNames',rowNames,'VariableNames',colNames);

%% supplementary figure 3: Kernel
kernel_PPEhat_all{1} = [happyData{1}.kernel_est{1,1};happyData{2}.kernel_est{2,1}];
kernel_PPEhat_all{2} = [happyData{1}.kernel_est{2,1};happyData{2}.kernel_est{1,1}];
kernel_PPE_all{1} = [happyData{1}.kernel_est{1,2};happyData{2}.kernel_est{2,2}];
kernel_PPE_all{2} = [happyData{1}.kernel_est{2,2};happyData{2}.kernel_est{1,2}];
for i=1:10
[p(i),h,stats(i)] = signrank(kernel_PPEhat_all{1}(:,i));
end
happy.forgettingFactorKernel(1,:) = p;
for i=1:10
[p(i),h,stats(i)] = signrank(kernel_PPEhat_all{2}(:,i));
end
happy.forgettingFactorKernel(2,:) = p;
for i=1:10
[p(i),h,stats(i)] = signrank(kernel_PPE_all{1}(:,i));
end
happy.forgettingFactorKernel(3,:) = p;
for i=1:10
[p(i),h,stats(i)] = signrank(kernel_PPE_all{2}(:,i));
end
happy.forgettingFactorKernel(4,:) = p;

colNames = {'t-1','t-2','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10'};
rowNames = {'PPEhatStableP','PPEhatVolatileP',...
           'PPEStableP','PPEVolatileP'};
happy.forgettingFactorKernel = array2table(happy.forgettingFactorKernel,'RowNames',rowNames,'VariableNames',colNames);

