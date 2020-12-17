function  [choice] = MoodTrackLearning_choice_doStats(allChoiceData)
% Bastien Blain, bastien.blain@gmail.com
% February, 2019
%% Choice accuracy table for all choice and incongruent

% Proba
maxProbaall{1} = [allChoiceData{1}.matperfProba(:,1);allChoiceData{2}.matperfProba(:,2);].*100;
maxProbaall{2} = [allChoiceData{1}.matperfProba(:,2);allChoiceData{2}.matperfProba(:,1);].*100;
for itest=1:2
    X=maxProbaall{itest};
    [p,h,stats] = signrank(X-50);
    choice.Probamax(itest,1) = nanmean(X);
    choice.Probamax(itest,2) = nanstd(X)./sqrt(size(X,1));
    choice.Probamax(itest,3) = stats.signedrank;
    choice.Probamax(itest,4) = stats.zval;
    choice.Probamax(itest,5) = p;
end
colNames = {'Mean','SE','t-value','z-value','p-value'};
rowNames = {'Stable','Volatile'}
choice.Probamax=array2table(choice.Probamax,'RowNames',rowNames,'VariableNames',colNames);

choice.allProbamax{1} = maxProbaall{1};
choice.allProbamax{2} = maxProbaall{2};

% All
maxEVall{1} = [allChoiceData{1}.matperfEV(:,1);allChoiceData{2}.matperfEV(:,2);].*100;
maxEVall{2} = [allChoiceData{1}.matperfEV(:,2);allChoiceData{2}.matperfEV(:,1);].*100;
for itest=1:2
    X=maxEVall{itest};
    [p,h,stats] = signrank(X-50);
    choice.EVmax(itest,1) = nanmean(X);
    choice.EVmax(itest,2) = nanstd(X)./sqrt(size(X,1));
    choice.EVmax(itest,3) = stats.signedrank;
    choice.EVmax(itest,4) = stats.zval;
    choice.EVmax(itest,5) = p;
end

% Incongruent
maxEVincextall{1} = [allChoiceData{1}.maxEVincext(:,1);allChoiceData{2}.maxEVincext(:,2);].*100;;
maxEVincextall{2} = [allChoiceData{1}.maxEVincext(:,2);allChoiceData{2}.maxEVincext(:,1);].*100;;
for itest=3:4
    X=maxEVincextall{itest-2};
    [p,h,stats] = signrank(X-50);
    choice.EVmax(itest,1) = nanmean(X);
    choice.EVmax(itest,2) = nanstd(X)./sqrt(size(X,1));
    choice.EVmax(itest,3) = stats.signedrank;
    choice.EVmax(itest,4) = stats.zval;
    choice.EVmax(itest,5) = p;
end
colNames = {'Mean','SE','t-value','z-value','p-value'};
rowNames = {'All trials, Stable','All trials, Volatile','Incongruent trials, stable','Incongruent trials, volatile'}
choice.EVmax=array2table(choice.EVmax,'RowNames',rowNames,'VariableNames',colNames);

%% Table 1. Choice model comparison results
BICall{1} = [allChoiceData{1}.matBIC{1};allChoiceData{2}.matBIC{2}];
BICall{2} = [allChoiceData{1}.matBIC{2};allChoiceData{2}.matBIC{1}];
r2all{1}  = [allChoiceData{1}.matr2{1};allChoiceData{2}.matr2{2}];
r2all{2}  = [allChoiceData{1}.matr2{2};allChoiceData{2}.matr2{1}];

for imodel=1:4
    choice.modelComparison(imodel,1) = allChoiceData{1}.dof(imodel);
    choice.modelComparison(imodel,2) = mean(r2all{1}(:,imodel));
    choice.modelComparison(imodel,3) = std(r2all{1}(:,imodel))./sqrt(size(r2all{1}(:,imodel),1));
    choice.modelComparison(imodel,4) = mean(r2all{2}(:,imodel));
    choice.modelComparison(imodel,5) = std(r2all{2}(:,imodel))./sqrt(size(r2all{2}(:,imodel),1));
    choice.modelComparison(imodel,6) = sum(BICall{1}(:,imodel));
    choice.modelComparison(imodel,7) = sum(BICall{2}(:,imodel));
    choice.modelComparison(imodel,8) = sum(BICall{1}(:,imodel))-sum(BICall{1}(:,1));
    choice.modelComparison(imodel,9) = sum(BICall{2}(:,imodel))-sum(BICall{2}(:,1));
end
colNames = {'NumberOfParameters','MeanStablePseudor2','SEstablePseudor2','MeanVolatilePseudor2','SEvolatilePseudor2',...
    'StableBIC','VolatileBIC','StableDeltaBIC','VolatileDeltaBIC'};
rowNames = {'Additive','Multiplicative','ProbabilityOnly','MagnitudeOnly'};
choice.modelComparison = array2table(choice.modelComparison,'RowNames',rowNames,'VariableNames',colNames);

%% Supplementary table 1: Choice parameter values
for imodel=1:4
    paramall{1}(imodel,:,:) = [allChoiceData{1}.param_est{1,imodel};allChoiceData{2}.param_est{2,imodel}];
    paramall{2}(imodel,:,:) = [allChoiceData{1}.param_est{2,imodel};allChoiceData{2}.param_est{1,imodel}];
end
for imodel=1:2
    choice.Alpha(imodel,1) = mean(paramall{1}(imodel,:,1));
    choice.Alpha(imodel,2) = std(paramall{1}(imodel,:,1))./sqrt(size(paramall{1}(imodel,:,:),2));
    choice.Alpha(imodel,3) = mean(paramall{2}(imodel,:,1));
    choice.Alpha(imodel,4) = std(paramall{2}(imodel,:,1))./sqrt(size(paramall{1}(imodel,:,:),2));
    paramdiff = paramall{2}(imodel,:,1)-paramall{1}(imodel,:,1);
    choice.Alpha(imodel,5) = mean(paramdiff);
    choice.Alpha(imodel,6) = std(paramdiff)./sqrt(size(paramdiff,2));
    [p,h,stats] = signrank(paramdiff);
    choice.Alpha(imodel,7) = stats.signedrank;
    choice.Alpha(imodel,8) = stats.zval;
    choice.Alpha(imodel,9) = p;
end
colNames = {'StableMean','StableSE','VolatileMean','VolatileSE','DifferenceMean','DifferenceSE','Tvalue','Zvalue','Pvalue'};
rowNames = {'Additive','Multiplicative'};
choice.Alpha = array2table(choice.Alpha,'RowNames',rowNames,'VariableNames',colNames);

for imodel=1:2
    choice.Beta(imodel,1) = mean(paramall{1}(imodel,:,2));
    choice.Beta(imodel,2) = std(paramall{1}(imodel,:,2))./sqrt(size(paramall{1}(imodel,:,:),2));
    choice.Beta(imodel,3) = mean(paramall{2}(imodel,:,2));
    choice.Beta(imodel,4) = std(paramall{2}(imodel,:,2))./sqrt(size(paramall{1}(imodel,:,:),2));
    paramdiff = paramall{2}(imodel,:,2)-paramall{1}(imodel,:,2);
    choice.Beta(imodel,5) = mean(paramdiff);
    choice.Beta(imodel,6) = std(paramdiff)./sqrt(size(paramdiff,2));
    [p,h,stats] = signrank(paramdiff);
    choice.Beta(imodel,7) = stats.signedrank;
    choice.Beta(imodel,8) = stats.zval;
    choice.Beta(imodel,9) = p;
end
colNames = {'StableMean','StableSE','VolatileMean','VolatileSE','DifferenceMean','DifferenceSE','Tvalue','Zvalue','Pvalue'};
rowNames = {'Additive','Multiplicative'};
choice.Beta = array2table(choice.Beta,'RowNames',rowNames,'VariableNames',colNames);

for imodel=1:2
    choice.Phi(imodel,1) = mean(paramall{1}(imodel,:,3));
    choice.Phi(imodel,2) = std(paramall{1}(imodel,:,3))./sqrt(size(paramall{1}(imodel,:,:),2));
    choice.Phi(imodel,3) = mean(paramall{2}(imodel,:,3));
    choice.Phi(imodel,4) = std(paramall{2}(imodel,:,3))./sqrt(size(paramall{1}(imodel,:,:),2));
    paramdiff = paramall{2}(imodel,:,3)-paramall{1}(imodel,:,3);
    choice.Phi(imodel,5) = mean(paramdiff);
    choice.Phi(imodel,6) = std(paramdiff)./sqrt(size(paramdiff,2));
    [p,h,stats] = signrank(paramdiff);
    choice.Phi(imodel,7) = stats.signedrank;
    choice.Phi(imodel,9) = stats.zval;
    choice.Phi(imodel,8) = p;
end
colNames = {'StableMean','StableSE','VolatileMean','VolatileSE','DifferenceMean','DifferenceSE','Tvalue','Zvalue','Pvalue'};
rowNames = {'Additive','Multiplicative'};
choice.Phi = array2table(choice.Phi,'RowNames',rowNames,'VariableNames',colNames);

%% win/stay lose/stay stats
meanchoice(:,[1:4]) = [allChoiceData{1}.meanchoice{1};allChoiceData{2}.meanchoice{2}];
meanchoice(:,[5:8]) = [allChoiceData{1}.meanchoice{2};allChoiceData{2}.meanchoice{1}];
for imodel=1:4
    meanprob{imodel}(:,1:4) = [allChoiceData{1}.meanprob{1,imodel};allChoiceData{2}.meanprob{2,imodel}];
    meanprob{imodel}(:,5:8) = [allChoiceData{1}.meanprob{2,imodel};allChoiceData{2}.meanprob{1,imodel}];
end
for itest=1:6
    if itest<=4
        idx = [1:2]+(2*(itest-1));
        X   = meanchoice(:,idx(1)).*100;
        Y   = meanchoice(:,idx(2)).*100;
        [p,h,stats] = signrank(X-Y);
        choice.winstay(itest,1) = nanmean(X-Y);
        choice.winstay(itest,2) = nanstd(X-Y)./sqrt(size(X,1));
        choice.winstay(itest,3) = stats.signedrank;
        choice.winstay(itest,4) = stats.zval;
        choice.winstay(itest,5) = p;
    elseif itest>4
        idx = itest+1*mod(itest,5)
        Y = meanchoice(:,idx-4).*100 - meanchoice(:,idx-3).*100;
        X = meanchoice(:,idx).*100   - meanchoice(:,idx+1).*100;
        [p,h,stats] = signrank(X-Y);
        choice.winstay(itest,1) = nanmean(X-Y);
        choice.winstay(itest,2) = nanstd(X-Y)./sqrt(size(X,1));
        choice.winstay(itest,3) = stats.signedrank;
        choice.winstay(itest,4) = stats.zval;
        choice.winstay(itest,5) = p;
    end
end
colNames = {'MeanDifference','SEDifference','Tvalue','Zvalue','Pvalue'};
rowNames = {'StableHighProba','StableLowProba','VolatileHighProba','VolatileLowProba','VolatileStableHighProba','VolatileStableLowProba'};
choice.winstay = array2table(choice.winstay,'RowNames',rowNames,'VariableNames',colNames);

imodel=1;
for itest=1:6
    if itest<=4
        idx = [1:2]+(2*(itest-1));
        Y   = meanprob{imodel}(:,idx(1)).*100;
        X   = meanprob{imodel}(:,idx(2)).*100;
        [p,h,stats] = signrank(X-Y);
        choice.winstayPred(itest,1) = nanmean(X-Y);
        choice.winstayPred(itest,2) = nanstd(X-Y)./sqrt(size(X,1));
        choice.winstayPred(itest,3) = stats.signedrank;
        choice.winstayPred(itest,4) = stats.zval;
        choice.winstayPred(itest,5) = p;
    elseif itest>4
        idx = itest+1*mod(itest,5)
        X = meanprob{imodel}(:,idx-4).*100 - meanprob{imodel}(:,idx-3).*100;
        Y = meanprob{imodel}(:,idx).*100   - meanprob{imodel}(:,idx+1).*100;
        [p,h,stats] = signrank(X-Y);
        choice.winstayPred(itest,1) = nanmean(X-Y);
        choice.winstayPred(itest,2) = nanstd(X-Y)./sqrt(size(X,1));
        choice.winstayPred(itest,3) = stats.signedrank;
        choice.winstayPred(itest,4) = stats.zval;
        choice.winstayPred(itest,5) = p;        
    end
end
colNames = {'MeanDifference','SEDifference','Tvalue','Zvalue','Pvalue'};
rowNames = {'StableHighProba','StableLowProba','VolatileHighProba','VolatileLowProba','VolatileStableHighProba','VolatileStableLowProba'};
choice.winstayPred = array2table(choice.winstayPred,'RowNames',rowNames,'VariableNames',colNames);


%% choice accuracy and clinical scores
score{1} = [allChoiceData{1}.scoreSTAI';allChoiceData{2}.scoreSTAI'];
score{2}  = [allChoiceData{1}.scorePHQ';allChoiceData{2}.scorePHQ'];
   choice.AccuracyClinical=[];
idx = 1:6;
for iquest=1:2
    for iparam = 1:2
        for icontext=1:2
            if iparam==1
                Y = maxProbaall{icontext};
            else
                Y = maxEVall{icontext};
            end
            [rho(icontext,iparam), p(icontext,iparam)] = corr(score{iquest},Y,'type','Spearman');
        end
        [rho(3,1), p(3,1)] = corr(score{iquest},maxProbaall{2}-maxProbaall{1},'type','Spearman');
        [rho(3,2), p(3,2)] = corr(score{iquest},maxEVall{2}-maxEVall{1},'type','Spearman');
        %store the correlations output
        choice.AccuracyClinical(iparam,idx(1)) = rho(1,iparam);
        choice.AccuracyClinical(iparam,idx(2)) = rho(2,iparam);
        choice.AccuracyClinical(iparam,idx(3)) = rho(3,iparam);
        choice.AccuracyClinical(iparam,idx(4)) = p(1,iparam);
        choice.AccuracyClinical(iparam,idx(5)) = p(2,iparam);
        choice.AccuracyClinical(iparam,idx(6)) = p(3,iparam);

    end
    idx = idx + 6;
end

colNames = {'STAIStable','STAIVolatile','STAIDiff','PHQStable','PHQVolatile','PHQDiff'};
rowNames = {'ProbaMax','EVMax'};
choice.AccuracyClinicalRho = array2table(choice.AccuracyClinical(:,[1:3 7:9]),'RowNames',rowNames,'VariableNames',colNames);
choice.AccuracyClinicalP = array2table(choice.AccuracyClinical(:,[4:6 10:12]),'RowNames',rowNames,'VariableNames',colNames);
%% Choice model parameters and clinical scores
%keep the best model
for imodel=1
    idx = 1:6;
    for iquest=1:2
        for iparam = 1:3
            for icontext=1:2
                poi{icontext}   = paramall{icontext}(imodel,:,iparam);% paramter of interest
                [rho(icontext,iparam), p(icontext,iparam)] = corr(score{iquest},poi{icontext}','type','Spearman');
            end
            [rho(3,iparam), p(3,iparam)] = corr(score{iquest},poi{2}'-poi{1}','type','Spearman');
            
            %store the correlations output
            choice.ParamClinical(iparam,idx(1)) = rho(1,iparam);
            choice.ParamClinical(iparam,idx(2)) = rho(2,iparam);
            choice.ParamClinical(iparam,idx(3)) = rho(3,iparam);
            choice.ParamClinical(iparam,idx(4)) = p(1,iparam);
            choice.ParamClinical(iparam,idx(5)) = p(2,iparam);
            choice.ParamClinical(iparam,idx(6)) = p(3,iparam);
        end
        idx = idx + 6;
    end
end
colNames = {'STAIStable','STAIVolatile','STAIDiff','PHQStable','PHQVolatile','PHQDiff'};
rowNames = {'Alpha','Beta','Phi'};
choice.ParamClinicalRho = array2table(choice.ParamClinical(:,[1:3 7:9]),'RowNames',rowNames,'VariableNames',colNames);
choice.ParamClinicalP = array2table(choice.ParamClinical(:,[4:6 10:12]),'RowNames',rowNames,'VariableNames',colNames);

% save('allStats','allStats')