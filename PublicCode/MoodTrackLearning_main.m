clear all
close all

%% fit & store the data in the appropriate format
%% CHOICE
matinx{1,1} = [0.5     1    0.5  0.5
               0.5     1    1    0.5
               0.5     1    1    0.5
               1       1    0    0.5];
matlb{1,1} = [0.01     0    0   0.5%additive
              0.01     0    .7  0.5%multiplicative
              0.01     0    1   0.5%additive, probability only
              1        0    0   0.5];%additive, magnitude only
matub{1,1} = [0.95    50    1   0.5
              0.95    50    1.3  0.5
              0.95    50    1    0.5
              1       50    0    0.5];
for igroup=1:2
    for icontext=1:2
        matinx{igroup,icontext} = matinx{1,1};
        matlb{igroup,icontext}  = matlb{1,1};
        matub{igroup,icontext}  = matub{1,1};
    end
end
% choice -per session
ncontext = 2; %1 means that the data are fitted for both sessions simultaneously
[choiceDataSession] = MoodTrackLearning_choice_getdata(matinx,matlb,matub,ncontext);

% choice -simultaneous fit
ncontext = 1; %1 means that the data are fitted for both sessions simultaneously
[choiceDataAll] = MoodTrackLearning_choice_getdata(matinx,matlb,matub,ncontext);

% choice -per session, swapping the learning rate
matinx = [];
matlb  = [];
matub  = [];
for igroup=1:2
    for icontext=1:2
        for imodel=1:4
            for iparam=1:4
                if iparam==1 % swapping the learning rate
                    matinx{igroup,icontext}(:,iparam) = choiceDataSession{igroup}.param_est{3-icontext,imodel}(:,iparam);
                    matlb{igroup,icontext}(:,iparam)  = choiceDataSession{igroup}.param_est{3-icontext,imodel}(:,iparam);
                    matub{igroup,icontext}(:,iparam)  = choiceDataSession{igroup}.param_est{3-icontext,imodel}(:,iparam);
                else %the other paramters stay the same
                    matinx{igroup,icontext}(:,iparam) = choiceDataSession{igroup}.param_est{icontext,imodel}(:,iparam);
                    matlb{igroup,icontext}(:,iparam)  = choiceDataSession{igroup}.param_est{icontext,imodel}(:,iparam);
                    matub{igroup,icontext}(:,iparam)  = choiceDataSession{igroup}.param_est{icontext,imodel}(:,iparam);
                end
            end
        end
    end
end
ncontext = 2; %1 means that the data are fitted for both sessions simultaneously
[choiceDataSessionSwappedAlpha] = MoodTrackLearning_choice_getdata(matinx,matlb,matub,ncontext);

%% HAPPINESS
% happiness -simultaneous fit
%%% here, the initial values are randomised and both sessions are fitted
%%% simultaneously. This is providing initial values for the next fit,
%%% session per session
nmodel   = 13;
ncontext = 1;
matinx   = [];
for igroup = 1:2
    for imodel = 1:nmodel
        if ismember(imodel,[1:4 9])==1
            matinx{igroup,imodel} = nan(length(choiceDataAll{igroup}.gender),3);
        else
            matinx{igroup,imodel} = nan(length(choiceDataAll{igroup}.gender),4);
        end
    end
end
[happyDataAll] = MoodTrackLearning_happy_getdata(matinx,ncontext,choiceDataAll,nmodel,'r');
[ZhappyDataAll] = MoodTrackLearning_happy_getdata(matinx,ncontext,choiceDataAll,nmodel,'z');


% happiness -per session
matinx   = [];
ncontext = 2;
for igroup = 1:2
    for imodel=1:nmodel
        matinx{igroup,imodel} = happyDataAll{igroup}.param_est{1,imodel};
    end
end
[happyDataSession] = MoodTrackLearning_happy_getdata(matinx,ncontext,choiceDataSession,nmodel,'r');
[ZhappyDataSession] = MoodTrackLearning_happy_getdata(matinx,ncontext,choiceDataSession,nmodel,'z');
matinx   = [];
ncontext = 2;
for igroup = 1:2
    for imodel=1:nmodel
        matinx{igroup,imodel} = happyDataAll{igroup}.param_est{1,imodel};
    end
end
[happyDataSessionSwapped] = MoodTrackLearning_happy_getdata(matinx,ncontext,choiceDataSessionSwappedAlpha,nmodel,'r');
[ZhappyDataSessionSwapped] = MoodTrackLearning_happy_getdata(matinx,ncontext,choiceDataSessionSwappedAlpha,nmodel,'z');
%% do the statistics
% choice stats, per session
allStats = struct;
[allStats.choice] = MoodTrackLearning_choice_doStats(choiceDataSession);
% happiness stats, per session
nmodel=13;  
[allStats.happy] = MoodTrackLearning_happy_doStats(happyDataSession,choiceDataSession,nmodel);
[allStats.Zhappy] = MoodTrackLearning_happy_doStats(ZhappyDataSession,choiceDataSession,nmodel);
% happiness stats, per session
[allStats.happySwappedAlpha] = MoodTrackLearning_happy_doStats(happyDataSessionSwapped,choiceDataSession,nmodel);
[allStats.ZhappySwappedAlpha] = MoodTrackLearning_happy_doStats(ZhappyDataSessionSwapped,choiceDataSession,nmodel);
% correlate gammas and "swapped" gammas
for imodel=1:12
    ZparamallCorrectOrder{1,imodel}(:,:) = [ZhappyDataSession{1}.param_est{1,imodel};ZhappyDataSession{2}.param_est{2,imodel}];
    ZparamallCorrectOrder{2,imodel}(:,:) = [ZhappyDataSession{1}.param_est{2,imodel};ZhappyDataSession{2}.param_est{1,imodel}];
    paramallCorrectOrder{1,imodel}(:,:) = [happyDataSession{1}.param_est{1,imodel};happyDataSession{2}.param_est{2,imodel}];
    paramallCorrectOrder{2,imodel}(:,:) = [happyDataSession{1}.param_est{2,imodel};happyDataSession{2}.param_est{1,imodel}];
    ZparamallSwappedOrder{1,imodel}(:,:) = [ZhappyDataSessionSwapped{1}.param_est{1,imodel};ZhappyDataSessionSwapped{2}.param_est{2,imodel}];
    ZparamallSwappedOrder{2,imodel}(:,:) = [ZhappyDataSessionSwapped{1}.param_est{2,imodel};ZhappyDataSessionSwapped{2}.param_est{1,imodel}];
    paramallSwappedOrder{1,imodel}(:,:) = [happyDataSessionSwapped{1}.param_est{1,imodel};happyDataSessionSwapped{2}.param_est{2,imodel}];
    paramallSwappedOrder{2,imodel}(:,:) = [happyDataSessionSwapped{1}.param_est{2,imodel};happyDataSessionSwapped{2}.param_est{1,imodel}];
end
allStats.Zhappy.paramGammaVSswappedGammaCorr = [];
for icontext=1:2
    X =  ZparamallCorrectOrder{icontext,5}(:,1);%
    Y =  ZparamallSwappedOrder{icontext,5}(:,2);%
    [rho,p] = corr(X,Y,'type','Spearman')
    allStats.Zhappy.paramGammaVSswappedGammaCorr (icontext,1) = rho;
    allStats.Zhappy.paramGammaVSswappedGammaCorr (icontext,2) = p;
end
colNames = {'rho','p'};
rowNames = {'Stable','Volatile'};
allStats.Zhappy.paramGammaVSswappedGammaCorr  = array2table(allStats.Zhappy.paramGammaVSswappedGammaCorr ,'RowNames',rowNames,'VariableNames',colNames);
%% do the figures
% choice
MoodTrackLearning_choice_doPlots(choiceDataSession);
% happiness
MoodTrackLearning_happy_doPlots(happyDataSession);
MoodTrackLearning_happy_doPlots(ZhappyDataSession);
