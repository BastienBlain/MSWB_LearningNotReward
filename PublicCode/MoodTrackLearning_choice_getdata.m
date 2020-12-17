function  [choiceData] = MoodTrackLearning_choice_getdata(matinx,matlb,matub,ncontext)
% Bastien Blain, bastien.blain@gmail.com
% February, 2019

% set the number of model to compare
nmodel        = 5;
for igroup=1:2
    % load the data
    load Blain_MoodTracksLearning_data % PUT LABELS ON THE TOP
    if igroup==1
        group_data = stable2volatile_data;
    else
        group_data = volatile2stable_data;
    end
    nsub           = size(group_data,2);
    
    % loop throught each individual participant from that group
    for isub = 1:nsub
        %% store mood and demographics
        choiceData{igroup}.scoreBDI(isub)  = group_data(isub).BDI.bdi_score;
        choiceData{igroup}.scoreSTAI(isub) = group_data(isub).STAI.STAI_score;
        choiceData{igroup}.scorePHQ(isub)  = sum(group_data(isub).PHQ.PHQ_raw-1);
        choiceData{igroup}.age(isub)       = group_data(isub).age;
        choiceData{igroup}.gender(isub)    = group_data(isub).gender;
        
        data            = group_data(isub).main;
        % loop through both contexts
        for icontext=1:ncontext
            %% get the relevant trial and variables
            % get the trials corresponding to the context
            if ncontext==2
                if icontext==1
                    range_trials = 1:80;
                elseif icontext==2
                    range_trials = 81:160;
                end
            else
                range_trials = 1:160;
            end
            % get choice, probability, reward and outcome
            
            proba = data.task_mu(:,range_trials)';
            outcome{isub,icontext} = double(data.maintask(range_trials,15)>0);
            for itrial=1:length(range_trials)% choice == 1 means option 1 chosen, choice ==0 means option 2 chosen
                %choice_made(itrial)   = 3-data.list_pair((data.maintask(range_trials(itrial),5)),range_trials(itrial));
                choice{isub,icontext}(itrial,1) = 3-data.list_pair((data.maintask(range_trials(itrial),5)),range_trials(itrial))-1;
                rewardmag(itrial,1)   = data.mat_mag(data.index_mag(range_trials(itrial)),1)/80;%
                rewardmag(itrial,2)   = data.mat_mag(data.index_mag(range_trials(itrial)),2)/80;%
                rewardproba(itrial,1) = data.task_mu(data.list_pair(1,range_trials(itrial)),range_trials(itrial));
                rewardproba(itrial,2) = data.task_mu(data.list_pair(2,range_trials(itrial)),range_trials(itrial));
            end
                     
            %% estimate the choice models
            for imodel=1:nmodel
                if imodel<5
                inx = matinx{igroup,icontext}(imodel,:);
                lb  = matlb{igroup,icontext}(imodel,:);
                ub  = matub{igroup,icontext}(imodel,:);
                else
                 inx = matinx{igroup,icontext}(2,:);
                lb  = matlb{igroup,icontext}(2,:);
                ub  = matub{igroup,icontext}(2,:);   
                end
                if imodel==2
                    for istart=1                       
                        % fit the data
                        fitoutput_tmp{istart} = fit_multiplicativemodel(inx,lb,ub, choice{isub,icontext}, outcome{isub,icontext},rewardmag);
                    end
                    % get the best set of initial values (if different intial
                    % values are used)
                    LL(istart) =  fitoutput_tmp{istart}.modelLL;
                    indexbest    = find(LL==max(LL));
                elseif (imodel==1 | imodel>2) & imodel < 5
                    % set the constraints
                    for istart=1                      
                        % fit the data
                        fitoutput_tmp{istart}  = fit_additivemodel(inx,lb,ub, choice{isub,icontext}, outcome{isub,icontext},rewardmag);
                    end
                    % get the best set of initial values (if different intial
                    % values are used)
                    LL(istart) = fitoutput_tmp{istart}.modelLL;
                    indexbest    = find(LL==max(LL));
                elseif imodel==5
                    % set the constraints
                    for istart=1                      
                        % fit the data
                        inx = [inx(1) inx(1) inx(2:end)];
                        lb  = [lb(1) lb(1) lb(2:end)];
                        ub  = [ub(1) ub(1) ub(2:end)];
                        fitoutput_tmp{istart}  = fit_additivemodelWL(inx,lb,ub, choice{isub,icontext}, outcome{isub,icontext},rewardmag);
                    end
                    % get the best set of initial values (if different intial
                    % values are used)
                    LL(istart) = fitoutput_tmp{istart}.modelLL;
                    indexbest    = find(LL==max(LL));
                end
                % store the fit output and other variable of interest
                choiceData{igroup}.fitoutput{isub,icontext,imodel}     = fitoutput_tmp{indexbest(1)};
                choiceData{igroup}.param_est{icontext,imodel}(isub,:)  = choiceData{igroup}.fitoutput{isub,icontext,imodel}.b;% (for supplementary table 1)
                choiceData{igroup}.probchoice{icontext,imodel}(isub,:) = (choiceData{igroup}.fitoutput{isub,icontext,imodel}.probchoice)';
                choiceData{igroup}.rpe{icontext,imodel}(isub,:)        = choiceData{igroup}.fitoutput{isub,icontext,imodel}.rpe;
                choiceData{igroup}.weight{icontext,imodel}(isub,:,:)   = choiceData{igroup}.fitoutput{isub,icontext,imodel}.weight;
                
                % store model performance (for table 1)
                choiceData{igroup}.matLL{icontext}(isub,imodel)   = choiceData{igroup}.fitoutput{isub,icontext,imodel}.modelLL; % log likelihood
                choiceData{igroup}.dof(imodel)                    = length(inx) - sum(ub==lb);
                choiceData{igroup}.matBIC{icontext}(isub,imodel)  = log(80)*choiceData{igroup}.dof(imodel)-2*(choiceData{igroup}.fitoutput{isub,icontext,imodel}.modelLL);
                choiceData{igroup}.matAIC{icontext}(isub,imodel)  = 2*choiceData{igroup}.dof(imodel)-2*(choiceData{igroup}.fitoutput{isub,icontext,imodel}.modelLL);
                choiceData{igroup}.matr2{icontext}(isub,imodel)   = choiceData{igroup}.fitoutput{isub,icontext,imodel}.pseudoR2;
                TP{icontext}(isub,imodel)      = nansum(choice{isub,icontext} & choiceData{igroup}.fitoutput{isub,icontext,imodel}.probchoice>0.5);
                P{icontext}(isub,imodel)       = nansum(choice{isub,icontext});
                TN{icontext}(isub,imodel)      = nansum(choice{isub,icontext}==0 & choiceData{igroup}.fitoutput{isub,icontext,imodel}.probchoice<=0.5);
                N{icontext}(isub,imodel)       = nansum(choice{isub,icontext}==0);
                choiceData{igroup}.matBacc{icontext}(isub,imodel) = (TP{icontext}(isub,imodel)/P{icontext}(isub,imodel) + TN{icontext}(isub,imodel)/N{icontext}(isub,imodel))/2;% balanced accuracy
            end
            
            %% compute different metrics for choice accuracy (for figure 2A)
            % in terms on ev
            EV = [rewardmag(:,1).*proba(:,1) rewardmag(:,2).*proba(:,2)];
            bestEV = EV(:,1)>EV(:,2);
            choiceData{igroup}.matperfEV(isub,icontext) = mean(choice{isub,icontext}==bestEV);
            % in terms of probability
            bestProba = proba(:,1)>proba(:,2);
            choiceData{igroup}.matperfProba(isub,icontext) = mean(choice{isub,icontext}==bestProba);
            % in terms of EV for incogruent choices
            index_inc = bestEV~=bestProba;
            choiceData{igroup}.maxEVinc(isub,icontext) = mean(choice{isub,icontext}(index_inc)==bestEV(index_inc));
            choiceData{igroup}.maxProbainc(isub,icontext) = mean(choice{isub,icontext}(index_inc)==bestProba(index_inc));
            index_incext = (proba(:,1)<proba(:,2) & (data.mat_mag(data.index_mag(range_trials),2)-data.mat_mag(data.index_mag(range_trials),1))<=-50)...
                | (proba(:,2)<proba(:,1) & (data.mat_mag(data.index_mag(range_trials),1)-data.mat_mag(data.index_mag(range_trials),2))<=-50);
            choiceData{igroup}.maxEVincext(isub,icontext) = mean(choice{isub,icontext}(index_incext)==bestEV(index_incext));
            % time series
            %%% behaviour
            choiceData{igroup}.bestEVChoice{icontext}(isub,:)=choice{isub,icontext}==bestEV;
            % store model prediction performance in termps of EV maximisation
            for imodel=1:4
                choiceData{igroup}.bestEVPred{icontext,imodel}(isub,bestEV==1) = choiceData{igroup}.probchoice{icontext,imodel}(isub,bestEV==1);
                choiceData{igroup}.bestEVPred{icontext,imodel}(isub,bestEV==0) = 1-choiceData{igroup}.probchoice{icontext,imodel}(isub,bestEV==0);
            end
            
            %% compute the proportion on win/stay lose/stay for the best and worst option (for figure 2B)
            for imodel=1:4
                % convert everything in terms of the best choice
                best_option = [proba(:,1)>proba(:,2)];% objective highest probabiltity of winning option; proba is encoded in terms of stim  
                matprobchoiceall_ctx = choiceData{igroup}.probchoice{icontext,imodel}(isub,:);% get model prediction (in terms of stim)
                choice_made_ctx      = choice{isub,icontext};% get actual choice (in terms of stim)
                choice_bestframe = choice_made_ctx;
                choice_bestframe(best_option==0) = 1-choice_made_ctx(best_option==0);% convert choice in terms of best option
                proba_bestframe = matprobchoiceall_ctx;
                proba_bestframe(best_option==0) = 1-matprobchoiceall_ctx(best_option==0);%convert prediction in terms of best option
                
                % store win/stay and model predictions
                matdata = [choice_bestframe double(data.maintask(range_trials,17)>0) proba_bestframe'];% create a matrix with choices, outcomes, and predictions
                matdata = [matdata [2*(1-matdata(:,1)) + (1-matdata(:,2)) + 1]]; %column 4 maps to 4 categories
                sumchoice = zeros(1,4); sumprob = zeros(1,4); nsample = zeros(1,4);
                for n = 1:size(matdata,1)-1,
                    sumchoice(matdata(n,4))=sumchoice(matdata(n,4))+matdata(n+1,1);
                    sumprob(matdata(n,4))=sumprob(matdata(n,4))+matdata(n+1,3);
                    nsample(matdata(n,4))=nsample(matdata(n,4))+1;
                end;
                
                % convert to winstay/losestay and probabilistic equivalent (mean probability vs mean proportion)
                choiceData{igroup}.meanchoice{icontext}(isub,:) = sumchoice./nsample;  
                choiceData{igroup}.meanchoice{icontext}(isub,[3 4]) = 1 -  choiceData{igroup}.meanchoice{icontext}(isub,[3 4]);
                choiceData{igroup}.meanprob{icontext,imodel}(isub,:)= sumprob./nsample; 
                choiceData{igroup}.meanprob{icontext,imodel}(isub,[3 4]) = 1 -  choiceData{igroup}.meanprob{icontext,imodel}(isub,[3 4]);
            end
        end
    end
end