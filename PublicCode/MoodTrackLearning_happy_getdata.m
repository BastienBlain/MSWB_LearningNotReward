function  [happyData] = MoodTrackLearning_happy_getdata(matinx,ncontext,choiceData,nmodel,happyType)
% Bastien Blain, bastien.blain@gmail.com
% February, 2019
cmodel = 1;
for igroup = 1:2
    % load the data
    load Blain_MoodTracksLearning_data
    if igroup == 1
        group_data = stable2volatile_data;
    else
        group_data = volatile2stable_data;
    end
    nsub           = size(group_data,2);
    % loop throught each individual participant from that group
    for isub = 1:nsub
        for icontext = 1:ncontext
            %% get the relevant trial and variables
            % get the trials corresponding to the context
            if ncontext==2
                if icontext     == 1
                    range_trials = 1:80;
                elseif icontext == 2
                    range_trials = 81:160;
                end
            elseif ncontext==1
                range_trials = 1:160;
            elseif ncontext==4
                if icontext     == 1
                    range_trials = 1:40;
                elseif icontext == 2
                    range_trials = 41:80;
                elseif icontext     == 3
                    range_trials = 81:120;
                elseif icontext == 4
                    range_trials = 121:160;
                end
                if mod(ncontext,2)==0
                    range_weight_idx = 41:80;
                else
                    range_weight_idx = 1:40;
                end
            end
            if ncontext > 1 & ncontext<4
                range_weight_idx = 1:80;
            elseif ncontext==1
                range_weight_idx = 1:160;
            end
            
            %% get raw & zscore happiness
            data                                         = group_data(isub).main;
            temp                                         = data.maintask(range_trials,:);
            happyData{igroup}.matmean{icontext}(isub) = nanmean(temp(:,8).*100); % mean happiness
            happyData{igroup}.matSD{icontext}(isub)   = nanstd(temp(:,8).*100);% SD happiness
            % get 1 X 80 rating vectors
            rawhappy{icontext}            = temp(:,8); %23 rating trials
            % get 1 X 23 ratings vector
            if ncontext==1
                happyind                      = data.happytrials_task;
                happyData{igroup}.mathappyalltrials{icontext}(isub,:) = nan(1,160);                    
                happyData{igroup}.mathappyalltrials{icontext}(isub,happyind) = rawhappy{icontext}(happyind);         
                matzhappy{icontext}(isub,:)   = nanzscore(temp(:,8));
                zhappy{icontext}              = (rawhappy{icontext}(happyind)-(happyData{igroup}.matmean{icontext}(isub))/100)/((happyData{igroup}.matSD{icontext}(isub))/100);
                happyData{igroup}.matZhappyalltrials{icontext}(isub,:) = matzhappy{icontext}(isub,:); 
            elseif ncontext==2
                happyind                      = data.happytrials_task(data.happytrials_task>80*(icontext-1)& data.happytrials_task<=(80+80*(icontext-1)))-(80*(icontext-1));
                happyData{igroup}.mathappyalltrials{icontext}(isub,:) = nan(1,80);           
%                 happyind                      = data.happytrials_task(data.happytrials_task>40*(icontext-1)& data.happytrials_task<=(40+40*(icontext-1)))-(40*(icontext-1));
%                 happyData{igroup}.mathappyalltrials{icontext}(isub,:) = nan(1,40);            
                happyData{igroup}.mathappyalltrials{icontext}(isub,happyind) = rawhappy{icontext}(happyind);         
                matzhappy{icontext}(isub,:)   = nanzscore(temp(:,8));
                zhappy{icontext}              = (rawhappy{icontext}(happyind)-(happyData{igroup}.matmean{icontext}(isub))/100)/((happyData{igroup}.matSD{icontext}(isub))/100);
                happyData{igroup}.matZhappyalltrials{icontext}(isub,:) = matzhappy{icontext}(isub,:); 
            else % zscoring for each condition divided into two parts (early and late trials)                   
                if icontext<=2
                    happyind_tmp                  = data.happytrials_task(data.happytrials_task>0& data.happytrials_task<=80);
                    matzhappy{icontext}(isub,:)   = nanzscore(data.maintask(1:80,8));
                    if icontext==1
                        happyind = happyind_tmp(happyind_tmp<=40);
                    else
                        happyind = happyind_tmp(happyind_tmp>40)-40;
                    end
                    zhappy{icontext}              = matzhappy{icontext}(isub,happyind+40*(1-mod(icontext,2)))';
                else
                    happyind_tmp                  = data.happytrials_task(data.happytrials_task>80& data.happytrials_task<=160)-80;
                    matzhappy{icontext}(isub,:)   = nanzscore(data.maintask(81:160,8));
                    if icontext==3
                        happyind = happyind_tmp(happyind_tmp<=40);
                    else
                        happyind = happyind_tmp(happyind_tmp>40)-40;
                    end
                    zhappy{icontext}              = matzhappy{icontext}(isub,happyind+40*(1-mod(icontext,2)))';
                end
                happyData{igroup}.mathappyalltrials{icontext}(isub,:) = nan(1,40); 
%                 icontext
%                 isub
%                 if icontext==2
%                     'ton pere'
%                 end
                happyData{igroup}.mathappyalltrials{icontext}(isub,happyind) = rawhappy{icontext}(happyind);  
            end
            
            %% get happiness after gain & after no gain
            happyData{igroup}.happy_gain(icontext,isub) = mean(temp(~isnan(temp(:,8)) & temp(:,17)>0,8)).*100;
            happyData{igroup}.happy_loss(icontext,isub) = mean(temp(~isnan(temp(:,8)) & temp(:,17)==0,8)).*100;
            happyData{igroup}.matRewardM(icontext,isub) = mean(temp(:,17));
                       
            %% Get the regressors
            % get the expectations and predition error terms
            for itrial = 1:size(temp,1)
                % convert left right into option1/option2
                temp(itrial,5)                 =  data.list_pair((data.maintask(range_trials(itrial),5)),range_trials(itrial));
                choice(itrial,1)               =  temp(itrial,5)-1;% 1 is option 2 chosen
                % objective expected value (reward is normalised
                chosen_proba(itrial)           = data.task_mu(temp(itrial,5),range_trials(itrial));
                chosen_EV(itrial)              = chosen_proba(itrial).*data.mat_mag(data.index_mag(range_trials(itrial)),temp(itrial,5))/80;
                % get the subjective probability from the choice model
                % (additive)
                % this piece of code is used when each context is split in
                % two parts
                if itrial==1 & range_trials(end)<=80 & ncontext>1
                    choicecontext = 1;
                elseif itrial==1 & range_trials(end)>80 & ncontext>1
                    choicecontext = 2;
                elseif itrial==1 & ncontext==1
                    choicecontext = icontext;
                end
                chosen_estproba_add(isub,itrial)    = [choiceData{igroup}.weight{choicecontext,cmodel}(isub,range_weight_idx(itrial),temp(itrial,5))];% get the probability of the chosen option temp(itrial,5)
                % get the sujective value   (reward is normalised)
                chosen_EVhat_add(isub,itrial)  = choiceData{igroup}.param_est{choicecontext,cmodel}(isub,3).* chosen_estproba_add(isub,itrial) + (1-choiceData{igroup}.param_est{choicecontext,cmodel}(isub,3)).*data.mat_mag(data.index_mag(range_trials(itrial)),temp(itrial,5))/80;
            end
            % define the regressor
            %%% expectations
            temp_chosenPhat       = (chosen_estproba_add(isub,:))'-0.5;%-mean(chosen_estproba_add(isub,:));%-0.5;
            temp_chosenEV         = (chosen_EV)' - mean(chosen_EV);
            temp_chosenEVhat_add  = (chosen_EVhat_add(isub,:))'  - mean(chosen_EVhat_add(isub,:));% centered additive SV      
            %%% outcomes
            temp_reward           = (temp(:,17));% received outcome magnitude
            temp_CNreward         = temp(:,17)-mean(temp(:,17));% received centered and normalised outcome magnitude
            temp_win              = temp(:,17)>0;
            temp_loss             = (temp(:,17)<=0).*-1;
            %%% prediction errors
            temp_rpe              = (temp(:,17)./80-chosen_EV');% objective RPE
            temp_ppe              = (((sign((temp(:,17)>0)-0.5))==1)-chosen_proba');
            temp_rpehat           = (temp(:,17)./80-chosen_EVhat_add(isub,:)');% multiplicative RPE hat
            temp_ppehat           = (choiceData{igroup}.rpe{choicecontext,cmodel}(isub,range_trials-80*(choicecontext-1)))';% additive PPE hat
            temp_ppehat_boost     = [[0; ~diff(max(sign(temp_ppehat),0))].*sign(temp_ppehat)]';
            
            % initialise the matrix
            %%% expectations
            chosenP_mtx           = zeros(length(happyind),size(temp,1));
            chosenPhat_mtx        = chosenP_mtx;
            chosenEV_mtx          = chosenP_mtx;
            chosenEVhat_add_mtx   = chosenP_mtx;     
            %%% outcomes
            reward_mtx            = chosenP_mtx; 
            CNreward_mtx          = chosenP_mtx;
            win_mtx               = chosenP_mtx;
            loss_mtx              = chosenP_mtx;
            %%% prediction errors
            rpe_mtx               = chosenP_mtx;
            ppe_mtx               = chosenP_mtx;
            rpehat_mtx            = chosenP_mtx;
            ppehat_mtx            = chosenP_mtx;
            temp_ppehat_boost_mtx = chosenP_mtx;
           
            % fill the matrix to regress
            for m=1:length(happyind),
                %%% expectations
                chosenPhat_mtx(m,1:length(1:happyind(m)))       = fliplr(transpose(temp_chosenPhat(1:happyind(m))));
                chosenEV_mtx(m,1:length(1:happyind(m)))         = fliplr(transpose(temp_chosenEV(1:happyind(m))));
                chosenEVhat_add_mtx(m,1:length(1:happyind(m)))  = fliplr(transpose(temp_chosenEVhat_add(1:happyind(m))));
                %%% outcomes
                reward_mtx(m,1:length(1:happyind(m)))           = fliplr(transpose(temp_reward(1:happyind(m))));
                CNreward_mtx(m,1:length(1:happyind(m)))         = fliplr(transpose(temp_CNreward(1:happyind(m))));
                win_mtx(m,1:length(1:happyind(m)))              = fliplr(transpose(temp_win(1:happyind(m))));
                loss_mtx(m,1:length(1:happyind(m)))             = fliplr(transpose(temp_loss(1:happyind(m))));
                %%% prediction errors
                rpe_mtx(m,1:length(1:happyind(m)))              = fliplr(transpose(temp_rpe(1:happyind(m))));
                ppe_mtx(m,1:length(1:happyind(m)))              = fliplr(transpose(temp_ppe(1:happyind(m))));                
                rpehat_mtx(m,1:length(1:happyind(m)))           = fliplr(transpose(temp_rpehat(1:happyind(m))));
                ppehat_mtx(m,1:length(1:happyind(m)))           = fliplr(transpose(temp_ppehat(1:happyind(m))));
                temp_ppehat_boost_mtx(m,1:length(1:happyind(m)))= fliplr(transpose(temp_ppehat_boost(1:happyind(m))));
            end
            
            % fill matrices for kernel analysis (suppl. fig. 3)
            pastrating = [];
            ztemp_ppe = zscore(temp_ppe);
            ztemp_ppehat = zscore(temp_ppehat);
            for ihr = 4:length(happyind)
                ltrial = (happyind(ihr)-10):happyind(ihr);
                pastppe(ihr-3,:)    = ztemp_ppe(ltrial);
                pastppehat(ihr-3,:) = ztemp_ppehat(ltrial);
                pastrating(ihr-3)   = happyData{igroup}.mathappyalltrials{icontext}(isub,happyind(ihr));
            end
             [bPPEhat,c,stats] = glmfit(fliplr(pastppehat),pastrating');
             [bPPE,c,stats] = glmfit(fliplr(pastppe),pastrating');           
             happyData{igroup}.kernel_est{icontext,1}(isub,:)= bPPEhat(2:end);            
             happyData{igroup}.kernel_est{icontext,2}(isub,:)= bPPE(2:end);
             
                 %% fitting the model using max LL
            if happyType=='z'%z-score
                happyrating = zhappy{icontext};
            elseif happyType=='dz'%detrend and zescore
                happyrating = detrend(zhappy{icontext});
            elseif happyType=='r'%raw
                happyrating = rawhappy{icontext}(happyind);
            end
            for imodel=1:nmodel               
                 priors = matinx{igroup,imodel}(isub,:);
                if imodel     == 1 % subjective probability prediction error
                    result{icontext,imodel} = fit_happy_model_rewardSWB_1termPriors2(ppehat_mtx,happyrating,priors,1,happyType);
                elseif imodel == 2 % semi subjective reward prediction error
                    result{icontext,imodel} = fit_happy_model_rewardSWB_1termPriors2(rpehat_mtx,happyrating,priors,1,happyType);
                elseif imodel == 3 % objective reward prediction error                    
                    result{icontext,imodel} = fit_happy_model_rewardSWB_1termPriors2(ppe_mtx,happyrating,priors,1,happyType);
                elseif imodel == 4 % objective prediction error                    
                    result{icontext,imodel} = fit_happy_model_rewardSWB_1termPriors2(rpe_mtx,happyrating,priors,1,happyType);                    
                elseif imodel == 5 % subjective probability & probability prediction error hat                    
                    result{icontext,imodel} = fit_happy_model_rewardSWB_2terms1gammaPriors2(chosenPhat_mtx,ppehat_mtx,happyrating,priors,1,happyType);
                elseif imodel == 6 % subjective probability & reward prediction error hat                    
                    result{icontext,imodel} = fit_happy_model_rewardSWB_2terms1gammaPriors2(chosenPhat_mtx,rpehat_mtx,happyrating,priors,1,happyType);
                elseif imodel == 7 % expected value hat additive & probability prediction error hat     
                    result{icontext,imodel} = fit_happy_model_rewardSWB_2terms1gammaPriors2(chosenEVhat_add_mtx,ppehat_mtx,happyrating,priors,1,happyType);                    
                elseif imodel == 8 % expected value hat additive & reward prediction error hat                    
                    result{icontext,imodel} = fit_happy_model_rewardSWB_2terms1gammaPriors2(chosenEVhat_add_mtx,rpehat_mtx,happyrating,priors,1,happyType);
                % extended model space: 
                elseif imodel == 9 % centered reward                    
                    result{icontext,imodel} = fit_happy_model_rewardSWB_1termPriors2(CNreward_mtx,happyrating,priors,1,happyType);  
                elseif imodel == 10% free centered reward                    
                    result{icontext,imodel} = fit_happy_model_rewardSWB_1termRefPriors2(reward_mtx,happyrating,priors,1,happyType);
                elseif imodel == 11
                     result{icontext,imodel} = fit_happy_model_rewardSWB_2terms1gammaPriors2(win_mtx,loss_mtx,happyrating,priors,1,happyType);   
                elseif imodel == 12 % interaction model 1: ceiling effect              
                    priors = [0.1 0.01 0.5 0.5];%[0.1 0.01 1 0.5 0.5];
                    result{icontext,imodel} = fit_happy_model_rewardSWB_2termPriors2_LinearInteractionHappy(ppehat_mtx,happyrating,priors,1,happyType);%fit_happy_model_rewardSWB_2termPriors2_interactionHappy(ppehat_mtx,happyrating,priors,1,happyType);
                elseif imodel == 13  %                  
                     priors = [0.1 0 0.5 0];
                    result{icontext,imodel} = fit_happy_model_rewardSWB_2terms1gammaPriors2(ppehat_mtx,temp_ppehat_boost_mtx,happyrating,priors,1,happyType);
                end
                
                %% storing parameter estimates, model performance & residuals
                happyData{igroup}.param_est{icontext,imodel}(isub,:)      = result{icontext,imodel}.b;
                happyData{igroup}.matr2{icontext}(isub,imodel)            = result{icontext,imodel}.r2;
                happyData{igroup}.dof{icontext}(isub,imodel)              = result{icontext,imodel}.dof;
                happyData{igroup}.matBIC{icontext}(isub,imodel)           = result{icontext,imodel}.bic;
                happyData{igroup}.matAIC{icontext}(isub,imodel)           = result{icontext,imodel}.aic;
                if ncontext==2
                happyData{igroup}.mathappyres{icontext,imodel}(isub,:)    = result{icontext,imodel}.happyscore - result{icontext,imodel}.happypred;
                happyData{igroup}.mathappyresall{icontext,imodel}(isub,:) = nan(1,80);
                happyData{igroup}.mathappyresall{icontext,imodel}(isub,happyind) = result{icontext,imodel}.happyscore - result{icontext,imodel}.happypred;
                % store expectations and PPE to check the residuals of the
                % outcome model
                happyData{igroup}.matExpectation{icontext}    = chosen_estproba_add;
                
                % backfill the residuals
                happyData{igroup}.happyres2backfill{icontext,imodel}(isub,:) = nan(1,80);
                happyData{igroup}.happyres2backfill{icontext,imodel}(isub,happyind) =happyData{igroup}. mathappyres{icontext,imodel}(isub,:);
                happyData{igroup}.matbackfilled_happyres{icontext,imodel}(isub,:) = backfilling(happyData{igroup}.happyres2backfill{icontext,imodel}(isub,:));
                happyData{igroup}.matpredhappy{icontext,imodel}(isub,:)   = result{icontext,imodel}.happypred;                
                happyData{igroup}.matpredhappyalltrials{icontext,imodel}(isub,:) = nan(1,80);
                happyData{igroup}.matpredhappyalltrials{icontext,imodel}(isub,happyind)   = result{icontext,imodel}.happypred;
                end
                
                %% interpolating
                if imodel==5
                    ntrial = length(range_trials);
                    itpl_chosenPhat_mtx = zeros(ntrial,ntrial);
                    itpl_ppehat_mtx     = zeros(ntrial,ntrial);
                    for m=1:ntrial,
                        itpl_chosenPhat_mtx(m,1:m) = fliplr(transpose(temp_chosenPhat(1:m)));
                        itpl_ppehat_mtx(m,1:m)     = fliplr(transpose(temp_ppehat(1:m)));
                    end                    
                    happyData{igroup}.itplhappy{icontext,imodel}(isub,:) =  happy_model_2terms(result{icontext,imodel}.b(1),result{icontext,imodel}.b(2),result{icontext,imodel}.b(3),result{icontext,imodel}.b(3),result{icontext,imodel}.b(4),itpl_chosenPhat_mtx,itpl_ppehat_mtx);
                end
            end
        end
    end
   happyData{igroup}.scorePHQ = choiceData{igroup}.scorePHQ;
   happyData{igroup}.scoreSTAI= choiceData{igroup}.scoreSTAI;
end
