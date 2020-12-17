clear all
close all
%%% CHOICE
matinx{1,1} = [0.5  1    0.5  0.5
               0.5    1    1    0.5
               0.5  1    1    0.5
               1    1    0    0.5];
matlb{1,1} = [0.01     0    0    0.5%additive
              0.01     0    .7   0.5%multiplicative
              0.01     0    1    0.5%additive, probability only
              1     0    0    0.5];%additive, magnitude only
matub{1,1} = [0.95    50    1    0.5
              0.95     50    1.3  0.5
              0.95     50    1    0.5
              1    50    0    0.5];
for igroup=1:2
    for icontext=1:2
        matinx{igroup,icontext} = matinx{1,1};
        matlb{igroup,icontext}  = matlb{1,1};
        matub{igroup,icontext}  = matub{1,1};
    end
end
% choice -per session
ncontext = 2; %1 means that the data are fitted for both sessions simultaneously
[choiceData] = MoodTrackLearning_choice_getdata(matinx,matlb,matub,ncontext);

cmodel = 1;
ncontext=2
for igroup = 1:2
    % load the data
    load Blain_MoodTracksLearning_data
    if igroup == 1
        group_data = stable2volatile_data;
        nsubG1 = size(group_data,2);
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
            end
            range_weight_idx = 1:80;
            data                                         = group_data(isub).main;
            temp                                         = data.maintask(range_trials,:);
            
            rawhappy{icontext}            = temp(:,8); %23 rating trials
            happyind                      = data.happytrials_task(data.happytrials_task>80*(icontext-1)& data.happytrials_task<=(80+80*(icontext-1)))-(80*(icontext-1));
          
                
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
                % (additive and multiplicative, respectively)
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
                chosen_estproba_mult(isub,itrial)   = [choiceData{igroup}.weight{choicecontext,2}(isub,range_weight_idx(itrial),temp(itrial,5))];
                % get the sujective value   (reward is normalised)
                chosen_EVhat_add(isub,itrial)  = choiceData{igroup}.param_est{choicecontext,cmodel}(isub,3).* chosen_estproba_add(isub,itrial) + (1-choiceData{igroup}.param_est{choicecontext,cmodel}(isub,3)).*data.mat_mag(data.index_mag(range_trials(itrial)),temp(itrial,5))/80;;
                chosen_EVhat_mult(isub,itrial) = chosen_estproba_mult(isub,itrial).*data.mat_mag(data.index_mag(range_trials(itrial)),temp(itrial,5))/80;
                % get the counterfactual
                counterfactual(isub,itrial)    = double((data.mat_mag(data.index_mag(range_trials(itrial)),3-temp(itrial,5))/80)*temp(itrial,17)==0);
            end
            % define the regressor
            %%% expectations
            temp_chosenP          = (chosen_proba)'-0.5;
            temp_chosenPhat       = (chosen_estproba_add(isub,:))'-0.5;%-mean(chosen_estproba_add(isub,:));%-0.5;
            temp_chosenPhat_mult  = (chosen_estproba_mult(isub,:))'-0.5;
            temp_chosenEV         = (chosen_EV)' - mean(chosen_EV);
            temp_chosenEVhat_add  = (chosen_EVhat_add(isub,:))'  - mean(chosen_EVhat_add(isub,:));% centered additive SV
            temp_chosenEVhat_mult = (chosen_EVhat_mult(isub,:))' - mean(chosen_EVhat_mult(isub,:));% centered multiplicative SV
            %%% outcomes
            temp_reward           = (temp(:,17));% received outcome magnitude
            %             temp_Creward          = (temp(:,17)-35);% received centered and normalised outcome magnitude
            %             temp_Nreward          = (temp(:,17))./80;% received outcome magnitude
            temp_CNreward         = temp(:,17)-mean(temp(:,17));% received centered and normalised outcome magnitude
            temp_signedoutcome    = (sign((temp(:,17)>0)-0.5));% binary outcome, 1 is winnin, -1 is not winning
            temp_counterfactual   = counterfactual(isub,:)';
            temp_win              = temp(:,17)>0;
            temp_loss             = (temp(:,17)<=0).*-1;
            %%% prediction errors
            temp_rpe              = (temp(:,17)./80-chosen_EV');% objective RPE
            temp_ppe              = (((sign((temp(:,17)>0)-0.5))==1)-chosen_proba');
            temp_rpehat           = (temp(:,17)./80-chosen_EVhat_mult(isub,:)');% multiplicative RPE hat
            temp_ppehat           = (choiceData{igroup}.rpe{choicecontext,cmodel}(isub,range_trials-80*(choicecontext-1)))';% additive PPE hat
            temp_ppehatP          = temp_ppehat.*(temp_ppehat>=0);
            temp_ppehatN          = temp_ppehat.*(temp_ppehat<0);
            temp_ppehat_boost     = [[0; ~diff(max(sign(temp_ppehat),0))].*sign(temp_ppehat)]';
            temp_ppehat_boost_pos = temp_ppehat_boost.*double(temp_ppehat_boost>0);
            temp_ppehat_boost_neg = temp_ppehat_boost.*double(temp_ppehat_boost<0);
            temp_ppehat_inter     = [temp_ppehat(2:end); 0]'.*[temp_ppehat]';
            temp_ppehat_mult      = (choiceData{igroup}.rpe{choicecontext,2}(isub,range_trials-80*(choicecontext-1)))';% mult PPE hat
            
            % initialise the matrix
            %%% expectations
            chosenP_mtx           = zeros(length(happyind),size(temp,1));
            chosenPhat_mtx        = chosenP_mtx;
            chosenPhat_mult_mtx   = chosenP_mtx;
            chosenEV_mtx          = chosenP_mtx;
            chosenEVhat_add_mtx   = chosenP_mtx;
            chosenEVhat_mult_mtx  = chosenP_mtx;
            %%% outcomes
            reward_mtx            = chosenP_mtx;
            CNreward_mtx          = chosenP_mtx;
            signedoutcome_mtx     = chosenP_mtx;
            counterfactual_mtx    = chosenP_mtx;
            win_mtx               = chosenP_mtx;
            loss_mtx              = chosenP_mtx;
            %%% prediction errors
            rpe_mtx               = chosenP_mtx;
            ppe_mtx               = chosenP_mtx;
            rpehat_mtx            = chosenP_mtx;
            ppehat_mtx            = chosenP_mtx;
            ppehatP_mtx           = chosenP_mtx;
            ppehatN_mtx           = chosenP_mtx;
            ppehat_mult_mtx       = chosenP_mtx;
            temp_ppehat_boost_mtx = chosenP_mtx;
            
            % fill the matrix to regress
            for m=1:length(happyind),
                %%% expectations
                chosenP_mtx(m,1:length(1:happyind(m)))          = fliplr(transpose(temp_chosenP(1:happyind(m))));
                chosenPhat_mtx(m,1:length(1:happyind(m)))       = fliplr(transpose(temp_chosenPhat(1:happyind(m))));
                chosenPhat_mult_mtx(m,1:length(1:happyind(m)))  = fliplr(transpose(temp_chosenPhat_mult(1:happyind(m))));
                chosenEV_mtx(m,1:length(1:happyind(m)))         = fliplr(transpose(temp_chosenEV(1:happyind(m))));
                chosenEVhat_add_mtx(m,1:length(1:happyind(m)))  = fliplr(transpose(temp_chosenEVhat_add(1:happyind(m))));
                chosenEVhat_mult_mtx(m,1:length(1:happyind(m))) = fliplr(transpose(temp_chosenEVhat_mult(1:happyind(m))));
                %%% outcomes
                reward_mtx(m,1:length(1:happyind(m)))           = fliplr(transpose(temp_reward(1:happyind(m))));
                CNreward_mtx(m,1:length(1:happyind(m)))         = fliplr(transpose(temp_CNreward(1:happyind(m))));
                signedoutcome_mtx(m,1:length(1:happyind(m)))    = fliplr(transpose(temp_signedoutcome(1:happyind(m))));
                counterfactual_mtx(m,1:length(1:happyind(m)))   = fliplr(transpose(temp_counterfactual(1:happyind(m))));
                win_mtx(m,1:length(1:happyind(m)))              = fliplr(transpose(temp_win(1:happyind(m))));
                loss_mtx(m,1:length(1:happyind(m)))             = fliplr(transpose(temp_loss(1:happyind(m))));
                %%% prediction errors
                rpe_mtx(m,1:length(1:happyind(m)))              = fliplr(transpose(temp_rpe(1:happyind(m))));
                ppe_mtx(m,1:length(1:happyind(m)))              = fliplr(transpose(temp_ppe(1:happyind(m))));
                rpehat_mtx(m,1:length(1:happyind(m)))           = fliplr(transpose(temp_rpehat(1:happyind(m))));
                ppehat_mtx(m,1:length(1:happyind(m)))           = fliplr(transpose(temp_ppehat(1:happyind(m))));
                ppehatP_mtx(m,1:length(1:happyind(m)))          = fliplr(transpose(temp_ppehatP(1:happyind(m))));
                ppehatN_mtx(m,1:length(1:happyind(m)))          = fliplr(transpose(temp_ppehatN(1:happyind(m))));
                ppehat_mult_mtx(m,1:length(1:happyind(m)))      = fliplr(transpose(temp_ppehat_mult(1:happyind(m))));
                temp_ppehat_boost_mtx(m,1:length(1:happyind(m)))= fliplr(transpose(temp_ppehat_boost(1:happyind(m))));
            end
            %% generate happiness ratings
            la = [0.7:-0.05:0.4];lb = [-80:20:80]./100;lg = 0.5;
            for ia=1:length(la)
                for ib = 1:length(lb)
                    for ig = 1
                        x(1) = la(ia);
                        x(2) = lg(ig);
                        x(3) = lb(ib);
                        x(4) = 0.5;
                        a = x(1);b = x(3); tau1 = x(2); const = x(end);
                        [happypred] = happy_model_1termRef(a,b,tau1,const, reward_mtx./100);
                        happypred = min(max(happypred,0),1);
                        
                        %% fit the ratings
                        priors = [0.1 0.5 0 0.5];
                        happyrating = happypred;%rawhappy{icontext}(happyind);
                        happyType = 'r';
                        result{icontext,ia,ib,ig} = fit_happy_model_rewardSWB_1termRefPriors2(reward_mtx./100,happyrating,priors,1,happyType);
                        if igroup ==1
                            matParam{icontext}(isub,ia,ib,ig,:) =  result{icontext,ia,ib,ig}.b;
                        else
                            matParam{icontext}(isub+nsubG1,ia,ib,ig,:) =  result{icontext,ia,ib,ig}.b;
                        end
                    end
                end
            end
        end
    end
end


for ib = 1:length(lb)
    for isub=1:75
        [r, p] = corr(la',squeeze(matParam{icontext}(isub,:,ib,1,1))');
        matCorrIA{icontext}(ia,ib) = r;
        matPIA{icontext}(ia,ib) = p;
    end
end

for ia = 1:length(la)
    for isub=1:75
        [r, p] = corr(lb',squeeze(matParam{icontext}(isub,ia,:,1,1)));
        matCorrIB{icontext}(ia,ib) = r;
        matPIB{icontext}(ia,ib) = p;
    end
end




figure;
subplot(2,1,1);hold on;
title('Stable')
[Y, X] = meshgrid(la,lb);
Z =  squeeze(mean(matParam{1}(:,:,:,ig,3),1))';
Z_expected = repmat(lb',1,length(la));
surf(X,Y,Z_expected,'FaceColor',[1 0 0],'FaceAlpha',0.2,'EdgeColor','none')
surf(X,Y,Z)
xlabel('RP generated')
ylabel('W_R generated')
zlabel('RP estimated')

subplot(2,1,2);hold on;
title('Volatile')
[Y, X] = meshgrid(la,lb);
Z =  squeeze(mean(matParam{2}(:,:,:,ig,3),1))';
Z_expected = repmat(lb',1,length(la));
surf(X,Y,Z_expected,'FaceColor',[1 0 0],'FaceAlpha',0.2,'EdgeColor','none')
surf(X,Y,Z)
xlabel('RP generated')
ylabel('W_R generated')
zlabel('RP estimated')

mean(squeeze(mean(matParam{1}(:,:,:,ig,3),1)),1)
mean(squeeze(mean(matParam{1}(:,:,:,ig,4),1)),1)
figure;hold on;
scatter(mean(squeeze(mean(matParam{1}(:,:,:,ig,3),1)),1),mean(squeeze(mean(matParam{1}(:,:,:,ig,4),1)),1),90,'MarkerFaceColor',[0.5 0.5 0.5])
xlabel('RP')
ylabel('Constant')
set(gca,'FontSize',20)

figure;
subplot(2,1,1);hold on;
title('Stable')
[Y, X] = meshgrid(lb,la);
Z =  squeeze(mean(matParam{1}(:,:,:,ig,1),1));
Z_expected = repmat(la',1,length(lb));
surf(X,Y,Z_expected,'FaceColor',[1 0 0],'FaceAlpha',0.2,'EdgeColor','none')
surf(X,Y,Z)
xlabel('W_{win} generated')
ylabel('W_{loss} generated')
zlabel('W_{win} estimated')

subplot(2,1,2);hold on;
title('Volatile')
[Y, X] = meshgrid(lb,la);
Z =  squeeze(mean(matParam{2}(:,:,:,ig,1),1));
Z_expected = repmat(la',1,length(lb));
surf(X,Y,Z_expected,'FaceColor',[1 0 0],'FaceAlpha',0.2,'EdgeColor','none')
surf(X,Y,Z)
xlabel('W_{win} generated')
ylabel('W_{loss} generated')
zlabel('W_{win} estimated')


