function [loglike, probchoice,EU,list_loglike] = eval_loglike_additive(choice, rewardmag, weight, beta,gamma) 

%Inputs: 
%   CHOICE - a vector of choices with 1s for choices to option 1 (choices
%      to option 2 could be 0s or 2s or whatever)
%   REWARDMAG - reward magnitude vectors, corresponding on the potential
%   gain of each option
%   WEIGHT - a matrix with 2 columns of weights, all >=0, computed using
%      the standard Q-learning model; the first column is the weights for
%      option 1 and the second is the weights for option 2
%   BETA - a noise parameter, the inverse temperature
%   GAMMA - a risk attitude parameter placed on the probability
%Outputs:
%   LOGLIKE - the total (negative) loglikelihood of a model given the
%      player's choices, the Q-learning model weight matrix, and the
%      player's noise parameter
%   PROBCHOICE - a vector with the probability of choosing option 1 on each
%      trial according to the model
%   EU - a vector with the expected utility
%
%Bastien Blain (b.blain@ucl.ac.uk)
%February, 2019

choice = double(choice == 1);
M1     = rewardmag(:,1);
M2     = rewardmag(:,2);
P1     = max(min(weight(:,1),1),0);
P2     = max(min(weight(:,2),1),0);
deltaM = M1-M2;
deltaP = P1-P2;
EU1    = P1.*M1;%w*P1*(1-w)*(M1);
EU2    = P2.*M2;%w*P2*(1-w)*(M2);
EU     = [EU1,EU2];

outcome_value_diff = gamma.*deltaP + (1-gamma).*deltaM;
logodds            = beta * (outcome_value_diff); %compute log odds of choice for each trial
probchoice         = 1 ./ (1 + exp(-logodds));        %convert log odds to probability
probchoice(find(probchoice == 0)) = eps;      %to prevent fmin crashing from a log zero
probchoice(find(probchoice == 1)) = 1 - eps;
loglike = - (transpose(choice(:)) * log(probchoice(:)) + transpose(1-choice(:)) * log(1-probchoice(:)));
for i=1:length(choice);list_loglike(i) = -(choice(i).*log(probchoice(i))+(1-choice(i))*log(1-probchoice(i)));end
