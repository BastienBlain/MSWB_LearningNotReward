function [weight,rpe] = qlearning_1arm_model(choice, reward, alpha,x0)
%
%Inputs:
%   CHOICE - choice vector (1 for option 1, 0 or some other number for
%      option 2)
%   REWARD - reward vector (1 or a positive number for reward, 0 for no
%      reward)
%   ALPHA - the learning rate constant
%   X0 - the prior belief about the option
%
%Outputs: 
%   WEIGHT - a matrix with 2 columns of weights, all >=0, computed using
%      the standard Q-learning model; the first column is the weights for
%      option 1 and the second is the weights for option 2; weights are
%      initiated as X0s
%   RPE - a vector of RPE, resulting from a standard Rescorla-Wagner model
%Bastien Blain (b.blain@ucl.ac.uk)
%February 2019

choice = double(choice == 1);  %1 for option 1, 0 for option 2
ntrial = length(choice);
weight = zeros(ntrial,2);
% prior belief (initial weight, default is .5). 
weight(:,1)= ones(ntrial,1).*x0;
weight(:,2)= 1-weight(:,1);
rpe = zeros(size(choice)); 

for n = 1:(ntrial),
    rpe(n) = reward(n) - weight(n, 2-choice(n));% compute rpe
    if n<ntrial
        weight(n+1, 2-choice(n)) = weight(n, 2-choice(n)) + alpha*rpe(n); % update chosen
        weight(n+1, 1+choice(n)) = 1-weight(n+1, 2-choice(n)); % the unchosen is the complementary weight(n, 1+choice(n));
    end
end;
