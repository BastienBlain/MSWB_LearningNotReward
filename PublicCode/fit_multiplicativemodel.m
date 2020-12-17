function [result] = fit_multiplicativemodel(inx,lb,ub, choice, reward, rewardmag)

%[result] = fit_qlearning(inx, choice, reward)
%
%Inputs:
%   INX - starting values for alpha and beta
%   CHOICE - choice vector (1 for option 1, 0 or some other number for
%      option 2)
%   REWARD - reward vector (1 or a positive number for reward, 0 for no
%      reward)
%   REWARDMAG - reward magnitude vectors, corresponding on the potential
%   gain of each option
%   INDEXBEST - vector indicated when participants chose the best options,
%   (for Farashahi's model)
%
%Outputs:
%   RESULT - a struct of the data and all information return by the model
%      fitting procedure
%
%Requires:
%   qlearning_1arm_model, eval_loglike_behrens
%
%Bastien Blain (b.blain@ucl.ac.uk)
%February, 2019

options = optimset('Display','off','MaxIter',10000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-6,'MaxFunEvals',500,'LargeScale','off');
% warning off;
result.choice = choice;
result.reward = reward;
result.rewardmag = rewardmag;
result.inx = inx;
result.lb = lb;
result.ub = ub;
result.options = options;

try,
    inx0 = inx;
    [b, loglike, exitflag, output, lambda, grad, H] = fmincon(@model, inx0, [],[],[],[],lb,ub,[], options, choice, reward, rewardmag);
    se = transpose(sqrt(diag(inv(H))));
    result.b       = b;
    result.alpha   = b(1);
    result.alphase = se(1);
    result.beta    = b(2);
    result.betase  = se(2);
    result.gamma   = b(3);
    result.gammase = se(3);
    result.x0      = b(4);
    result.x0se    = se(4);

    result.modelLL     = -loglike;
    result.nullmodelLL = log(0.5)*size(choice,1);         %LL of random-choice model
    result.pseudoR2    = 1 + loglike / (result.nullmodelLL); %pseudo-R2 statistic
    result.exitflag    = exitflag;
    result.output      = output;
    result.H           = H; %Hessian
    [loglike, probchoice, weight,rpe,EU] = model(b, choice, reward, rewardmag); %best fitting model
    result.probchoice  = probchoice; %prob of choosing option 1 on each trial
    result.weight      = weight;         %model fits Q-values for each trial
    result.rpe         = rpe;
    result.EU          = EU;
catch,
    lasterr
    result.modelLL  = 0;
    result.exitflag = 0;
end;


function [loglike, probchoice, weight,rpe,EU] = model(x, choice, reward, rewardmag)
%function to evaluate the loglikelihood of the model for parameters alpha
%gamma and beta given the data
alpha = x(1);
beta  = x(2);
gamma = x(3);
x0    = x(4);

[weight,rpe] = qlearning_1arm_model(choice, reward, alpha,x0); %compute the weights
[loglike, probchoice,EU] = eval_loglike_multiplicative(choice, rewardmag, weight, beta, gamma); %compute the likelihood