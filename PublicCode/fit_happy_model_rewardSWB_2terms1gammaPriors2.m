function result_best = fit_happy_model_rewardSWB_2terms1gammaPriors2(term1_mtx,term2_mtx,happyscore,priors,constant,type)

%
% fit happiness model with constant term:
%    result = fit_happy_model_rewardSWB_2terms(term1_mtx,term2_mtx,happyscore,1)
% fit happiness model with z-scored ratings:
%    result = fit_happy_model_rewardSWB_2terms(term1_mtx,term2_mtx,zhappyscore)
%
% Bastien Blain, Janvier 2019
if sum(isnan(priors))>=1
    nstart = 40;
else
    nstart = 1;
end
for istart=1%:nstart
    result(istart).modelname = mfilename;
    result(istart).term1_mtx = term1_mtx;
    result(istart).term2_mtx  = term2_mtx;
    result(istart).happyscore  = happyscore;
    if (sum(isnan(priors))>=1) 
        if (istart==1)
            %inx = [2-4*rand 2-4*rand rand rand];
            %         rangeW1     = [0.1];
            %         rangeW2     = [0.1];
            %         rangeGamma = [0.1:0.1:1];
            %         rangeCst   = [0.5];
            %         inx = CombVec(rangeW1,rangeW2,rangeGamma,rangeCst)';
            %         rangeW1     = [.1.*ones(10,1)];%[-1:1:1];
            %         rangeW2     = [.1.*ones(10,1)];%[-1:1:1];
            %         rangeGamma = [0.1:0.1:1]';
            %         rangeCst   = 0.5.*ones(10,1);%[0.25:0.25:0.75];
            %         inx = [rangeW1,rangeW2,rangeGamma,rangeCst];%CombVec(rangeW1,rangeGamma,rangeCst)';
            rangeW1    = .1;%[-0.1 0.1];%[.1.*ones(10,1)];%[-1:1:1];
            rangeW2    = .1;%[-0.1 0.1];%[.1.*ones(10,1)];%[-1:1:1];
            rangeGamma = 0.5;%[0.1:0.1:1]';%;%rand(10,1);%[0.1:0.1:1]';
            rangeCst   = 0.5;%.*ones(10,1);%[0.25:0.25:0.75];
            inx = [rangeW1,rangeW2,rangeGamma,rangeCst];%CombVec(rangeW1,rangeW2,rangeGamma',rangeCst)'; CombVec(rangeW1,rangeW2,rangeGamma',rangeCst)';%
        end
    else
        inx = priors;%;
    end
  
    options = optimset('Display','notify','MaxIter',1000,'TolFun',1e-5,'TolX',1e-5,...
        'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
    warning off; %display,iter to see outputs
    %lb = [-5 -5 -5 0];
    %ub = [5 5 5 1]; %max tau of 1.1 or use 1.5

    if exist('constant'),
        if type=='r'
            lb = [-0.4 -0.4 0 ];
            ub = [ 1  1 1 ]; %max tau of 1.1 or use 1.5
            inx = [inx];
            lb = [lb  0];
            ub = [ub  1];
        elseif type == 'z'
            lb = [-0.8 -0.8 0 ];
            ub = [ 2  2     1 ]; %max tau of 1.1 or use 1.5
            inx = [inx];
            lb = [lb  -2];
            ub = [ub  2];
        end  
    else
        inx = [inx 0];
        lb = [lb 0];
        ub = [ub 0];
    end;
    
    result(istart).inx = inx(istart,:);
    result(istart).dof = length(inx(istart,:));
    result(istart).options = options;
    result(istart).lb = lb;
    result(istart).ub = ub;
    result(istart).b = zeros(1,length(inx(istart,:)));
    result(istart).se = zeros(1,length(inx(istart,:)));
    result(istart).r2 = 0;
    result(istart).happypred = 0;
    
    [b, ~, ~, ~, ~, ~, H] = fmincon(@model, inx(istart,:), [],[],[],[],lb,ub,[], options, result(istart));
    result(istart).blabel = {'Term 1','Term 2','tau1','tau2','const'};
    result(istart).b  = b([1:4]);
    result(istart).se = transpose(sqrt(diag(inv(H)))); %does not always work
    [sse, happypred, happyr2] = model(b, result(istart));
    result(istart).happypred = happypred;
    result(istart).r2 = happyr2;
    result(istart).sse = sse;
    result(istart).bic       = length(happypred)*log(sse/length(happypred)) + sum(lb~=ub)*log(length(happypred));
    result(istart).aic       = length(happypred)*log(sse/length(happypred)) + 2*sum(lb~=ub);
end
ibest = find([result(:).r2]==max([result(:).r2]));
result_best = result(ibest(1));

function [sse, happypred, happyr2] =  model(x, result)
a = x(1);b = x(2);tau1 = x(3);tau2 = x(3);const = x(end);
[happypred] = happy_model_2terms(a,b,tau1,tau2,const,result.term1_mtx,result.term2_mtx);
sse         = sum((result.happyscore-happypred).^2); %sum least-squares error
re          = sum((result.happyscore-mean(result.happyscore)).^2); 
happyr2     = 1-sse/re;
