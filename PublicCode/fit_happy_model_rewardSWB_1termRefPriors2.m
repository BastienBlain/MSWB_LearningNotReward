function result_best = fit_happy_model_rewardSWB_1termRefPriors2(term1_mtx,happyscore,priors,constant,type)

% result = fit_happy_model_rewardSWB_1term(term1_nmtx,happyscore,constant)
%
% fit happiness model with constant term:
%    result = fit_happy_model_rewardSWB(term1_nmtx,happyscore,1)
% fit happiness model with z-scored ratings:
%    result = fit_happy_model_rewardSWB(term1_nmtx,zhappyscorey)
%
% Bastien Blain, January 2019
if sum(isnan(priors))>=1
    nstart = 40;
else
    nstart = 1;
end
lW2 = [-1,0,1];
for istart=1:3
    result(istart).modelname = mfilename;
    result(istart).term1_mtx = term1_mtx;
    result(istart).happyscore  = happyscore;
    if (sum(isnan(priors))>=1) 
        if (istart==1)
            %         rangeW1     = [0.1];
            %         rangeW2     = [1];
            %         rangeGamma = 0.1:0.1:1;
            %         rangeCst   = [0.5];
            %         inx = CombVec(rangeW1,rangeGamma,rangeW2,rangeCst)';
            %         rangeW1     = [.1.*ones(10,1)];%[-1:1:1];
            %         rangeW2     = [1.*ones(10,1)];%[-1:1:1];
            %         rangeGamma = [0.1:0.1:1]';
            %         rangeCst   = 0.5.*ones(10,1);%[0.25:0.25:0.75];
            %         inx = [rangeW1,rangeW2,rangeGamma,rangeCst];%CombVec(rangeW1,rangeGamma,rangeCst)';
            rangeW1    = 0.1;%[-0.1 0.1];%[.1.*ones(10,1)];%[-1:1:1];
            rangeW2    = 0.1;%[-1 1];%[.1.*ones(10,1)];%[-1:1:1];
            rangeGamma = 0.5;%[0.1:0.1:1]';%rand(10,1);%[0.1:0.1:1]';
            rangeCst   = 0.5;%.*ones(10,1);%[0.25:0.25:0.75];
            inx = CombVec(rangeW1, rangeGamma,rangeW2, rangeCst)';%;%CombVec(rangeW1,rangeW2,rangeGamma',rangeCst)';
        end
    else
        inx = priors;%;
    end
    
    inx(istart,:) = CombVec(0.1,0.5,lW2(istart),0)';
    
    options = optimset('Display','off','MaxIter',10000,'TolFun',1e-5,'TolX',1e-5,...
        'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',1000,'LargeScale','off');
    warning off; %display,iter to see outputs
    %lb = [-5 -5 -5 0];
    %ub = [5 5 5 1]; %max tau of 1.1 or use 1.5
    lb = [-16 0 -max(term1_mtx(:))];%-max(term1_mtx(:))
    ub = [16  1 max(term1_mtx(:))]; %max tau of 1.1 or use 1.5
    if exist('constant'),
        if type=='r'
            inx(istart,:) = [inx(istart,:)];
            lb = [lb 0];
            ub = [ub 1];
        elseif type=='z'
            inx(istart,:) = [inx(istart,:)];
            lb = [lb  -2];
            ub = [ub  2];
        end
    else
        inx = [inx(istart,:) 0];
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
a = x(1);b = x(3); tau1 = x(2); const = x(end);
[happypred] = happy_model_1termRef(a,b,tau1,const, result.term1_mtx);
sse         = sum((result.happyscore-happypred).^2); %sum least-squares error
re          = sum((result.happyscore-mean(result.happyscore)).^2);
happyr2     = 1-sse/re;
