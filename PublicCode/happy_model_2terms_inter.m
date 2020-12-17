function [happypred] = happy_model_2terms_inter(a,b,tau1,tau2,const,term1_mtx,term2_mtx)
% happiness model for including one term
decayvecterm1  = tau1.^[0:size(term1_mtx,2)-1]; 
decterm1       = decayvecterm1(:);
decayvecterm2  = tau2.^[0:size(term1_mtx,2)-1];
term1          = term1_mtx; 
term2          = term2_mtx; 
happypred      = a*term1*decterm1 + b*term2' + const;


