function [happypred] = happy_model_3terms_inter(a,b,c,tau1,tau2,const,term1_mtx,term2_mtx,term3_mtx)
% happiness model for including one term
decayvecterm1  = tau1.^[0:size(term1_mtx,2)-1]; 
decterm1       = decayvecterm1(:);
decayvecterm2  = tau2.^[0:size(term1_mtx,2)-1];
term1          = term1_mtx; 
term2          = term2_mtx; 
term3          = term3_mtx; 
happypred      = a*term1*decterm1 + b*term2' +c*term3' + const;


