function [happypred] = happy_model_1term(a,tau,const,term1_mtx)
% happiness model for including one term
decayvec  = tau.^[0:(size(term1_mtx,2)-1)]; 
decayvec  = decayvec(:);
term1     = term1_mtx; 
dec       = decayvec;
happypred =  a*term1*dec  + const;

