function [happypred] = happy_model_1termRef(a,b,tau,const,term1_mtx)
% happiness model for including one term
decayvec  = tau.^[0:(size(term1_mtx,2)-1)]; 
decayvec  = decayvec(:);
term1     = term1_mtx-b;
term1(isnan(term1)) = 0;
dec       = decayvec;
happypred =  a*(term1)*dec  + const;

