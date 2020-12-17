function [happypred] = happy_model_1term_interaction(a,tau,const,term1_mtx)
% happiness model for including one term
decayvec  = tau.^[0:(size(term1_mtx,2)-1)]; 
decayvec  = decayvec(:);
term1     = term1_mtx; 
nrow = size(term1,1);
term2     = [zeros(1,size(term1,2)); term1_mtx(1:(nrow-1),:)];
dec       = decayvec;
happypred =  a*((term1*dec).*(term2*dec)) + const;

