function [happypred] = happy_model_2term_interaction(a,b,tau,const,term1_mtx)
% happiness model for including one term
decayvec  = tau.^[0:(size(term1_mtx,2)-1)]; 
decayvec  = decayvec(:);
term1     = term1_mtx; 
nrow      = size(term1,1);
term2     = [zeros(1,size(term1,2)); term1_mtx(1:(nrow-1),:)];
dec       = decayvec;
happypred =  a*term1*dec + b*((term1*dec).*(a*term2*dec)) + const;

