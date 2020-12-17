function [happypred] = happy_model_2term_interactionHappy(a,b,c,tau,const,term1_mtx)
% happiness model for including one term
decayvec  = tau.^[0:(size(term1_mtx,2)-1)]; 
decayvec  = decayvec(:);
term1     = term1_mtx; 
dec       = decayvec;
for t=1:size(term1,1)
    if t==1
        happypred(t) =  a*sum(term1(t,:).*dec') + const;
    else
        if sum(term1(t,:).*dec') >= 0
            powerTerm = (happypred(t-1)+sum(term1(t,:).*dec')).^c - happypred(t-1).^c;
            happypred(t) =  a*sum(term1(t,:).*dec') + const + b*powerTerm;%(happypred(t-1)*-const)*sum(term1(t,:).*dec');
        else
            powerTerm = -[((1-happypred(t-1))+abs(sum(term1(t,:).*dec'))).^c - (1-happypred(t-1)).^c];
            happypred(t) =  a*sum(term1(t,:).*dec') + const + b*powerTerm;
        end
    end
end
happypred = happypred';

% % to get what this model is doing check the following:
% c = [0.4 0.6 0.8 1 1.2 1.4];
% lcolor = colormap(jet(length(c)));
% figure;hold on;
% for ic=1:length(c)
% plot([0:0.1:1],([0:0.1:1]+0.12).^c(ic)-[0:0.1:1].^c(ic),'Color',lcolor(ic,:))
% zz{ic} = num2str(c(ic));
% end
% leg = legend(zz);
% title(leg,'Power = ')
% ylabel('PPE impact on happiness')
% xlabel('Previous happiness level')
% 
% figure;hold on;
% for ic=1:length(c)
% plot([0:0.1:1],-((1-[0:0.1:1]+0.12).^c(ic)-(1-[0:0.1:1]).^c(ic)),'Color',lcolor(ic,:))
% end
% ylabel('PPE impact on happiness')
% xlabel('Previous happiness level')
% leg = legend(zz);
% title(leg,'Power = ')