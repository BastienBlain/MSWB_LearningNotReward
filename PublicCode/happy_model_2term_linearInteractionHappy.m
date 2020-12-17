function [happypred] = happy_model_2term_linearInteractionHappy(a,b,tau,const,term1_mtx)
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
            happypred(t) =  a*sum(term1(t,:).*dec') - b*happypred(t-1) + const;%(happypred(t-1)*-const)*sum(term1(t,:).*dec');
        else
            happypred(t) =  a*sum(term1(t,:).*dec') + b*(1-happypred(t-1)) + const;
        end
    end
end
happypred = happypred';

% % to get what this model is doing check the following:
% weight = [-0.5 0:0.1:0.8];
% PPE = 0.8;
% lcolor = colormap(jet(length(weight)));
% figure;
% subplot(1,2,1);hold on;
% for iw=1:length(weight)
% plot([0:0.1:1],PPE-(([0:0.1:1]-0.5).*weight(iw)),'Color',lcolor(iw,:))
% zz{iw} = num2str(weight(iw));
% end
% leg = legend(zz);
% title(leg,'weight = ')
% ylabel('PPE impact on happiness')
% xlabel('Previous happiness level')
% title('PPE = 0.8')
% ylim([-0.8 0.8])
% 
% PPE = -0.8;
% lcolor = colormap(jet(length(weight)));
% subplot(1,2,2);hold on;
% for iw=1:length(weight)
% plot([0:0.1:1],PPE+((1-[0:0.1:1]).*weight(iw)),'Color',lcolor(iw,:))
% zz{iw} = num2str(weight(iw));
% end
% ylabel('PPE impact on happiness')
% xlabel('Previous happiness level')
% ylim([-0.8 0.8])
% title('PPE = - 0.8')