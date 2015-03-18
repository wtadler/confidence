%% bound inspector
% after cco
for m = 1:length(model)
    model_samples{m}=[];
    for d = 1:length(model(m).extracted);
        subject_samples = [];
        for c=1:length(model(m).extracted(d).p)
            subject_samples = cat(1,subject_samples,model(m).extracted(d).p{c});
        end
        model_samples{m}=cat(1,model_samples{m},subject_samples);
    end
    nParams = length(model(m).parameter_names);
    for p = 1:nParams
        tight_subplot(5,nParams,m,p,[.06,.005])
        hist(model_samples{m}(:,p),50);
        xlim([model(m).lb(p) model(m).ub(p)])
        xlabel(model(m).parameter_names{p})
        set(gca,'yticklabel','','box','off')
    end
end