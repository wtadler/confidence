function tmp = end_early_routine(tmp)
vars = setdiff(fieldnames(tmp),'tmp');
dataset = 0; % initialize dataset
for v = 1:length(vars)
    commandLine = sprintf('%s = tmp.%s;', vars{v}, vars{v});

    eval(commandLine);
end

min_chain_completion = min(sum(~isnan(ex_nll)));
ex_nll = ex_nll(1:min_chain_completion,:);

done_samples = ~isnan(ex_nll);
done_samples = sum(done_samples);

ex_logprior = ex_logprior(1:min_chain_completion,:);
ex_p = ex_p(1:min_chain_completion,:,:);
%postburn = 0;
postburn = round(done_samples/2);

ex_p = ex_p(postburn+1:end,:,:);
ex_nll = ex_nll(postburn+1:end,:);
ex_logprior = ex_logprior(postburn+1:end,:);


ex.p = ex_p;
ex.nll = ex_nll;
ex.exitflag = ex_exitflag;
ex.output = ex_output;
ex.lambda = ex_lambda;
ex.grad = ex_grad;
ex.hessian = ex_hessian;
ex.ncall = ex_ncall;
ex.logprior = ex_logprior;


if crossvalidate
    fields = fieldnames(ex);
    for field = 1 : length(fields)
        gen(gen_model_id).opt(opt_model_id).extracted(dataset).(fields{field}) = ex.(fields{field});
    end
else
    if strcmp(optimization_method, 'mcmc_slice')
        %                     all_nll = vertcat(ex.nll{:});
        all_nll = reshape(ex.nll, numel(ex.nll),1);
        %                     all_p = vertcat(ex.p{:});
        all_p = reshape(permute(ex.p,[1 3 2]),[],size(ex.p,2),1);
        [ex.min_nll, ex.min_idx] = min(all_nll);
        ex.best_params = all_p(ex.min_idx,:)';
        ex.mean_params = mean(all_p);
%         dbar = 2*mean(all_nll);
%         f = @(p) nloglik_fcn(p, d, o, nDNoiseSets, category_params);%, optimization_method, randn_samples{dataset});
%         
%         dtbar= 2*f(ex.mean_params); % f is nll
%         ex.dic=2*dbar-dtbar; %DIC = 2(LL(theta_bar)-2LL_bar)
        
        ex.best_hessian = [];
        ex.hessian = [];
        ex.laplace = [];
        ex.n_good_params = [];
    else
        [ex.min_nll, ex.min_idx]    = min(ex.nll);
        ex.dic = [];
        ex.best_params          = ex.p(:, ex.min_idx);
        ex.n_good_params                          = sum(ex.nll < ex.min_nll + nll_tolerance & ex.nll > 10);
        paramprior      = o.param_prior;
        ex.best_hessian = ex.hessian(:,:,ex.min_idx);
        h               = ex.best_hessian;
        ex.laplace = -ex.min_nll + log(paramprior) +  (nParams/2)*log(2*pi) - .5 * log(det(h));
    end
    [ex.aic, ex.bic, ex.aicc] = aicbic(-ex.min_nll, nParams, gen_nSamples);
    if strcmp(data_type, 'real')
        gen(gen_model_id).opt(opt_model_id).extracted(dataset).name = gen.data(dataset).name; % need to change this to genB when just fitting task B data
    end
    if slimdown
        fields = {'p','nll','logprior','hessian','min_nll','min_idx','best_params','n_good_params','aic','bic','aicc','best_hessian','laplace'};%,'dic'};
    else
        fields = fieldnames(ex);
    end
    
    for field = 1 : length(fields)
        gen(gen_model_id).opt(opt_model_id).extracted(dataset).(fields{field}) = ex.(fields{field});
    end
end
clear ex;
vars = setdiff(vars,'ex');
for v = 1:length(vars)
    commandLine = sprintf('tmp.%s = %s;', vars{v}, vars{v});
    eval(commandLine)
end