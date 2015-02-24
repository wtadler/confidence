% gen.opt=model; % this is for after CCO
if ~strcmp(optimization_method,'mcmc_slice') && strcmp(data_type,'fake') && length(active_opt_models)==1 && length(active_gen_models) == 1 && strcmp(opt_models(active_opt_models).name, gen_models(active_gen_models).name)
    % COMPARE TRUE AND FITTED PARAMETERS IN SUBPLOTS
    figure;
    % for each parameter, plot all datasets
    for parameter = 1 : nParams
        subplot(5,5,parameter);
        extracted_params = [gen(active_gen_models).opt(active_opt_models).extracted.best_params];
        plot(gen(active_gen_models).p(parameter,:), extracted_params(parameter,:), '.','markersize',10);
        hold on
        xlim([g.lb_gen(parameter) g.ub_gen(parameter)]);
        ylim([g.lb_gen(parameter) g.ub_gen(parameter)]);
        
        %axis square;
        plot([g.lb(parameter) g.ub(parameter)], [g.lb(parameter) g.ub(parameter)], '--');
        
        title(g.parameter_names{parameter});
    end
    %         suplabel('true parameter', 'x');
    %         suplabel('extracted parameter', 'y');
elseif strcmp(optimization_method,'mcmc_slice')
    % DIAGNOSE MCMC
    % open windows for every model/dataset combo.
    for gen_model_id = 1%:length(gen)
        g = gen_models(gen_model_id);
        if strcmp(data_type,'real')
            g.name = [];
        end
        for opt_model = 1:length(gen(gen_model_id).opt) % active_opt_models
            o = gen(gen_model_id).opt(opt_model);
            for dataset_id = 1:length(o.extracted) % dataset
                if ~isempty(o.extracted(dataset_id).p) % if there's data here
                    nChains = size(o.extracted(dataset_id).p,3);
                    [samples,logposteriors]=deal(cell(1,nChains));
                    
                    extraburn_prop=0; % burn some proportion of the samples
                    for c = 1:nChains
                        nSamples = size(o.extracted(dataset_id).p,1);
                        burn_start = max(1,round(nSamples*extraburn_prop));
                        samples{c} = o.extracted(dataset_id).p(burn_start:end,:,c);
                        logposteriors{c} = -o.extracted(dataset_id).nll(burn_start:end,c) + o.extracted(dataset_id).log_prior(burn_start:end,c);
                    end
                    
                    [true_p,true_logposterior]=deal([]);
                    if strcmp(data_type,'fake') && strcmp(o.name, g.name)
                        true_p = gen(gen_model_id).p(:,dataset_id);
                        true_logposterior = gen(gen_model_id).data(dataset_id).true_logposterior;
                    end
                    [fh,ah]=mcmcdiagnosis(samples,'logposteriors',logposteriors,'fit_model',o,'true_p',true_p,'true_logposterior',true_logposterior,'dataset',dataset_id,'dic',o.extracted(dataset_id).dic,'gen_model',g);
                    pause(.5); % to plot
                end
            end
        end
    end
end
