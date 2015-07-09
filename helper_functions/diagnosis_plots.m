function diagnosis_plots(model_struct, varargin)

fig_type = 'mcmc_grid'; % 'mcmc_figures' or 'mcmc_grid' or 'parameter_recovery'
    gen_struct = []; % just needs to contain true parameters, and match model_struct in length if doing parameter recovery
    show_cornerplot = true;

assignopts(who,varargin);

if strcmp(fig_type, 'parameter_recovery')
    % COMPARE TRUE AND FITTED PARAMETERS IN SUBPLOTS

    if length(gen_struct) ~= length(model_struct) || isempty(gen_struct)
        error('gen_struct and model_struct need to have the same length.')
    end

    % for each model plot a figure
    for model_id = 1:length(model_struct)
        figure;

        g = gen_struct(model_id);
        m = model_struct(model_id);
        if length(m.parameter_names) ~= length(g.parameter_names)
            error('generating and fitting models need to have the same numbers of parameters')
        end
        
        % for each parameter, plot all datasets
        for parameter = 1:length(m.parameter_names)
            subplot(5,5,parameter);
            extracted_params = [m.extracted.best_params];
            plot(g.p(parameter,:), extracted_params(parameter,:), '.','markersize',10);
            hold on
            xlim([g.lb_gen(parameter) g.ub_gen(parameter)]);
            ylim([g.lb_gen(parameter) g.ub_gen(parameter)]);
            
            %axis square;
            plot([g.lb(parameter) g.ub(parameter)], [g.lb(parameter) g.ub(parameter)], '--');
            
            title(g.parameter_names{parameter});
        end
        suplabel(m.name, 't')
        suplabel('true parameter', 'x');
        suplabel('extracted parameter', 'y');

    end
elseif strcmp(fig_type, 'mcmc_figures')
    % DIAGNOSE MCMC
    % open windows for every model/dataset combo.
    if ~isempty(gen_struct) && length(gen_struct) ~= length(model_struct)
        error('gen_struct and model_struct need to have the same length.')
    end
    
    for model_id = 1:length(model_struct)
        m = model_struct(model_id);
        if ~isempty(gen_struct)
            g = gen_struct(model_id);
            if length(m.parameter_names) ~= length(g.parameter_names)
                error('generating and fitting models need to have the same numbers of parameters')
            end
        else
            g = [];
        end
        for dataset_id = 1:length(m.extracted)
            ex = m.extracted(dataset_id);
            
            [true_p, true_logposterior] = deal([]);
            if ~isempty(gen_struct) % if this is mcmc recovery
                true_p = g.p(:, dataset_id);
                true_logposterior = g.data(dataset_id).true_logposterior;
            end
            
            tic
            mcmcdiagnosis(ex.p, 'logposterior', ex.logposterior, 'fit_model', m, ...
                'true_p', true_p, 'true_logposterior', true_logposterior, 'dataset', dataset_id, ...
                'dic', ex.dic, 'gen_model', g, 'show_cornerplot', show_cornerplot);
            toc
            pause(1e-3); % to plot
        end
    end
elseif strcmp(fig_type, 'mcmc_grid')
    if ~isempty(gen_struct) && length(gen_struct) ~= length(model_struct)
        error('gen_struct and model_struct need to have the same length.')
    end
    
    show_legend = false;
    
    for model_id = 1:length(model_struct)
        m = model_struct(model_id);
        if ~isempty(gen_struct)
            g = gen_struct(model_id);
            if length(m.parameter_names) ~= length(g.parameter_names)
                error('generating and fitting models need to have the same numbers of parameters')
            end
        else
            g = [];
        end
        
        for dataset_id = 1:length(m.extracted)
            ex = m.extracted(dataset_id);
            
            [true_p, true_logposterior] = deal([]);
            if ~isempty(gen_struct) % if this is mcmc recovery
                true_p = g.p(:, dataset_id);
                true_logposterior = g.data(dataset_id).true_logposterior;
            end
            

            if dataset_id == length(m.extracted) && model_id == length(model_struct)
                show_legend = true;
            end
            
            tight_subplot(length(model_struct), 2*length(m.extracted), model_id, 2*dataset_id-1, [], [.06 .01 .04 .06]);
            plot_logposterior_over_samples(ex.logposterior, 'true_logposterior', true_logposterior', 'show_legend', false, 'show_labels', false)
            if model_id == 1
                set(gca, 'visible', 'on')
                title(sprintf('%i %s', dataset_id, upper(m.extracted(dataset_id).name)))
            end
            if dataset_id == 1
                set(gca, 'visible', 'on')
                ylabel(num2str(model_id))
            end
            tight_subplot(length(model_struct), 2*length(m.extracted), model_id, 2*dataset_id, [], [.06 .01 .04 .06]);
            plot_logposterior_hist(ex.logposterior, 'true_logposterior', true_logposterior, 'show_legend', show_legend, 'show_labels', false)
            view(90,-90)

        end
    end
end



return







show_cornerplot = false;
gen.opt=model; % this is for after CCO

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
        for opt_model_id = 1:length(gen(gen_model_id).opt) % active_opt_models
            o = gen(gen_model_id).opt(opt_model_id);
            for dataset_id = 1:length(o.extracted) % dataset
                ex = o.extracted(dataset_id);
                if ~isempty(ex.p) % if there's data here                    
                    [true_p,true_logposterior]=deal([]);
                    if strcmp(data_type,'fake') && strcmp(o.name, g.name)
                        true_p = gen(gen_model_id).p(:,dataset_id);
                        true_logposterior = gen(gen_model_id).data(dataset_id).true_logposterior;
                    end
                    tic
                    [fh,ah]=mcmcdiagnosis(ex.p,'logposterior',ex.logposterior,'fit_model',o,'true_p',true_p,'true_logposterior',true_logposterior,'dataset',dataset_id,'dic',ex.dic,'gen_model',g, 'show_cornerplot', show_cornerplot);
                    toc
                    pause(.00001); % to plot
                end
            end
        end
    end
end

