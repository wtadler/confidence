function diagnosis_plots(model_struct, varargin)

fig_type = 'mcmc_grid'; % 'mcmc_figures' or 'mcmc_grid' or 'parameter_recovery'
    gen_struct = []; % just needs to contain true parameters, and match model_struct in length if doing parameter recovery
    show_cornerplot = true;
    plot_datasets = [];
    mcm = 'dic';
    
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
        nParams = length(m.parameter_names);
        if nParams ~= size(g.p, 1)
            error('generating and fitting models need to have the same numbers of parameters')
        end
        nRows = 4;
        nCols = ceil(nParams/nRows);

        % color points from green (fitted NLL lower than true NLL) to red (fitted NLL >> true_nll)
        delta = [m.extracted.min_nll]-[g.data.true_nll];
        worst_delta = max(delta);
        frac = max(0, delta/abs(worst_delta));
        color = [frac' 1-frac' zeros(length(m.extracted), 1)];
               
        % for each parameter, plot all datasets
        for parameter = 1:length(m.parameter_names)
            row = ceil(parameter/nCols);
            col = rem(parameter, nCols);
            col(col==0) = nCols;
            tight_subplot(nRows, nCols, row, col, [.05 .07], [.1 .02 .1 .1]);
            extracted_params = [m.extracted.best_params];
            scatter(g.p(parameter, :), extracted_params(parameter, :), 30, color, 'filled');
            hold on
            xlim([m.lb(parameter) m.ub(parameter)]);
            ylim([m.lb(parameter) m.ub(parameter)]);
            
            plot([m.lb(parameter) m.ub(parameter)], [m.lb(parameter) m.ub(parameter)], '--', 'linewidth', .5);
            
            xlabel(m.parameter_names{parameter});
            set(gca, 'tickdir','out','box','off')

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
        
        if isempty(m.name)
            continue
        end
        
        if isempty(plot_datasets)
            datasets = 1:length(m.extracted);
        else
            datasets = plot_datasets;
        end
        
        if ~isempty(gen_struct)
            g = gen_struct(model_id);
            if length(m.parameter_names) ~= length(g.parameter_names)
                error('generating and fitting models need to have the same numbers of parameters')
            end
        else
            g = [];
        end
        for dataset_id = datasets
            ex = m.extracted(dataset_id);
            
            if isempty(ex.name)
                continue
            end
            dataset_name = upper(ex.name);
            
            [true_p, true_logposterior] = deal([]);
            if ~isempty(gen_struct) % if this is mcmc recovery
                true_p = g.p(:, dataset_id);
                true_logposterior = g.data(dataset_id).true_logposterior;
            end
            
            tic
            try
            mcmcdiagnosis(ex.p, 'logposterior', ex.logposterior, 'fit_model', m, ...
                'true_p', true_p, 'true_logposterior', true_logposterior, 'dataset_name', dataset_name, ...
                'mcm', ex.(mcm), 'gen_model', g, 'show_cornerplot', show_cornerplot);
            
            catch
                'error'
                toc
            end
            pause(1e-3); % to plot
        end
    end
elseif strcmp(fig_type, 'mcmc_grid')
    figure;
    if ~isempty(gen_struct) && length(gen_struct) ~= length(model_struct)
        error('gen_struct and model_struct need to have the same length.')
    end
    
    show_legend = false;
    
    for model_id = 1:length(model_struct)
        m = model_struct(model_id);
        if isempty(m.name)
            continue
        end
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
            if isempty(ex.name)
                continue
            end
            [true_p, true_logposterior] = deal([]);
            if ~isempty(gen_struct) % if this is mcmc recovery
                true_p = g.p(:, dataset_id);
                true_logposterior = g.data(dataset_id).true_logposterior;
            end
            

            if dataset_id == length(m.extracted) && model_id == length(model_struct)
                show_legend = true;
            end
            
            tight_subplot(length(model_struct), 2*length(m.extracted), model_id, 2*dataset_id-1, .01, [.06 .01 .04 .06]);
            plot_logposterior_over_samples(ex.logposterior, 'true_logposterior', true_logposterior', 'show_legend', false, 'show_labels', false)
            if model_id == 1
                set(gca, 'visible', 'on')
                title(sprintf('%i %s', dataset_id, upper(m.extracted(dataset_id).name)))
            end
            if dataset_id == 1
                set(gca, 'visible', 'on')
                ylabel(rename_models(m.name))
            end
            tight_subplot(length(model_struct), 2*length(m.extracted), model_id, 2*dataset_id, .01, [.06 .01 .04 .06]);
            plot_logposterior_hist(ex.logposterior, 'true_logposterior', true_logposterior, 'show_legend', show_legend, 'show_labels', false)
            view(90,-90)

        end
    end
end