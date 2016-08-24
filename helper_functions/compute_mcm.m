function model_out = compute_mcm(model_in, varargin)

MCM = 'dic';
datadir='~/Google Drive/Will - Confidence/Data/v3_all';
maxWorkers = 0;
assignopts(who, varargin);

datadir = check_datadir(datadir);

datadir

if isfield(datadir, 'A')
    stA = compile_data('datadir', datadir.A);
end
stB = compile_data('datadir', datadir.B);

category_params.sigma_s = 5; % for 'diff_mean_same_std' and 'half_gaussian'
category_params.a = 0; % overlap for sym_uniform
category_params.mu_1 = -4; % mean for 'diff_mean_same_std'
category_params.mu_2 = 4;
category_params.uniform_range = 1;
category_params.sigma_1 = 3;
category_params.sigma_2 = 12;

t_start = tic;

nModels = length(model_in);

for m = 1:nModels
    nSubjects = length(model_in(m).extracted);
    
    for d = 1:nSubjects
        if isempty(model_in(m).extracted(d).p)
            continue
        end
        if model_in(m).joint_task_fit
            sm = prepare_submodels(model_in(m));
            loglik_fcn = @(p) two_task_ll_wrapper(p, stA.data(d).raw, stB.data(d).raw, sm, 51, category_params, true, true);
            flip_ll_sign = false;
            nTrials = length(stA.data(d).raw.C) + length(stB.data(d).raw.C);
        else
            if model_in(m).diff_mean_same_std
                data_st = stA;
                nTrials = length(stA.data(d).raw.C);
            else
                data_st = stB;
                nTrials = length(stB.data(d).raw.C);
            end
            
            loglik_fcn = @(p) nloglik_fcn(p, data_st.data(d).raw, model_in(m), 51, category_params);
            flip_ll_sign = true;
        end
        
        all_p = vertcat(model_in(m).extracted(d).p{:});
        nSamples = size(all_p, 1);
        
        switch MCM
            case 'dic'
                [dic_score, dbar, dtbar] = dic(all_p, ...
                    model_in(m).extracted(d).nll, loglik_fcn, flip_ll_sign);
                
                model_in(m).extracted(d).dic   = dic_score;
                model_in(m).extracted(d).dbar  = dbar;
                model_in(m).extracted(d).dtbar = dtbar;
                
            case {'waic','psis'}
%                 nSamples = round(nSamples/100);
                loglikes = nan(nSamples, nTrials);
                
                tenth = floor(nSamples/10);
                
                parfor(p = 1:nSamples, maxWorkers)
                    [~, ll_trials] = loglik_fcn(all_p(p,:));
                    loglikes(p, :) = ll_trials;
                    
                    if mod(p, tenth) == 0
                        prop_complete = (nSubjects*(m-1)+d)/nSubjects/nModels;
                        secs_remaining = (toc(t_start)/prop_complete - toc(t_start));
                        fprintf('model %i/%i, subject %i/%i: %.1f%% complete, %.f mins remaining\n', m, nModels, d, nSubjects, 100*prop_complete, secs_remaining/60)
                    end
                end
                
                [waic1, waic2] = waic(loglikes);
                
                model_in(m).extracted(d).waic1 = waic1;
                model_in(m).extracted(d).waic2 = waic2;
                
                [model_in(m).extracted(d).loopsis, loos, pk] = psisloo(loglikes);
                if all(pk<0.5)
                    model_in(m).extracted(d).pareto_tail_indices = 'all under 0.5';
                else
                    model_in(m).extracted(d).loos = loos;
                    model_in(m).extracted(d).pareto_tail_indices = pk;
                end

                
        end
    end
end

model_out = model_in;