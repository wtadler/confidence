function model_out = compute_mcm(model_in, varargin)

MCM = 'dic';
datadir='/Users/will/Google Drive/Will - Confidence/Data/v3_all';
maxWorkers = 0;
assignopts(who, varargin);

datadir = check_datadir(datadir);

datadir

save cmtest

stA = compile_data('datadir', datadir.A);
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
    
    for d = 1:length(model_in(m).extracted)
        
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
                
            case 'waic'
                loglikes = nan(nSamples, nTrials);
                
                tenth = floor(nSamples/10);
                
                for p = 1:nSamples
                    [~, ll_trials] = loglik_fcn(all_p(p,:));
                    loglikes(p, :) = ll_trials;
                    
                    if mod(p, tenth) == 0
                        prop_complete = p/nSamples;
                        secs_remaining = (toc(t_start)/prop_complete - toc(t_start));
                        fprintf('model %i/%i, subject %i/%i: %.i%% complete, %.f secs remaining\n', m, nModels, d, nSubjects, round(100*prop_complete), secs_remaining)
                    end
                end
                
                [waic1, waic2] = waic(loglikes);
                
                model_in(m).extracted(d).waic1 = waic1;
                model_in(m).extracted(d).waic2 = waic2;
                
        end
    end
end

model_out = model_in;