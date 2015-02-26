%%

for mm = 1:5
    for d = 1:5
        [model(mm).extracted(d).p, model(mm).extracted(d).nll, model(mm).extracted(d).log_prior, model(mm).extracted(d).hessian] = deal([]);
    end
end


%% generate a bunch of fake data sets, fit them as if they were real.
nSubjects = 5;
datasetsPerSubject = 2;

opt_models = struct;

opt_models(1).family = 'opt';
opt_models(1).multi_lapse = 1;
opt_models(1).partial_lapse = 0;
opt_models(1).repeat_lapse = 0;
opt_models(1).choice_only = 0;
opt_models(1).diff_mean_same_std = 1;
opt_models(1).ori_dep_noise = 0;
opt_models(1).d_noise = 1;

opt_models(2) = opt_models(1);
opt_models(2).family = 'fixed';

opt_models(3) = opt_models(1);
opt_models(3).family = 'lin';

opt_models(4) = opt_models(1);
opt_models(4).family = 'quad';

opt_models(5) = opt_models(1);
opt_models(5).family = 'MAP';

opt_models = parameter_constraints(opt_models);

%%
category_params.sigma_s = 5; % for 'diff_mean_same_std' and 'half_gaussian'
category_params.a = 0; % overlap for sym_uniform
category_params.mu_1 = -4; % mean for 'diff_mean_same_std'
category_params.mu_2 = 4;
category_params.uniform_range = 1;
category_params.sigma_1 = 3;
category_params.sigma_2 = 12;

ext=load('/Users/will/Google Drive/Will - Confidence/Data/v3_A_fmincon_params_only.mat')

gen = struct;
for m = 1:length(opt_models)
%     gen(m) = rmfield(ext.m(m),'extracted');
    gen(m).name = ext.m(m).name;
    for d = 1:nSubjects
        for D = 1:datasetsPerSubject
            gen(m).p(:,datasetsPerSubject*(d-1)+D) = ext.m(m).extracted(d).best_params;
            gen(m).data(datasetsPerSubject*(d-1)+D).raw = trial_generator(ext.m(m).extracted(d).best_params, ext.m(m), 'n_samples', 2160, 'dist_type', 'diff_mean_same_std', 'contrasts', exp(linspace(-5.5,-2,6)));
            gen(m).data(datasetsPerSubject*(d-1)+D).true_nll = nloglik_fcn(ext.m(m).extracted(d).best_params, gen(m).data(datasetsPerSubject*(d-1)+D).raw, ext.m(m), 101, category_params);            
        end
    end
end