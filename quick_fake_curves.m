%% examine fake data quickly

% this is a good script to make sure that lb_gen and ub_gen are well chosen before parameter recovery.
% it doesn't help much with ub and lb because you dont know what parameters subjects might be using.

clear g
% run constraints
g = struct;

g(1).family = 'neural1';
g(1).multi_lapse = 0;
g(1).partial_lapse = 0;
g(1).repeat_lapse = 0;
g(1).choice_only = 1;
g(1).diff_mean_same_std = 0;
g(1).ori_dep_noise = 0; 
g(1).symmetric = 1;
g(1).joint_task_fit = 0;

g = parameter_constraints(g);

%%
clf
nSets = 8;
p = random_param_generator(nSets,g(1), 'generating_flag', true);
% p = repmat(random_param_generator(1, g(1), 'generating_flag', true),1,nSets);
p(4,:) = 0; % no lapse
p(5,:) = exp(linspace(0,1.7,nSets));

gen_nSamples = 1e5;

nBins = 20;

if g(1).diff_mean_same_std == 1
    category_params.category_type = 'diff_mean_same_std';
elseif g(1).diff_mean_same_std == 0
    category_params.category_type = 'same_mean_diff_std'; % 'same_mean_diff_std' (Qamar) or 'diff_mean_same_std' or 'sym_uniform' or 'half_gaussian' (Kepecs)
end

category_params.sigma_1 = 3;
category_params.sigma_2 = 12;
category_params.sigma_s = 5; % for 'diff_mean_same_std' and 'half_gaussian'
category_params.a = 0; % overlap for sym_uniform
category_params.mu_1 = -4; % mean for 'diff_mean_same_std'
category_params.mu_2 = 4;
category_params.uniform_range = 1;

for i = 1:nSets
    st(1).data(i).raw = trial_generator(p(:,i), g(1), 'category_params', category_params, 'n_samples', gen_nSamples);
    %st(2).data(i).raw = trial_generator(p(:,i), 'n_samples', gen_nSamples, 'dist_type', 'qamar', 'contrasts', exp(-4:.5:-1.5), 'model', 'quad');
end

[model_bins, model_axis] = bin_generator(nBins);

for dataset = 1:nSets
    [st(1).data(dataset).stats, st(1).data(dataset).sorted_raw] = indiv_analysis_fcn(st(1).data(dataset).raw, model_bins);
    %[st(2).data(dataset).stats, st(2).data(dataset).sorted_raw] = indiv_analysis_fcn(st(2).data(dataset).raw, model_bins);
end
%st(1).sumstats = sumstats_fcn(st(1).data);
%st(2).sumstats = sumstats_fcn(st(2).data);

for i = 1:nSets
    subplot(2,nSets,i)
    plot(model_axis,st(1).data(i).stats.all.mean.Chat)
    hold on
    ylim([0 1])
    subplot(2,nSets,i+nSets)
    plot(model_axis,st(1).data(i).stats.all.mean.g)
    %plot(model_axis,st(2).data(i).stats.all.Chat1_prop)
    ylim([1 4])
end
