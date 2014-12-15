clear g
% run constraints
g(1).family = 'opt';
g(1).symmetric = 0;
g(1).d_noise = 0;
g(1).multi_lapse = 1;
g(1).partial_lapse = 1;
g(1).repeat_lapse = 1;
g(1).choice_only = 0;
g(1).non_overlap = 1;
g(1).free_cats = 0;

g(2).family = 'opt';
g(2).symmetric = 0;
g(2).d_noise = 0;
g(2).multi_lapse = 1;
g(2).partial_lapse = 1;
g(2).repeat_lapse = 1;
g(2).choice_only = 0;
g(2).non_overlap = 0;
g(2).free_cats = 0;

g = parameter_constraints(g);

%%
gen_nSamples = 1e5;
nBins = 20;
%p = random_param_generator(8, g.lb_gen, g.ub_gen, g.A, g.b, 'monotonic_params', g.monotonic_params, 'model', g)
%p = repmat(random_param_generator(1, g.lb_gen, g.ub_gen, g.A, g.b, 'monotonic_params', g.monotonic_params, 'model', g),1,8);
p(4,:) = -5;%linspace(-5,-.8,8);
p(5,:) = -1;%linspace(-3,-.1,8);
p(6,:) = -.01;
p(7,:) = .01;
p(8,:) = 1;
p(9,:) = 2;
p(10,:) = linspace(.21,8,8);

%p(12,:) = linspace(0,.5,8);
%p(13,:) = linspace(0,.6,8);

for i = 1:8
    st(1).data(i).raw = trial_generator(p(:,i), g(1), 'n_samples', gen_nSamples, 'dist_type', 'qamar', 'contrasts', exp(-4:.5:-1.5));
    %st(2).data(i).raw = trial_generator(p(:,i), 'n_samples', gen_nSamples, 'dist_type', 'qamar', 'contrasts', exp(-4:.5:-1.5), 'model', 'quad');
end

[model_bins, model_axis] = bin_generator(nBins);

for dataset = 1:8
    [st(1).data(dataset).stats, st(1).data(dataset).sorted_raw] = indiv_analysis_fcn(st(1).data(dataset).raw, model_bins);
    %[st(2).data(dataset).stats, st(2).data(dataset).sorted_raw] = indiv_analysis_fcn(st(2).data(dataset).raw, model_bins);
end
%st(1).sumstats = sumstats_fcn(st(1).data);
%st(2).sumstats = sumstats_fcn(st(2).data);

for i = 1:8
    subplot(2,8,i)
    plot(model_axis,st(1).data(i).stats.all.Chat1_prop)
    hold on
    ylim([0 1])
    subplot(2,8,i+8)
    plot(model_axis,st(1).data(i).stats.all.g_mean)
    %plot(model_axis,st(2).data(i).stats.all.Chat1_prop)
    ylim([1 4])
end
