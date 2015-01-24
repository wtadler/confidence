function [samples, neval, logliks] = slice_sample_test(varargin)

%%
opt_models = struct;
d = 0;
opt_models(1+d).family = 'opt'; % works well with 1e4 samples
opt_models(1+d).multi_lapse = 0;
opt_models(1+d).partial_lapse = 0;
opt_models(1+d).repeat_lapse = 1;
opt_models(1+d).choice_only = 1;
opt_models(1+d).diff_mean_same_std = 0;
opt_models(1+d).ori_dep_noise = 0;

opt_models(2+d) = opt_models(1+d); %1e5 samples. 1459 seconds
opt_models(2+d).family = 'fixed';

opt_models(3+d) = opt_models(1+d); %5e4 samples. 1772 seconds. looking a bit ragged though. neval 17.
opt_models(3+d).family = 'lin';

opt_models(4+d) = opt_models(1+d); %1e5 samples 2270 seconds. a bit ragged. neval 16
opt_models(4+d).family = 'quad';

opt_models(5+d) = opt_models(1+d); %1e5 samples, ? seconds. neval 11
opt_models(5+d).family = 'MAP';

d = 5;
opt_models(1+d).family = 'opt';
opt_models(1+d).multi_lapse = 0;
opt_models(1+d).partial_lapse = 0;
opt_models(1+d).repeat_lapse = 0;
opt_models(1+d).choice_only = 1;
opt_models(1+d).diff_mean_same_std = 0;
opt_models(1+d).ori_dep_noise = 1;

opt_models(2+d) = opt_models(1+d);
opt_models(2+d).family = 'fixed';

opt_models(3+d) = opt_models(1+d);
opt_models(3+d).family = 'lin';

opt_models(4+d) = opt_models(1+d);
opt_models(4+d).family = 'quad';

opt_models(5+d) = opt_models(1+d); % TAKES A LONNNNNNGGGGG TIME!!!! 8 hours to do 90 opts, on 4 processors
opt_models(5+d).family = 'MAP';

d = 10;
opt_models(1+d).family = 'opt'; %1e6 samples. neval 18. took a long time, like 10 hours? added symmetric=1.
opt_models(1+d).multi_lapse = 1;
opt_models(1+d).partial_lapse = 1;
opt_models(1+d).repeat_lapse = 1;
opt_models(1+d).choice_only = 0;
opt_models(1+d).diff_mean_same_std = 0;
opt_models(1+d).ori_dep_noise = 0;

opt_models(2+d) = opt_models(1+d); %5e5 samples. neval 15. took 3 hours
opt_models(2+d).family = 'fixed';

opt_models(3+d) = opt_models(1+d); %1e5 not enough. 5e5 samples still not enough. chains hoppingneval 18. took 3.5 h. 1.5e6 enough. 7.5 h. neval 18
opt_models(3+d).family = 'lin';

opt_models(4+d) = opt_models(1+d); %1e5 samples not enough. neval 18
opt_models(4+d).family = 'quad';

opt_models(5+d) = opt_models(1+d);
opt_models(5+d).family = 'MAP';

d = 15;
opt_models(1+d).family = 'opt';
opt_models(1+d).multi_lapse = 0;
opt_models(1+d).partial_lapse = 0;
opt_models(1+d).repeat_lapse = 0;
opt_models(1+d).choice_only = 0;
opt_models(1+d).diff_mean_same_std = 0;
opt_models(1+d).ori_dep_noise = 1;

opt_models(2+d) = opt_models(1+d);
opt_models(2+d).family = 'fixed';

opt_models(3+d) = opt_models(1+d);
opt_models(3+d).family = 'lin';

opt_models(4+d) = opt_models(1+d);
opt_models(4+d).family = 'quad';

opt_models(5+d) = opt_models(1+d);
opt_models(5+d).family = 'MAP';

d = 20;
opt_models(1+d).family = 'opt'; %sigma_d a bit off in one dataset
opt_models(1+d).multi_lapse = 0;
opt_models(1+d).partial_lapse = 0;
opt_models(1+d).repeat_lapse = 0;
opt_models(1+d).choice_only = 1;
opt_models(1+d).diff_mean_same_std = 0;
opt_models(1+d).ori_dep_noise = 0;
opt_models(1+d).d_noise = 1;

opt_models(22) = opt_models(21); %sigma_d is off in one, and sig amp is off for both
opt_models(22).ori_dep_noise = 1;

opt_models(23) = opt_models(21);
opt_models(23).choice_only = 0;

opt_models(24) = opt_models(23);
opt_models(24).ori_dep_noise = 1; % works lovely?

opt_models(25) = opt_models(23);
opt_models(25).symmetric = 1;

opt_models(26) = opt_models(24);
opt_models(26).symmetric = 1; % this works? better than model 22 which is choice only? why?

opt_models = parameter_constraints(opt_models);
%%
hpc = 0;
active_model = 1; % do 1 2 3 4 5
g=opt_models(active_model);
nDatasets=1;
nTrials = 3240;
nChains = 4;
nRealSamples = 1e5; % important for this to be high
progress_report_interval = 1;
assignopts(who,varargin);


nKeptSamples = nRealSamples; % doesn't matter as much
g=opt_models(active_model);
assignopts(who,varargin);

true_p=random_param_generator(nDatasets,g,'generating_flag',1);

nParams = length(true_p);

data = struct;
for dataset = 1:nDatasets
    data(dataset).raw = trial_generator(true_p(:,dataset),g,'gen_nSamples',nTrials);
    data(dataset).true_nll = nloglik_fcn(true_p(:,dataset),data(dataset).raw, g);
end

x0 = random_param_generator(nChains,g)';

-data(dataset).true_nll
thin = max(1,round(nRealSamples/nKeptSamples));
burnin = nRealSamples/2;

samples = cell(1,nChains);
neval = zeros(1,nChains);
logliks = cell(1,nChains);

for dataset = 1:nDatasets
    ww = @wrapper;
    parfor c = 1:nChains
        [samples{c},neval(c)]=slicesample(x0(c,:), nKeptSamples,'logpdf', ww, 'width', g.ub-g.lb, 'burnin', burnin, 'thin', thin);
        logliks{c} = nan(nKeptSamples,1);
        for j = 1:nKeptSamples
            logliks{c}(j) = ww(samples{c}(j,:));
        end
    end
end

save sst2.mat
%%

if ~hpc
    [fh,ah]=mcmcdiagnosis(samples,logliks,g,true_p,-data(dataset).true_nll);
end

%%
    function ll = wrapper(x)
        if any(x<g.lb) || any(x>g.ub) || any(g.A*x' > g.b) % re-parameterize A stuff. Should result in like a 2% speed up.
            ll = -Inf;
        else
            ll = -nloglik_fcn(x, data(dataset).raw, g); % + logprior_fcn(x, g);
        end
    end
end