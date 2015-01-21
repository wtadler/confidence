function [samples, neval, logliks] = slice_sample_test(varargin)

%%
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
opt_models(1+d).family = 'opt'; %1e5 samples not enough. neval 21. asym. try sym. instead
opt_models(1+d).multi_lapse = 1;
opt_models(1+d).partial_lapse = 1;
opt_models(1+d).repeat_lapse = 1;
opt_models(1+d).choice_only = 0;
opt_models(1+d).diff_mean_same_std = 0;
opt_models(1+d).ori_dep_noise = 0;
opt_models(1+d).symmetric = 1;

opt_models(2+d) = opt_models(1+d);
opt_models(2+d).family = 'fixed';

opt_models(3+d) = opt_models(1+d);
opt_models(3+d).family = 'lin';

opt_models(4+d) = opt_models(1+d);
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
    ww = @(x) wrapper(x, g, data(dataset).raw);
    
    parfor c = 1:nChains
        [samples{c},neval(c)]=slicesample(x0(c,:), nKeptSamples,'logpdf', ww, 'width', g.ub-g.lb, 'burnin', burnin, 'thin', thin);
        logliks{c} = nan(nKeptSamples,1);
        for j = 1:nKeptSamples
            logliks{c}(j) = ww(samples{c}(j,:));
        end
    end
end

save sst2.mat

bins = 30;

if ~hpc
    for f=1:3
        fig{f}=figure;
    end
    
    for c = 1:nChains
        figure(fig{1});
        plot(logliks{c})
        hold on
        
        figure(fig{2});
        [n,x]=hist(logliks{c},bins);
        plot(x,n)
        hold on
        
        figure(fig{3});
        for p = 1:nParams
            subplot(5,5,p)
            
            [n,x]=hist(samples{c}(:,p),bins);
            plot(x,n)
            hold on
            if c==nChains
                yl=get(gca,'ylim');
                plot([true_p(p,dataset) true_p(p,dataset)],yl,'k--')
                xlabel(g.parameter_names{p})
                ylabel('freq')
            end
        end
        corner_plot(samples{c},'truths',true_p,'names',g.parameter_names);
    end
    
    figure(fig{1})
    xlabel('sample')
    ylabel('log lik')
    xl=get(gca,'xlim');
    plot(xl, [-data(dataset).true_nll -data(dataset).true_nll],'k--')
    
    figure(fig{2})
    xlabel('log lik')
    ylabel('freq')
    yl=get(gca,'ylim');
    plot([-data(dataset).true_nll -data(dataset).true_nll],yl,'k--')
    
end
end


function ll = wrapper(x,g,raw)
if any(x<g.lb) || any(x>g.ub) || any(g.A*x' > g.b)
    ll = -Inf;
else
    ll = -nloglik_fcn(x, raw, g);
end
end