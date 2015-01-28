function gen=optimize_fcn(varargin)
%%
hpc = false;
job_id = '';

%%

opt_models = struct;

opt_models(1).family = 'fixed';
opt_models(1).multi_lapse = 0;
opt_models(1).partial_lapse = 0;
opt_models(1).repeat_lapse = 0;
opt_models(1).choice_only = 0;
opt_models(1).diff_mean_same_std = 0;
opt_models(1).ori_dep_noise = 0;

opt_models(2) = opt_models(1);
opt_models(2).family = 'lin';

opt_models(3) = opt_models(1);
opt_models(3).family = 'quad';

opt_models(4) = opt_models(1);
opt_models(4).family = 'opt';
opt_models(4).d_noise = 1;
opt_models(4).symmetric = 0;

opt_models(5) = opt_models(4);
opt_models(5).symmetric = 1;

opt_models(6) = opt_models(1);
opt_models(6).family = 'MAP';

opt_models(7).family = 'fixed';
opt_models(7).multi_lapse = 0;
opt_models(7).partial_lapse = 0;
opt_models(7).repeat_lapse = 1;
opt_models(7).choice_only = 1;

opt_models(8) = opt_models(7);
opt_models(8).family = 'lin';

opt_models(9) = opt_models(7);
opt_models(9).family = 'quad';

opt_models(10) = opt_models(7);
opt_models(10).family = 'opt';
opt_models(10).d_noise = 1;

opt_models(11) = opt_models(7);
opt_models(11).family = 'MAP';

opt_models(12) = opt_models(6);
opt_models(12).ori_dep_noise = 1;

opt_models(13) = opt_models(6);
opt_models(13).choice_only = 0;

opt_models(14) = opt_models(13);
opt_models(14).ori_dep_noise = 1;


opt_models = parameter_constraints(opt_models);
%%

assignopts(who,varargin);

active_opt_models = 1:length(opt_models);

%%
nRegOptimizations = 40;
nDNoiseOptimizations = 8;
nMAPOriDepNoiseOptimizations = 4;
RaiseOptsToParams = false;
nDNoiseSets = 101;
maxWorkers = Inf; % set this to 0 to turn parfor loops into for loops

progress_report_interval = Inf;
time_lim = Inf; % time limit for slice_sample

x0_reasonable = false; %limits starting points for optimization to those reasonable ranges defined in lb_gen and ub_gen

optimization_method = 'mcmc_slice'; % 'fmincon', 'lgo','npsol','mcs','snobfit','ga', 'patternsearch','mcmc_slice'
maxtomlabsecs = 100; % for lgo,npsol only
maxsnobfcalls = 10000; % for snobfit only
nKeptSamples = 1e4; % for mcmc_slice
nChains = 4;
slicesampler = 'luigi'; % 'builtin' or 'luigi'


data_type = 'real'; % 'real' or 'fake'
% 'fake' generates trials and responses, to test parameter extraction
% 'real' takes real trials and real responses, to extract parameters


%% fake data generation parameters
fake_data_params =  'random'; % 'extracted' or 'random'

dist_type = 'same_mean_diff_std'; % 'same_mean_diff_std' (Qamar) or 'diff_mean_same_std' or 'sym_uniform' or 'half_gaussian' (Kepecs)

category_params.sigma_s = 1; % for 'diff_mean_same_std' and 'half_gaussian'
category_params.a = 0; % overlap for sym_uniform
category_params.mu_1 = 5; % mean for 'diff_mean_same_std'
category_params.mu_2 = -5;
category_params.uniform_range = 1;
category_params.sigma_1 = 3;
category_params.sigma_2 = 12;

gen_models = opt_models;

active_gen_models = 1 : length(gen_models);
gen_nSamples = 3240;
fixed_params_gen = []; % can fix parameters here. set fixed values in beq, in parameter_constraints.m.
fixed_params_opt = [];
%%
slimdown = true; % throws away information like hessian for every optimization, etc. that takes up a lot of space.
crossvalidate = false;
k = 1; % for k-fold cross-validation

if ~crossvalidate
    k = 1;
end

nll_tolerance = 1e-3; % this is for determining what "good" parameters are.

if hpc
    datadir='/home/wta215/data/v2/';
else
    datadir = '/Users/will/Google Drive/Ma lab/repos/qamar confidence/data/v2/';
end

assignopts(who,varargin);

switch optimization_method
    case 'fmincon'
        fmincon_opts = optimoptions(@fmincon,'Algorithm','interior-point', 'display', 'off', 'UseParallel', 'never');
    case 'patternsearch'
        patternsearch_opts = psoptimset('MaxIter',1e4,'MaxFunEvals',1e4);
    case 'ga'
        ga_opts = gaoptimset('PlotFcns', {@gaplotbestf, @gaplotscorediversity, @gaplotbestindiv, @gaplotgenealogy, @gaplotscores, @gaplotrankhist});
end


if strcmp(data_type, 'real')
    gen = compile_data('datadir', datadir, 'crossvalidate', crossvalidate, 'k', k);
    nDatasets = length(gen.data);
    datasets = 1 : nDatasets;
    gen_models = struct; % to make it length 1, to execute big optimization loop just once.
    active_gen_models = 1;
elseif strcmp(data_type,'fake')
    nDatasets = 6;
    assignopts(who,varargin);
    datasets = 1:nDatasets;
    extracted_param_file = '';
end

assignopts(who,varargin); % reassign datasets (for multiple jobs)

datetime_str = datetimefcn;
if hpc
    savedir = '/home/wta215/Analysis/output/'
    
    if strcmp(data_type,'real')
        if length(datasets)==1
            filename = sprintf('%g_subject%g.mat', job_id, datasets)
        else
            filename = sprintf('%g_multiplesubjects.mat', job_id)
        end
    elseif strcmp(data_type,'fake')
        filename = sprintf('%g_model%g.mat', job_id, active_gen_models);
    end
else
    savedir = '/Users/will/Google Drive/Ma lab/output/'
    filename = sprintf('%s.mat', datetime_str)
end

%% GENERATE FAKE DATA %%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(data_type, 'fake')
    
    for gen_model_id = active_gen_models;
        
        g = gen_models(gen_model_id);
        o = g; % this is for log_posterior function
        
        % generate parameters, or use previously extracted parameters
        switch fake_data_params
            case 'random'
                gen(gen_model_id).p = random_param_generator(nDatasets, g, 'fixed_params', fixed_params_gen, 'generating_flag', 1);
                
            case 'extracted' % previously extracted using this script (deprecated, add model(i)... or something)
                
                if hpc
                    exm=load(extracted_param_file);
                else
                    exm=load('/Users/will/Google Drive/Will - Confidence/Analysis/4confmodels.mat');
                end
                
                for dataset = 1 : nDatasets;
                    gen(gen_model_id).p(:,dataset) = exm.m(gen_model_id).extracted(dataset).best_params;
                end
        end
        for dataset = datasets;
            % generate data from parameters
            d = trial_generator(gen(gen_model_id).p(:,dataset), g, 'n_samples', gen_nSamples, 'dist_type', dist_type, 'contrasts', exp(-4:.5:-1.5), 'category_params', category_params);
            gen(gen_model_id).data(dataset).raw = d;
            gen(gen_model_id).data(dataset).true_nll = nloglik_fcn(gen(gen_model_id).p(:,dataset), d, g, nDNoiseSets, category_params);
            gen(gen_model_id).data(dataset).true_logposterior = -gen(gen_model_id).data(dataset).true_nll + log_prior(gen(gen_model_id).p(:,dataset)');
        end
    end
end


%% OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%

start_t=tic;

for gen_model_id = active_gen_models
    if strcmp(data_type,'fake')
        fprintf('\n\n%%%%%%%%%%%% GENERATING MODEL ''%s'' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n',gen_models(gen_model_id).name);
    end
    for opt_model_id = active_opt_models
        model_start_t = tic;
        o = opt_models(opt_model_id);
        nParams = length(o.lb);

        fprintf('\n\nFITTING MODEL ''%s''\n\n', o.name)
        
        if strcmp(o.family, 'opt') && o.d_noise
            nOptimizations = nDNoiseOptimizations;
        elseif strcmp(o.family, 'MAP') && o.ori_dep_noise
            nOptimizations = nMAPOriDepNoiseOptimizations;
        else
            nOptimizations = nRegOptimizations;
        end
        
        if RaiseOptsToParams % raise number of optimizations to the power of the number of dimensions in the model
            nOptimizations = nOptimizations^nParams;
        end
        
        if strcmp(optimization_method,'mcmc_slice')
            nSamples = nOptimizations; % real samples to run
            burnin = round(nSamples/2);
            thin = ceil(nSamples/nKeptSamples);
            nOptimizations = nChains;
        end
        
        
        unfixed_params = setdiff(1:nParams, fixed_params_opt);
        o.Aeq = eye(nParams);
        o.Aeq(unfixed_params, unfixed_params) = 0;
        o.beq(unfixed_params) = 0;
        
        [extracted_p, extracted_grad] = deal(zeros(nParams, nOptimizations)); % these will be filled with each optimization and overwritten for each dataset
        extracted_nll = zeros(1, nOptimizations);
        extracted_hessian=zeros(nParams, nParams, nOptimizations);
        tmp = o;
        tmp.extracted(max(datasets)) = struct;
        gen(gen_model_id).opt(opt_model_id) = tmp;
        
        % OPTIMIZE
        for dataset = datasets;
            
            data = gen(gen_model_id).data(dataset);
            
            ex = struct;
            for trainset = 1:k % this is just 1 if not cross-validating
                %% OPTIMIZE PARAMETERS
                % random starting points x0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                x0 = random_param_generator(nOptimizations, o, 'generating_flag', x0_reasonable); % try turning on generating_flag.
                
                if crossvalidate
                    d = data.train(trainset);
                    d_test = data.test(trainset);
                else
                    d = data.raw;
                end
                
                % use anon objective function to fix data parameter.
                f = @(p) nloglik_fcn(p, d, o, nDNoiseSets, category_params);%, optimization_method, randn_samples{dataset});
                
                ex_p = nan(nParams,nOptimizations);
                ex_nll = nan(1,nOptimizations);
                ex_exitflag = nan(1, nOptimizations);
                ex_output = cell(1,nOptimizations);
                ex_lambda = cell(1,nOptimizations);
                ex_grad = nan(nParams, nOptimizations);
                ex_hessian = nan(nParams, nParams, nOptimizations);
                ex_xmin=cell(1,nOptimizations);
                ex_fmi=cell(1,nOptimizations);
                ex_ncall=nan(1,nOptimizations);
                ex_ncloc=nan(1,nOptimizations);
                if strcmp(optimization_method,'mcmc_slice')
                    ex_p = nan(nKeptSamples,nParams,nOptimizations);
                    ex_ncall = nan(1,nOptimizations);
                    [ex_nll, ex_logprior, ex_logposterior] = deal(nan(nKeptSamples,nOptimizations));
                    aborted_flags = zeros(1,nOptimizations);
                end

                loglik_wrapper = @(p) -nloglik_fcn(p, d, o, nDNoiseSets, category_params);
                logprior_wrapper = @log_prior;
                
                parfor (optimization = 1 : nOptimizations, maxWorkers)
%                     if rand < 1 / progress_report_interval % every so often, print est. time remaining.
%                         fprintf('Dataset: %.0f\nElapsed: %.1f mins\n\n', dataset, toc(start_t)/60);
%                     end
                    
                    switch optimization_method
                        case 'mcmc_slice'
                            fprintf('Dataset: %.0f\nElapsed: %.1f mins\n\n', dataset, toc(start_t)/60);
                            switch slicesampler
                                case 'builtin'
                                    [ex_p(:,:,optimization), ex_ncall(optimization)]=my_slicesample(x0(:,optimization)',nKeptSamples,'logpdf',lp_wrapper, 'width', o.ub-o.lb,'burnin',burnin,'thin',thin,'progress_report_interval',progress_report_interval,'chain_id',optimization);
                                case 'luigi'
                                    [samples, loglikes, logpriors, aborted_flags(optimization)] = slice_sample(nKeptSamples,'logdist',loglik_wrapper,'logpriordist',logprior_wrapper,'burnin',burnin,'thin',thin,'xx',x0(:,optimization)','widths',(o.ub-o.lb)','progress_report_interval',progress_report_interval,'chain_id',optimization, 'time_lim',.97*time_lim);
                                    if aborted_flags(optimization)
                                        fprintf('aborted during slice_sample, chain %i.\n',optimization)
                                    end
                                    ex_p(:,:,optimization) = samples';
                                    ex_nll(:,optimization) = -loglikes';
                                    ex_logprior(:,optimization) = logpriors';
                            end
                            %                                case 'fmincon'
                            %                                     [ex_p(:, optimization), ex_nll(optimization), ex_exitflag(optimization), ex_output{optimization}, ex_lambda{optimization}, ex_grad(:,optimization), ex_hessian(:,:,optimization)] = fmincon(f, x0(:,optimization), o.A, o.b, o.Aeq, o.beq, o.lb, o.ub, [], fmincon_opts);
                            %                                 case 'npsol'
                            %                                     Prob=ProbDef;
                            %                                     Prob.MaxCPU = maxtomlabsecs; % this doesn't work either. apparently can't limit npsol.
                            %                                     %%%% Prob.MajorIter = 200; % this
                            %                                     %%%% doesn't seem to eliminate the
                            %                                     %%%% error "too many major
                            %                                     %%%% iterations".
                            %                                     Prob.Solver.Tomlab = 'npsol';
                            %                                     [ex_p(:, optimization), ex_nll(optimization), ex_exitflag(optimization), ex_output{optimization}, ex_lambda{optimization}, ex_grad(:,optimization), ex_h] = fmincont(f, x0(:,optimization), A, b, Aeq, o.beq, lb, ub, [], [], Prob);
                            %                                     if size(ex_h)==[nParams nParams] % because it doesn't seem to want to make hessians.
                            %                                         ex_hessian(:,:,optimization) = ex_h;
                            %                                     end
                            %                                 case 'lgo'
                            %                                     Prob=ProbDef;
                            %                                     Prob.MaxCPU = maxtomlabsecs;
                            %                                     Prob.Solver.Tomlab='lgo'; %no license for 'oqnlp'. 'lgo' is supposedly good. 'npsol' is default.
                            %                                     %%this chokes up if I
                            %                                     %%include lambda in outputs. same
                            %                                     %%if I just say ~ instead of
                            %                                     %%getting a lambda.
                            %                                     [ex_p(:, optimization), ex_nll(optimization), ex_exitflag(optimization), ex_output{optimization}] = fmincont(f, x0(:,optimization), A, b, Aeq, o.beq, lb, ub, [], [], Prob);
                            %                                 case 'mcs'
                            %                                     % increase smax.
                            %                                     smax = 50*nParams+10; %number of levels. governs the relative amount of global versus local search. default 5n+10. just ran at 50. trying 5..
                            %                                     % By increasing smax, more weight is
                            %                                     % given to global search.
                            %                                     nf   = 480*nParams^2; %maximum number of function evaluations default 50n^2. try 4800n^2. was 20, with smax 50
                            %                                     stop = 60*nParams; % max number of evals with no progress being made. default 3n
                            %                                     [ex_p(:, optimization), ex_nll(optimization), ex_xmin{optimization}, ex_fmi{optimization}, ex_ncall(optimization), ex_ncloc(optimization), ex_exitflag(optimization)] = mcs('feval', f, lb', ub', 0, smax, nf, stop);
                            %                                 case 'snobfit'
                            %                                     ncall = maxsnobfcalls;   % limit on the number of function calls
                            %                                     mysnobtest
                            %
                            %                                     ex_p(:, optimization) = xbest';
                            %                                     ex_nll(optimization) = fbest;
                            %                                     ex_ncall(optimization) = ncall0;
                            %                                 case 'patternsearch'
                            %                                     [ex_p(:, optimization), ex_nll(optimization)] = patternsearch(f, x0(:,optimization), A, b, Aeq, o.beq, lb, ub, [], patternsearch_opts);
                            %                                 case 'ga'
                            %                                     [ex_p(:, optimization), ex_nll(optimization)] = ga(f, length(lb), A, b, Aeq, o.beq, lb, ub, [], ga_opts)
                    end
                end
                
                if any(aborted_flags)
                    save([savedir 'aborted/aborted_' filename])
                    return
                end
                    
                
%                 priorlikt = tic;
%                 if strcmp(optimization_method,'mcmc_slice')                    
%                     parfor (optimization = 1:nOptimizations, maxWorkers)
%                         for s = 1:nKeptSamples
%                             ex_nll(s,optimization) = f(ex_p(s,:,optimization));
%                             ex_logprior(s,optimization) = logprior_fcn(ex_p(s,:,optimization),o);
%                             ex_logposterior(s,optimization) = ex_logprior(s,optimization)-ex_nll(s,optimization);
%                         end
%                     end
%                 end
%                 
%                 fprintf('\nIt took %g minutes to re-compute sample logpriors and loglikelihoods.\n\n',round(toc(priorlikt)*100/60)/100)
                
                ex.p = ex_p;
                ex.nll = ex_nll;
                ex.exitflag = ex_exitflag;
                ex.output = ex_output;
                ex.lambda = ex_lambda;
                ex.grad = ex_grad;
                ex.hessian = ex_hessian;
                ex.xmin = ex_xmin;
                ex.fmi = ex_fmi;
                ex.ncall = ex_ncall;
                ex.ncloc = ex_ncloc;
                ex.log_prior = ex_logprior;
                
                if crossvalidate
                    ex.train(trainset).p = ex.p;
                    ex.train(trainset).nll = ex.nll;
                    ex.train(trainset).exitflag = ex.exitflag;
                    ex.train(trainset).output = ex.output;
                    ex.train(trainset).lambda = ex.lambda;
                    ex.train(trainset).grad = ex.grad;
                    ex.train(trainset).hessian = ex.hessian;
                    
                    [ex.train(trainset).min_nll, ex.train(trainset).min_idx] = min(ex.train(trainset).nll);
                    ex.train(trainset).best_params = ex.train(trainset).p(:, ex.train(trainset).min_idx);
                    
                    ex.train(trainset).test_nll = nloglik_fcn(ex.train(trainset).best_params, d_test, fitting_model, nDNoiseSets, category_params);
                end
                
            end
            
            if crossvalidate
                ex.sum_test_nll = sum([ex.train.test_nll]);
                ex.mean_test_nll= mean([ex.train.test_nll]);
            end
            
            %% COMPILE BEST EXTRACTED PARAMETERS AND SCORES AND STUFF
            if crossvalidate
                fields = fieldnames(ex);
                for field = 1 : length(fields)
                    gen(gen_model_id).opt(opt_model_id).extracted(dataset).(fields{field}) = ex.(fields{field});
                end
            else
                if strcmp(optimization_method, 'mcmc_slice')
                    %                     all_nll = vertcat(ex.nll{:});
                    all_nll = reshape(ex.nll, numel(ex.nll),1);
                    %                     all_p = vertcat(ex.p{:});
                    all_p = reshape(permute(ex.p,[1 3 2]),[],size(ex.p,2),1);
                    [ex.min_nll, ex.min_idx] = min(all_nll);
                    ex.best_params = all_p(ex.min_idx,:)';
                    ex.mean_params = mean(all_p);
                    dbar = 2*mean(all_nll);
                    dtbar= 2*f(ex.mean_params); % f is nll
                    ex.dic=2*dbar-dtbar; %DIC = 2(LL(theta_bar)-2LL_bar)
                    
                    ex.best_hessian = [];
                    ex.hessian = [];
                    ex.laplace = [];
                    ex.n_good_params = [];
                else
                    [ex.min_nll, ex.min_idx]    = min(ex.nll);
                    ex.dic = [];
                    ex.best_params          = ex.p(:, ex.min_idx);
                    ex.n_good_params                          = sum(ex.nll < ex.min_nll + nll_tolerance & ex.nll > 10);
                    paramprior      = o.param_prior;
                    ex.best_hessian = ex.hessian(:,:,ex.min_idx);
                    h               = ex.best_hessian;
                    ex.laplace = -ex.min_nll + log(paramprior) +  (nParams/2)*log(2*pi) - .5 * log(det(h));
                end
                [ex.aic, ex.bic, ex.aicc] = aicbic(-ex.min_nll, nParams, gen_nSamples);
                
                if strcmp(data_type, 'real')
                    gen(gen_model_id).opt(opt_model_id).extracted(dataset).name = data.name;
                end
                if slimdown
                    fields = {'p','nll','log_prior','hessian','min_nll','min_idx','best_params','n_good_params','aic','bic','aicc','dic','best_hessian','laplace'};
                else
                    fields = fieldnames(ex)
                end
                
                for field = 1 : length(fields)
                    gen(gen_model_id).opt(opt_model_id).extracted(dataset).(fields{field}) = ex.(fields{field});
                end
            end
            clear ex;
            save([savedir filename '~'])
        end
        
        fprintf('Total model %s time: %.1f mins\n\n', o.name, toc(model_start_t)/60);
        
    end
end

fprintf('Total optimization time: %.2f mins.\n',toc(start_t)/60);
clear ex_p ex_nll ex_logposterior ex_logprior all_nll all_p d varargin;
if hpc
    gen.data = [];
    data = [];
end

%%
hpc=0;
if ~hpc
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
            %ylim([g.lb(parameter) g.ub(parameter)]);
            axis square;
            plot([g.lb(parameter) g.ub(parameter)], [g.lb(parameter) g.ub(parameter)], '--');
            
            title(g.parameter_names{parameter});
        end
        suplabel('true parameter', 'x');
        suplabel('extracted parameter', 'y');
    elseif strcmp(optimization_method,'mcmc_slice')
        % DIAGNOSE MCMC
        % open windows for every model/model//dataset combo.
        for gen_model_id = 1:length(gen)
            g = gen_models(gen_model_id);
            if strcmp(data_type,'real')
                g.name = [];
            end
            for opt_model = active_opt_models%1:length(gen(gen_model_id).opt)
                o = gen(gen_model_id).opt(opt_model);
                for dataset_id = dataset%1:length(o.extracted)
                    [samples,logposteriors]=deal(cell(1,nChains));
                    
                    extraburn_pct=0; % burn some proportion of the samples
                    for c = 1:nChains
                        nSamples = size(o.extracted(dataset_id).p,1);
                        burn_start = max(1,round(nSamples*extraburn_pct));
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
%%
delete([savedir filename '~'])
save([savedir filename])
%%
    function lp = log_prior(x)
        lapse_params = strncmpi(o.parameter_names,'lambda',6);
        lapse_sum = sum(x(lapse_params));
        if o.multi_lapse % add lambda_1 and lambda_4 again because of the interpolation
            lapse_sum = lapse_sum + x(strcmp(o.parameter_names,'lambda_1')) + x(strcmp(o.parameter_names,'lambda_4'));
        end
        if any(x<o.lb) || any(x>o.ub) || lapse_sum > 1
            lp = -Inf;
        else
            uniform_range = o.ub(~lapse_params)-o.lb(~lapse_params);
            a=1; % beta dist params
            b=20;
            lp=sum(log(1./uniform_range))+sum(log(betapdf(x(lapse_params),a,b)));
        end
    end
end