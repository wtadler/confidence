function gen=optimize_fcn(varargin)
%%
hpc = false;
job_id = '';

% main fitting parameters
%%

opt_models = struct;

opt_models(1).family = 'fixed';
opt_models(1).multi_lapse = 0;
opt_models(1).partial_lapse = 0;
opt_models(1).repeat_lapse = 0;
opt_models(1).choice_only = 1;
opt_models(1).diff_mean_same_std = 0;
opt_models(1).ori_dep_noise = 0;

opt_models(2) = opt_models(1);
opt_models(2).family = 'lin';

opt_models(3) = opt_models(1);
opt_models(3).family = 'quad';

opt_models(4) = opt_models(1);
opt_models(4).family = 'opt';
opt_models(4).d_noise = 0;
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
nDNoiseSets = 101;

progress_report_interval = 20;

x0_reasonable = false; %limits starting points for optimization to those reasonable ranges defined in lb_gen and ub_gen

optimization_method = 'fmincon'; % 'fmincon', 'lgo','npsol','mcs','snobfit','ga', 'patternsearch', 'gridsearch','mcmc_slice'
    nGrid = 50; % for gridsearch only
    maxtomlabsecs = 100; % for lgo,npsol only
    maxsnobfcalls = 10000; % for snobfit only
    nKeptSamples = 1e4; % for mcmc_slice
    nChains = 4;
    
data_type = 'real'; % 'real' or 'fake' or 'modelfit'
% 'fake' generates trials and responses, to test parameter extraction
% 'real' takes real trials and real responses, to extract parameters
% 'modelfit' takes real trials and generates responses, to extract
% parameters again, like telephone


%% fake or modelfit data generation parameters
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

% optimization options
fmincon_opts = optimoptions(@fmincon,'Algorithm','interior-point', 'display', 'off', 'UseParallel', 'never');
%fmincon_opts = optimoptions(@fmincon,'Algorithm','interior-point','PlotFcns',{@optimplotx,@optimplotfunccount;@optimplotfval,@optimplotconstrviolation;@optimplotstepsize,@optimplotfirstorderopt});
patternsearch_opts = psoptimset('MaxIter',1e4,'MaxFunEvals',1e4);
ga_opts = gaoptimset('PlotFcns', {@gaplotbestf, @gaplotscorediversity, @gaplotbestindiv, @gaplotgenealogy, @gaplotscores, @gaplotrankhist});

assignopts(who,varargin);

if hpc
    datadir='/home/wta215/data/v2/';
else
    datadir = '/Users/will/Google Drive/Ma lab/repos/qamar confidence/data/v2/';
    %datadir = '/Users/will/Ma lab/repos/qamar confidence/data/';
end    
    
assignopts(who,varargin); % reassign datadir now if it's been specified


if strcmp(data_type, 'real')
    gen = compile_data('datadir', datadir, 'crossvalidate', crossvalidate, 'k', k);
    nDatasets = length(gen.data);
    datasets = 1 : nDatasets;
    gen_models = struct; % to make it length 1, to execute big optimization loop just once.
    active_gen_models = 1;
elseif strcmp(data_type,'fake') | strcmp(data_type,'modelfit')
    nDatasets = 6;
    assignopts(who,varargin);
    datasets = 1:nDatasets;
    extracted_param_file = '';
end

assignopts(who,varargin); % reassign datasets (for multiple jobs)

datetime_str = datetimefcn;
if hpc
    if strcmp(data_type,'real')
        if length(datasets)==1
            savedir = sprintf('/home/wta215/Analysis/output/%.f_subject%g.mat', job_id, datasets)
        else
            savedir = sprintf('/home/wta215/Analysis/output/%.f_multiplesubjects.mat', job_id)
        end
    elseif strcmp(data_type,'fake')
        savedir = sprintf('/home/wta215/Analysis/output/%.f_model%g.mat', job_id, active_gen_models);
    end
else
    savedir = sprintf('/Users/will/Google Drive/Ma lab/output/%s.mat', datetime_str);
end


%% GENERATING FAKE DATA, OR GETTING REAL TRIALS %%%%%%%%%%%%%%%

% generate d_noise samples if necessary
if any([opt_models.d_noise]) || (isfield(gen_models,'d_noise') && any([gen_models.d_noise]))
%if any(~cellfun('isempty', regexp(opt_models, 'd_noise'))) | any(~cellfun('isempty', regexp(gen_models, 'd_noise')))
    for dataset = 1 : nDatasets
        if strcmp(data_type,'real')
            if crossvalidate
                nTrials = length(gen.data(dataset).train(1).C);
            else
                nTrials = length(gen.data(dataset).raw.C);
            end
        end
    end
end

if strcmp(data_type, 'fake')
    
    for gen_model_id = active_gen_models;
        
        g = gen_models(gen_model_id);
                
        % define fixed parameters
        grid_params = [1 2];
        
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
            switch data_type
                case 'fake'
                    % generate data from parameters
                    save before.mat
                    gen(gen_model_id).data(dataset).raw = trial_generator(gen(gen_model_id).p(:,dataset), g, 'n_samples', gen_nSamples, 'dist_type', dist_type, 'contrasts', exp(-4:.5:-1.5), 'category_params', category_params);
                    gen(gen_model_id).data(dataset).true_nll = nloglik_fcn(gen(gen_model_id).p(:,dataset), gen(gen_model_id).data(dataset).raw, g, nDNoiseSets, category_params);
                    save after.mat
                case 'modelfit' % deprecated, might not work
                    streal = compile_data('datadir', datadir); % get real data
                    gen.data(dataset).raw = trial_generator(p(:,dataset), 'n_samples', gen_nSamples, 'dist_type', 'qamar', 'model_fitting_data', streal.data(dataset).raw, 'contrasts', exp(-4:.5:-1.5), 'model', g);
                    
                    true_nll(1, dataset) = nloglik_fcn(p(:, dataset), gen.data(dataset).raw, g, nDNoiseSets);
            end
        end
        gen(gen_model_id).true_nll = [gen(gen_model_id).data.true_nll];
        
    end
end


%% OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%

start_t=tic;

for gen_model_id = active_gen_models
   %gen_model=gen_models(gen_model_id);
    for opt_model_id = active_opt_models
        model_start_t = tic;
        o = opt_models(opt_model_id);
        
        fprintf('\n\nFITTING MODEL ''%s''\n\n', o.name)
        
        if strcmp(o.family, 'opt') && o.d_noise
            nOptimizations = nDNoiseOptimizations;
        elseif strcmp(o.family, 'MAP') && o.ori_dep_noise
            nOptimizations = nMAPOriDepNoiseOptimizations;
        else
            nOptimizations = nRegOptimizations;
        end
        
        if strcmp(optimization_method,'mcmc_slice')
            nSamples = nOptimizations;
            burnin = round(nSamples/2);
            thin = max(1,round(nSamples/nKeptSamples));
            nOptimizations = nChains;
        end
            
        
        nParams = length(o.lb);
        
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

            switch optimization_method
                %%
                case 'gridsearch'
                    f1 = figure(1);
                    set(f1,'position',[51 1 1390 805]);
                    subplot(3,3,dataset)
                    %if length(unfixed_params) ~= 2; error('there can only be 2 unfixed parameters for gridsearch.'); end
                    % this won't work for b2 and b3 yet
                    nll_map = zeros(nGrid,nGrid);
                    
                    param1_vec = linspace(lb(grid_params(1)), ub(grid_params(1)), nGrid);
                    param2_vec = linspace(lb(grid_params(2)), ub(grid_params(2)), nGrid);
                    
                    for i = 1:nGrid;
                        parfor j = 1:nGrid;
                            pVec = p(:,dataset);
                            pVec(grid_params(1)) = param1_vec(i);
                            pVec(grid_params(2)) = param2_vec(j);
                            nll_map(i,j) = nloglik_fcn(pVec, gen.data(dataset).raw, fitting_model);
                        end
                    end
                    
                    imagesc(param2_vec, param1_vec, nll_map)
                    colorbar
                    colormap(flipud(hot))
                    title(sprintf('\\alpha=%.1f; \\beta=%.1f; \\sigma_0=%.1f; prior=%.1f; b_1=%.1f; b_2=%.1f; b_3=%.1f; \\lambda=%.1f; \\lambda_\\gamma%.1f;', p(:,dataset)))            % find and plot extracted min
                    [minNum, minIndex] = min(nll_map(:))
                    [row,col] = ind2sub(size(nll_map), minIndex)
                    hold on
                    plot(p(grid_params(2),dataset), p(grid_params(1),dataset),'.') % true p
                    plot(param2_vec(col),param1_vec(row),'b*','markersize',10) % extracted p
                    % plot true p
                    if dataset == nDatasets
                        suplabel(parameter_names{grid_params(2)},'x')
                        suplabel(parameter_names{grid_params(1)},'y')
                        lh=legend('true', 'extracted');
                        set(lh,'position', [0.45, .955, 0.1, 0.05])
                    end
                    
                    % in 2nd figure, contrast curves
                    f2=figure(2);
                    set(f2,'position',[1441 1 1390 805]);
                    subplot(3,3,dataset);
                    x = .015:.001:.226;
                    contrasts = exp(-4:.5:-1.5);
                    h1 = plot(x,sqrt(p(3,dataset)^2 + p(1,dataset) * x .^ -p(2,dataset)),'b-');
                    hold on
                    h2 = plot(contrasts, sqrt(p(3,dataset)^2 + p(1,dataset) * contrasts .^ -p(2,dataset)),'b.','markersize',20);
                    h3 = plot(contrasts, sqrt(p(3,dataset)^2 + param1_vec(row) * contrasts .^ -param2_vec(col)),'b*','markersize',10);
                    xlim([.015 .226])
                    if dataset == nDatasets
                        suplabel('contrast','x')
                        suplabel('sigma','y')
                        lh=legend([h2 h3],{'true', 'extracted'});
                        set(lh,'position', [0.45, .955, 0.1, 0.05])
                    end
                    
                    
                otherwise
                    ex = struct;
                    for trainset = 1:k % this is just 1 if not cross-validating
                        %% OPTIMIZE PARAMETERS
                        % random starting points x0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        x0 = random_param_generator(nOptimizations, o, 'generating_flag', x0_reasonable); % try turning on generating_flag.
                        
                        %if ~isempty(fixed_params) % unnecessary, I think...
                        %    x0(fixed_params,:) = repmat(o.beq(fixed_params),1,nOptimizations);
                        %end
                    
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
                            ex_nll = nan(nKeptSamples,nOptimizations);
                        end
                        ww = @wrapper;
                        
                        parfor optimization = 1 : nOptimizations;
                            if rand < 1 / progress_report_interval % every so often, print est. time remaining.
                                fprintf('Dataset: %.0f\nElapsed: %.1f mins\n\n', dataset, toc(start_t)/60);
                            end
                            
                            switch optimization_method
                                case 'mcmc_slice'
                                    [ex_p(:,:,optimization), ex_ncall(optimization)]=slicesample(x0(:,optimization)',nKeptSamples,'logpdf',ww, 'width', o.ub-o.lb,'burnin',burnin,'thin',thin);
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
                        if strcmp(optimization_method,'mcmc_slice')
                            parfor optimization = 1:nOptimizations
                                for s = 1:nKeptSamples
                                    ex_nll(s,optimization) = -ww(ex_p(s,:,optimization));
                                end
                            end
                        end
                        
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
            end
            
            %% COMPILE BEST EXTRACTED PARAMETERS AND SCORES AND STUFF
            %datasets=1;
            %for gen_model_id = 1
            %   for opt_model_id = 1:3;
            %  nParams = size(gen.opt(opt_model_id).extracted(1).p,1);
            % gen_nSamples = 3240
            %%
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
                    ex.dic=2*dbar-dtbar;
                    
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
                    fields = {'p','nll','hessian','min_nll','min_idx','best_params','n_good_params','aic','bic','aicc','dic','best_hessian','laplace'};
                else
                    fields = fieldnames(ex)
                end

                for field = 1 : length(fields)
                    gen(gen_model_id).opt(opt_model_id).extracted(dataset).(fields{field}) = ex.(fields{field});
                end
            end
            clear ex;
            save([savedir '~'])
        end

        fprintf('Total model %s time: %.1f mins\n\n', o.name, toc(model_start_t)/60);

    end
end

fprintf('Total optimization time: %.2f mins.\n',toc(start_t)/60);

%% COMPARE TRUE AND REAL PARAMETERS IN SUBPLOTS
if ~hpc
    if ~strcmp(optimization_method,'mcmc_slice') && strcmp(data_type,'fake') && length(active_opt_models)==1 && length(active_gen_models) == 1 && strcmp(opt_models(active_opt_models).name, gen_models(active_gen_models).name)
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
        % open windows for every model/dataset combo.
        for gen_model = 1:length(gen)
            g = gen(gen_model);
            for opt_model = 1:length(g.opt)
                o = g.opt(opt_model);
                for dataset = 1:length(o.extracted)
                    d = g.data(dataset).raw;
                    ww = @wrapper;
                    samples=cell(1,nChains);
                    logliks=cell(1,nChains);
                    for c = 1:nChains
                        samples{c} = o.extracted(dataset).p(:,:,c);
                        logliks{c} = nan(nKeptSamples,1);
                        for s=1:nSamples
                            logliks{c}(s) = ww(samples{c}(s,:));
                        end
                    end
                    
                    [true_p,true_ll]=deal([]);
                    if strcmp(data_type,'fake') && length(active_opt_models)==1 && length(active_gen_models) == 1 && strcmp(opt_models(active_opt_models).name, gen_models(active_gen_models).name)
                        true_p = g.p(:,dataset);
                        true_ll = -g.true_nll(dataset);
                    end
                    [fh,ah]=mcmcdiagnosis(samples,logliks,o,true_p,true_ll,dataset);
                     
                end
            end
        end
    end
end


%% NEED PRIORS FOR MCMC SLICE.
%%
delete([savedir '~'])
save(savedir)

function ll = wrapper(x)
if any(x<o.lb) || any(x>o.ub) || any(o.A*x' > o.b)
    ll = -Inf;
else
    ll = -nloglik_fcn(x, d, o);
end
end
end