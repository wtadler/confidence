function compile_cluster_opts(varargin)
% compile optimizations from the cluster
%clear all

datadir='/Users/will/Google Drive/Ma lab/output/v3_joint_feb28';
rawdatadir='/Users/will/Google Drive/Will - Confidence/Data/v3/taskA/'; % for computing DIC
rawdatadirB='/Users/will/Google Drive/Will - Confidence/Data/v3/taskB/'; % for computing DIC

toss_bad_samples = true;
extraburn_prop = 0;

jobid = 'ab';
require_exact_name = false;
hpc = true;

include_aborted_jobs = false;
assignopts(who,varargin)

st = compile_data('datadir',rawdatadir);
stB = compile_data('datadir',rawdatadirB);

if strcmp(jobid,'newest')
    D = dir([datadir '/*.mat']);
    [~,idx]=sort([D.datenum]); % idx of mat files by date. last one is newest
    jobid = str2num(D(idx(end)).name(1:7));
end
jobid

% load all complete sessions
cd(datadir)
files = what(datadir);
mat_files = files.mat;
if require_exact_name
    job_files = mat_files(~cellfun(@isempty,regexp(mat_files,sprintf('^%s\\..*\\.mat', jobid))));
else
    job_files = mat_files(~cellfun(@isempty,regexp(mat_files,sprintf('^%s.*\\.mat', jobid))));
end

% load aborted chains
if include_aborted_jobs
    aborted_files = what([datadir '/aborted']);
    aborted_mat_files = aborted_files.mat;
    if require_exact_name
        aborted_job_files = aborted_mat_files(~cellfun(@isempty, regexp(aborted_mat_files, sprintf('^aborted_%s\\..*\\.mat', jobid))));
    else
        aborted_job_files = aborted_mat_files(~cellfun(@isempty, regexp(aborted_mat_files, sprintf('^aborted_%s.*\\.mat', jobid))));
    end
    for f = 1:length(aborted_job_files);
        aborted_file = regexp(aborted_job_files{f}, '^aborted_(.*)', 'tokens'); % a{1}{1} is a filename string without ('aborted_')
        aborted_file = aborted_file{1}{1};
        if ~any(cellfun(@(s) ~isempty(strfind(aborted_file, s)), job_files)) % if aborted_file is not already in job_files
            job_files = [job_files;strcat('aborted/aborted_', aborted_file)];
        end
    end
        
    
%     job_files = [job_files;strcat('aborted/', aborted_job_files)];
end
        
newjob = zeros(1,length(job_files));
for j = 1:length(job_files)
    t = regexp(job_files{j},'\.(.*?)\.','tokens');
    newjob(j) = str2num(t{1}{1});
end
[~,sort_idx]=sort(newjob);

job_files = job_files(sort_idx)

load(job_files{1}) % load first file to get nDatasets

if any(strfind(job_files{1},'model')) && any(regexp(job_files{1},'m[0-9]*.'))% new model recovery
    
    for fid = 2:length(job_files);
        tmp = load(job_files{fid});
        m = tmp.active_opt_models;
        g = tmp.active_gen_models;
        
        if m==1
            gen(g)=tmp.gen(g);
        else
            gen(g).opt(m)=tmp.gen(g).opt(m);
        end
    end
    
elseif strfind(job_files{1},'model') % old model recovery
    
        if isempty(model(m).extracted(d).name) % initialize subject details if they are not there
            model(m).extracted(d) = tmp.gen.opt(m).extracted(d);
        else % go through and append the different chain data
            for f = 1:length(e_fields)
                if strcmp(optimization_method,'mcmc_slice')
                    if strcmp(e_fields{f},'p')
                        model(m).extracted(d).(e_fields{f}) = cat(3,model(m).extracted(d).(e_fields{f}),tmp.gen.opt(m).extracted(d).(e_fields{f}));
                    else
                        model(m).extracted(d).(e_fields{f}) = cat(2,model(m).extracted(d).(e_fields{f}),tmp.gen.opt(m).extracted(d).(e_fields{f}));
                    end
                else
                    model(m).extracted(d).(e_fields{f}) = cat(2,model(m).extracted(d).(e_fields{f}),tmp.gen.opt(m).extracted(d).(e_fields{f}));
                end

            end
        
        end
        
    
    %     nGenModels = length(gen_models);
%     nOptModels = length(gen(1).opt);
%     
%     for model_id = 2 :  nGenModels
%         tmp = load(job_files{model_id});
%         gen(model_id) = tmp.gen(model_id);
%     end
    
    if ~hpc
        %% change this, make it more adaptable. not appropriate for this script anyway.
        heatmap = zeros(nOptModels, nGenModels);
        for gen_model = 1 :nGenModels
            for opt_model = 1 : nOptModels
                heatmap(opt_model, gen_model) = mean(real([gen(gen_model).opt(opt_model).extracted.aic]));
            end
            heatmap(:, gen_model) = heatmap(:, gen_model) - min(heatmap(:, gen_model));
        end
        model_names = {'Bayes','Fixed','Lin','Quad','MAP_s'};
        im=imagesc(-heatmap);
        colormap(bone(256));
        caxis([-50 0]);
        cb=colorbar;
        yh=get(cb,'ylabel');
        set(cb,'ticklength',0);
        set(yh,'String','AIC - AIC_b_e_s_t','rot',-90)
        set(gca,'xtick', 1:nGenModels, 'ytick', 1:nOptModels, 'yticklabel', model_names,'xticklabel',model_names,'fontsize',12,'ticklength',[0 0])
        xlabel('Generating model')
        ylabel('Fitting model')
        pause(eps) % can't get yh position unless we wait. wtf matlab
        set(yh,'position',get(yh,'position')+[.5 0 0])
        export_fig('model_recovery_subject_params.png','-m2')
    end
    
elseif any(regexp(job_files{1},'m[0-9]*.s[0-9]*.c[0-9]*.mat')) % indicates single chain real data
    tmp=load(job_files{1});
    
%     if exist('ex_p','var') % indicates aborted file
%         end_early_routine
%     end
    
    nModels = length(tmp.opt_models);
    
    % initialize empty struct of the right size.
    model = tmp.gen.opt;
    m_fields = setdiff(fieldnames(model),'extracted');
    for f = 1:length(m_fields)
        model(tmp.active_opt_models).(m_fields{f}) = [];
    end
    
    if isfield(tmp, 'ex_p') % if file is an aborted file
        tmp = end_early_routine(tmp);
        e_fields = fieldnames(tmp.gen.opt(tmp.active_opt_models).extracted(tmp.dataset));
    else
        e_fields = fieldnames(model(active_opt_models).extracted);
    end
    
    for d = 1:nDatasets
        for f = 1:length(e_fields)
            model(active_opt_models).extracted(d).(e_fields{f}) = [];
        end
    end
    e_fields = setdiff(e_fields,'name'); % don't want to append names later
    for d = 1:nDatasets
        model(tmp.active_opt_models).extracted(d) = model(tmp.active_opt_models).extracted(1);
    end
    for m = 1:nModels
        model(m) = model(tmp.active_opt_models);
    end
    
    
    for fid = 1:length(job_files);
        fid
        tmp = load(job_files{fid});
        
        if isfield(tmp,'ex_p')
            tmp = end_early_routine(tmp);
            e_fields = fieldnames(tmp.gen.opt(tmp.active_opt_models).extracted(tmp.dataset));
            e_fields = setdiff(e_fields,'name'); % don't want to append names later
        end
        m = tmp.active_opt_models;
        d = tmp.dataset;
        
%         e_fields = fieldnames(tmp.gen.opt(m).extracted(d));
%         e_fields = setdiff(e_fields,'name'); % don't want to append names later

        
        if isempty(model(m).name) % initialize model details if they are not there
            for f = 1:length(m_fields)
                model(m).(m_fields{f}) = tmp.gen.opt(m).(m_fields{f});
            end
        end
        
        if length(model(m).extracted) < d || isempty(model(m).extracted(d).name) % initialize subject details if they are not there
            %             model(m).extracted(d).dic = [];
            %             model(m).extracted(d) = tmp.gen.opt(m).extracted(d);
            model(m).extracted(d).name = tmp.gen.opt(m).extracted(d).name;
            for f = 1:length(e_fields)
                    model(m).extracted(d).(e_fields{f}) = {tmp.gen.opt(m).extracted(d).(e_fields{f})};
            end
        else % go through and append the different chain data
            for f = 1:length(e_fields)
                    model(m).extracted(d).(e_fields{f}) = cat(2,model(m).extracted(d).(e_fields{f}),tmp.gen.opt(m).extracted(d).(e_fields{f}));
            end
            
        end
        
    end
    
    if strcmp(optimization_method,'mcmc_slice')
        for m = 1:length(model)
            for d = 1:length(model(m).extracted);
                ex = model(m).extracted(d);
                if ~isempty(ex.p) % if there's data here
                    nChains = length(ex.nll);
                    for c = 1:nChains
                        nSamples = length(ex.nll{c});
                        burn_start = max(1,round(nSamples*extraburn_prop));
                        ex.p{c} = ex.p{c}(burn_start:end,:);
                        ex.nll{c} = ex.nll{c}(burn_start:end,:);
                        ex.logprior{c} = ex.logprior{c}(burn_start:end,:);
                        ex.logposterior{c} = -ex.nll{c} + ex.logprior{c};
                        
%                         logposterior{c} = -o.extracted(d).nll(burn_start:end,c) + o.extracted(d).logprior(burn_start:end,c);
                    end
                    
                    all_samples = [];
                    all_nll = [];

                    threshold = 40;
                    max_logpost = max(cellfun(@max, ex.logposterior));
                    for c = 1:nChains
                        if toss_bad_samples
                            keepers = ex.logposterior{c} > max_logpost-threshold;
                            ex.p{c} = ex.p{c}(keepers,:);
                            ex.nll{c} = ex.nll{c}(keepers);
                            ex.logprior{c} = ex.logprior{c}(keepers);
                            ex.logposterior{c} = ex.logposterior{c}(keepers);
                        end
                        all_samples = cat(1,all_samples,ex.p{c});
                        all_nll = cat(1,all_nll,ex.nll{c});
                    end

%                     ex.mean_params = mean(all_samples)';
%                     dbar = 2*mean(all_nll);
                    
                    if ~model(m).joint_task_fit
                        loglik_fcn = @(params) -nloglik_fcn(params, st.data(d).raw, model(m), tmp.nDNoiseSets, tmp.category_params);

%                         dtbar= 2*nloglik_fcn(ex.mean_params, st.data(d).raw, model(m), tmp.nDNoiseSets, tmp.category_params);
                        
                    elseif model(m).joint_task_fit
                        sm=prepare_submodels(model(m));
                        loglik_fcn = @(params) two_task_ll_wrapper(params, st.data(d).raw, stB.data(d).raw, sm, nDNoiseSets, category_params, true);
                        
%                         dtbar=-2*two_task_ll_wrapper(ex.mean_params, st.data(d).raw, stB.data(d).raw, sm, nDNoiseSets, category_params, true);
                    end
                    
%                     old_dic=2*dbar-dtbar; %DIC = 2(LL(theta_bar)-2LL_bar)
                    
                    [ex.dic, ex.dbar, ex.dtbar] = dic(all_samples, -all_nll, loglik_fcn);
                    
%                     if abs(old_dic-ex.dic) > .1
%                         error('old and new DIC methods don''t seem to be equivalent.')
%                     else
%                         fprintf(['success!  ' num2str(old_dic) ' = ' num2str(ex.dic) '\n'])
%                     end
                    
                    
                    
                    [~,chain_idx] = min([ex.min_nll{:}]);
                    fields = {'min_nll','aic','bic','aicc','best_params'};
                    for f = 1:length(fields)
                        ex.(fields{f}) = ex.(fields{f}){chain_idx};
                    end
                    
                    fields = fieldnames(ex);
                    for f = 1:length(fields)
                        model(m).extracted(d).(fields{f}) = ex.(fields{f});
                    end
                end
            end
        end
    end
    
        
else %real data, old style
    
    %job_files = mat_files;
    load(job_files{1}) % load first file to get nDatasets
    
    if strfind(optimization_method,'mcmc')
        model = gen.opt;
        for fid = 2:length(job_files);
            tmp = load(job_files{fid});
            cur_model = tmp.active_opt_models;
            if length(model)<cur_model || ~isfield(model(cur_model), 'name') || isempty(model(cur_model).name)
                if isfield(tmp.gen.opt,'param_prior')
                    tmp.gen.opt=rmfield(tmp.gen.opt,'param_prior');
                end
                model(cur_model) = tmp.gen.opt(cur_model);
            end
            model(cur_model).extracted(tmp.dataset) = tmp.gen.opt(cur_model).extracted(tmp.dataset);
        end
    else
        
        
        
        %nDatasets=7;
        for dataset = 1 : nDatasets
            subject_files{dataset} = job_files(~cellfun(@isempty, regexp(job_files, sprintf('subject%i\\.mat$', dataset))));
        end
        
        
        load(subject_files{length(subject_files)}{1}) % load the first file for the last subject, to get the full structure shape
        model = gen.opt;
        
        
        fields = {'p','nll','hessian'};%,'exitflag','output','lambda','grad'};%, 'sum_test_nll'};
        nFields = length(fields);
        
        nModels = length(model);
        % clear data from loaded file
        for model_id = 1 : nModels
            for field = 1 : nFields
                model(model_id).extracted(length(subject_files)).(fields{field})=[];
            end
        end
        
        
        for dataset = 1 : nDatasets
            tic
            dataset
            for model_id = 1 : nModels
                model_id;
                for fid = 1 : length(subject_files{dataset})
                    %run through all the files, and keep tacking on fields.
                    tmp = load(subject_files{dataset}{fid});
                    tmp.model = tmp.gen.opt;
                    
                    if crossvalidate % shouldn't have to do this except this one time.
                        for k = 1 : 10;
                            if any(regexp(tmp.opt_models{model_id},'noise'))
                                tmp.model(model_id).extracted(dataset).train(k).test_nll = nloglik_fcn(tmp.model(model_id).extracted(dataset).train(k).best_params, tmp.data.test(k), tmp.opt_models{model_id}, randn(1000, 324));
                            else
                                tmp.model(model_id).extracted(dataset).train(k).test_nll = nloglik_fcn(tmp.model(model_id).extracted(dataset).train(k).best_params, tmp.data.test(k), tmp.opt_models{model_id});
                            end
                        end
                        tmp.model(model_id).extracted(dataset).sum_test_nll = sum([tmp.model(model_id).extracted(dataset).train.test_nll]);
                        tmp.model(model_id).extracted(dataset).mean_test_nll = mean([tmp.model(model_id).extracted(dataset).train.test_nll]);
                    end
                    for field = 1 : nFields
                        dim = ndims(tmp.model(model_id).extracted(dataset).(fields{field}));
                        model(model_id).extracted(dataset).(fields{field}) = cat(dim, model(model_id).extracted(dataset).(fields{field}), tmp.model(model_id).extracted(dataset).(fields{field}));
                    end
                end
                if crossvalidate
                    nTrainSamples = 2916;%length(tmp.data(dataset).train(1).C);
                    nTestSamples = 324;%length(tmp.data(dataset).test(1).C);
                    dataset
                    [model(model_id).extracted(dataset).min_sum_test_nll, model(model_id).extracted(dataset).min_idx] = min(model(model_id).extracted(dataset).sum_test_nll);
                    model(model_id).extracted(dataset).min_sum_test_nll
                else
                    nSamples = length(tmp.data.raw.C);
                    [model(model_id).extracted(dataset).min_nll, model(model_id).extracted(dataset).min_idx] = min(model(model_id).extracted(dataset).nll(model(model_id).extracted(dataset).nll > 10));
                    if strfind(optimization_method,'mcmc')
                        all_p = reshape(permute(model(model_id).extracted(dataset).p,[1 3 2]),[],size(model(model_id).extracted(dataset).p,2),1);
                        model(model_id).extracted(dataset).best_params = all_p(model(model_id).extracted(dataset).min_idx,:)';
                        model(model_id).extracted(dataset).dic = tmp.model(model_id).extracted(dataset).dic;
                    else
                        model(model_id).extracted(dataset).best_params = model(model_id).extracted(dataset).p(:, model(model_id).extracted(dataset).min_idx);
                        model(model_id).extracted(dataset).best_hessian = model(model_id).extracted(dataset).hessian(:,:,model(model_id).extracted(dataset).min_idx);
                        model(model_id).extracted(dataset).hessian=[]; % clear hessian matrix after finding the best one. too big.
                        param_prior = model(model_id).param_prior;
                        h = model(model_id).extracted(dataset).best_hessian;
                        model(model_id).extracted(dataset).laplace = -model(model_id).extracted(dataset).min_nll + log(param_prior) +  (nParams/2)*log(2*pi) - .5 * log(det(h));
                    end
                    nParams = length(model(model_id).extracted(dataset).best_params);
                    
                    %model(model_id).best_params(:, dataset) = model(model_id).extracted(dataset).best_params;
                    %model(model_id).extracted(dataset).n_good_params = sum(model(model_id).extracted(dataset).nll < model(model_id).extracted(dataset).min_nll + nll_tolerance & model(model_id).extracted(dataset).nll > 1000);
                    [model(model_id).extracted(dataset).aic, model(model_id).extracted(dataset).bic, model(model_id).extracted(dataset).aicc] = aicbic(-model(model_id).extracted(dataset).min_nll, nParams, nSamples);
                    
                    model(model_id).extracted(dataset).name = tmp.model(model_id).extracted(dataset).name; % otherwise you have a repeating name, eg 'wtawtawta'
                end
            end
            toc
        end
        clear ex_output ex_lambda ex_hessian extracted_hessian
    end
    gen.opt=[];
    %cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations')
end

clear tmp
save(sprintf('COMBINED_%s.mat', jobid))%,'-v7.3')

return

%% miniaturize, and/or clean up
%modelsmall=struct;
%fields = {'name','min_nll','best_params','aic','bic','aicc','best_hessian','laplace','n_good_params','min_idx'};
c=parameter_constraints;
for m=1:5
    %modelsmall(m).best_params = model(m).best_params;
    for dataset=1:11
        nParams = length(mtmp(m).extracted(dataset).best_params);
        nSamples = 3240;
        [mtmp(m).extracted(dataset).aic, mtmp(m).extracted(dataset).bic, mtmp(m).extracted(dataset).aicc] = aicbic(-mtmp(m).extracted(dataset).min_nll, nParams, nSamples);
        param_prior = c.(mtmp(m).name).param_prior;
        h = mtmp(m).extracted(dataset).best_hessian;
        mtmp(m).extracted(dataset).laplace = -mtmp(m).extracted(dataset).min_nll + log(param_prior) +  (nParams/2)*log(2*pi) - .5 * log(det(h));
        
        
        %for f=1:length(fields)
        %    modelsmall(m).extracted(dataset).(fields{f}) = model(m).extracted(dataset).(fields{f});
        %end
    end
end