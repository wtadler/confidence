function compile_cluster_opts(varargin)
% compile optimizations from the cluster
%clear all

datadir='/Users/will/Desktop/v3_B_fmincon_feb21';
jobid = 'v3_B_';
hpc = true;
assignopts(who,varargin)

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
job_files = mat_files(~cellfun(@isempty,regexp(mat_files,sprintf('^%s.*\\.mat', jobid))));

newjob = zeros(1,length(job_files));
for j = 1:length(job_files)
    t = regexp(job_files{j},'\.(.*?)\.','tokens');
    newjob(j) = str2num(t{1}{1});
end
[~,sort_idx]=sort(newjob);

job_files = job_files(sort_idx)

%%
if strfind(job_files{1},'model') % model recovery
    
    load(job_files{1}) % load first file to get nDatasets
    nGenModels = length(gen_models);
    nOptModels = length(gen(1).opt);
    
    for model_id = 2 :  nGenModels
        tmp = load(job_files{model_id});
        gen(model_id) = tmp.gen(model_id);
    end
    
    if ~hpc
        %% change this, make it more adaptable. not appropriate for this script anyway.
        heatmap = zeros(nOptModels, nGenModels);
        for gen_model = 1 :nGenModels
            for opt_model = 1 : nOptModels
                heatmap(opt_model, gen_model) = -mean(real([gen(gen_model).opt(opt_model).extracted.dic]));
            end
            heatmap(:, gen_model) = heatmap(:, gen_model) - min(heatmap(:, gen_model));
        end
        model_names = {'Fixed','Lin','Quad','Bayesian'};
        im=imagesc(-heatmap);
        colormap(bone(256));
        caxis([-100 0]);
        cb=colorbar;
        yh=get(cb,'ylabel');
        set(cb,'ticklength',0);
        set(yh,'String','\DeltaLaplace Approx.','rot',-90)
        set(gca,'xtick', 1:nGenModels, 'ytick', 1:nOptModels, 'yticklabel', model_names,'xticklabel',model_names,'fontsize',16,'ticklength',[0 0])
        xlabel('Generating model')
        ylabel('Fitted model')
        export_fig('model_recovery_subject_params.png','-m2')
    end
    
elseif any(regexp(job_files{1},'m[0-9]*.s[0-9]*.c[0-9]*.mat')) % indicates single chain data
    load(job_files{1}) % to get nDatasets
    nModels = length(opt_models);
    
    % initialize empty struct of the right size.
    model = gen.opt;
    m_fields = setdiff(fieldnames(model),'extracted')
    for f = 1:length(m_fields)
        model.(m_fields{f}) = [];
    end
    e_fields = fieldnames(model.extracted);
    for f = 1:length(e_fields)
        model.extracted.(e_fields{f}) = [];
    end
    e_fields = setdiff(e_fields,'name'); % don't want to append names later
    for d = 1:nDatasets
        model(1).extracted(d) = model(1).extracted(1);
    end
    for m = 1:nModels
        model(m) = model(1);
    end
    
    for fid = 1:length(job_files);
        tmp = load(job_files{fid});
        m = tmp.active_opt_models;
        d = tmp.dataset;
        
        if isempty(model(m).name) % initialize model details if they are not there
            for f = 1:length(m_fields)
                model(m).(m_fields{f}) = tmp.gen.opt(m).(m_fields{f});
            end
        end
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
        
    end
    if strcmp(optimization_method,'mcmc_slice')
        datadir='/Users/will/Google Drive/Will - Confidence/Data/v3/taskA/';
        st = compile_data('datadir',datadir);
        for m = 1:length(model)
            for d = 1:4%length(model(m).extracted);
                ex = model(m).extracted(d);
                all_p = reshape(permute(ex.p,[1 3 2]),[],size(ex.p,2),1);
                mean_params = mean(all_p)';
                dbar = 2*mean(ex.nll(:));
                dtbar= 2*nloglik_fcn(mean_params, st.data(d).raw, model(m), tmp.nDNoiseSets, tmp.category_params);
                ex.dic=2*dbar-dtbar; %DIC = 2(LL(theta_bar)-2LL_bar)
                
                [~,ex.min_idx] = min(ex.nll(:));
                [~,chain_idx]=ind2sub(size(ex.nll),ex.min_idx);
                fields = {'min_nll','aic','bic','aicc','best_params'};
                for f = 1:length(fields)
                    ex.(fields{f}) = ex.(fields{f})(:,chain_idx);
                end
                model(m).extracted(d) = ex;
                model(m).extracted(d).mean_params = mean_params;
            end
        end
    end
            
        
else %real data
    
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