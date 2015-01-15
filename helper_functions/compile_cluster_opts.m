function compile_cluster_opts(varargin)
% compile optimizations from the cluster
%clear all

datadir='/Users/will/Google Drive/Ma lab/output';
jobid = 'newest';
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
job_files = mat_files(~cellfun(@isempty,regexp(mat_files,sprintf('^%i.*\\.mat', jobid))));
%job_files = mat_files;
load(job_files{1}) % load first file to get nDatasets
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
% nDatasets
%
% for file = 1:length(job_files)
%     tmp=load(job_files{file});
%     file
%     dataset= tmp.dataset
%     tmp.model = tmp.gen.opt;
%     for model_id = 1:length(tmp.model)
%         [~,idx]=ismember(tmp.model(model_id).name, {model.name})
%         if idx==0
%             model(length(model)+1).name = tmp.model(model_id).name;
%             [~,idx]=ismember(tmp.model(model_id).name, {model.name})
%         end
%
%         if crossvalidate % shouldn't have to do this except this one time.
%             for k = 1 : 10;
%                 tmp.model(model_id).extracted(dataset).train(k).test_nll = nloglik_fcn(tmp.model(model_id).extracted(dataset).train(k).best_params, tmp.data.test(k), tmp.opt_models{model_id}, randn(1000, 324));
%             end
%             tmp.model(model_id).extracted(dataset).sum_test_nll = sum([tmp.model(model_id).extracted(dataset).train.test_nll]);
%             tmp.model(model_id).extracted(dataset).mean_test_nll = mean([tmp.model(model_id).extracted(dataset).train.test_nll]);
%         end
%
%         for field = 1 : nFields
%             model(idx).extracted(dataset).(fields{field}) = [model(idx).extracted(dataset).(fields{field}) tmp.model(model_id).extracted(dataset).(fields{field})];
%         end
%     end
% end
% return

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
                    %save ccotest.mat
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
            model(model_id).extracted(dataset).best_params = model(model_id).extracted(dataset).p(:, model(model_id).extracted(dataset).min_idx);
            nParams = length(model(model_id).extracted(dataset).best_params);
            
            model(model_id).best_params(:, dataset) = model(model_id).extracted(dataset).best_params;
            model(model_id).extracted(dataset).n_good_params = sum(model(model_id).extracted(dataset).nll < model(model_id).extracted(dataset).min_nll + nll_tolerance & model(model_id).extracted(dataset).nll > 1000);
            [model(model_id).extracted(dataset).aic, model(model_id).extracted(dataset).bic, model(model_id).extracted(dataset).aicc] = aicbic(-model(model_id).extracted(dataset).min_nll, nParams, nSamples);
            
            model(model_id).extracted(dataset).best_hessian = model(model_id).extracted(dataset).hessian(:,:,model(model_id).extracted(dataset).min_idx);
            model(model_id).extracted(dataset).hessian=[]; % clear hessian matrix after finding the best one. too big.
            param_prior = model(model_id).param_prior;
            h = model(model_id).extracted(dataset).best_hessian;
            model(model_id).extracted(dataset).laplace = -model(model_id).extracted(dataset).min_nll + log(param_prior) +  (nParams/2)*log(2*pi) - .5 * log(det(h));
            
            model(model_id).extracted(dataset).name = tmp.model(model_id).extracted(dataset).name; % otherwise you have a repeating name, eg 'wtawtawta'
        end
    end
    toc
end
gen.opt=[];
clear tmp ex_output ex_lambda ex_hessian extracted_hessian
%cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations')
save(sprintf('COMBINED_%i.mat', jobid))%,'-v7.3')
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




%%

% model recovery

% compile optimizations from the cluster
clear all

datadir='/Users/will/Google Drive/Ma lab/output';
cd(datadir)
jobid = 2020218;

% load all complete sessions
files = what(datadir);
mat_files = files.mat;
job_files = mat_files(~cellfun(@isempty,regexp(mat_files,sprintf('^%i.*\\.mat', jobid))));
%job_files = mat_files;
load(job_files{1}) % load first file to get nDatasets
nGenModels = length(gen_models);
nOptModels = length(gen(1).opt);

for model_id = 2 :  nGenModels
    tmp = load(job_files{model_id});
    gen(model_id) = tmp.gen(model_id);
end
%%
heatmap = zeros(nOptModels, nGenModels);
for gen_model = 1 :nGenModels
    for opt_model = 1 : nOptModels
        heatmap(opt_model, gen_model) = -mean(real([gen(gen_model).opt(opt_model).extracted.laplace]));
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