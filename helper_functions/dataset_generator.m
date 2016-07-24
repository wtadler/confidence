function fake_datasets = dataset_generator(model, param_samples, nSamples, varargin)

% this makes a bunch of fake datasets for each model/subject combination, and summarizes them
% the tasks, tasks_in thing here is weird. figure it out. 9/23/15. commented out, 5/28/16

nBins = 13;
raw = [];
% tasks_in = [];
dep_vars = {'resp','g','Chat'};
symmetrify = false;
bin_types = {'c_s'};
attention_manipulation = false;
trial_types = {'all'};
analyze = true;
assignopts(who,varargin);

[tasks, modelstruct, param_idx] = submodels_for_analysis(model);
% 
% if length(tasks_in) == 1 && length(tasks) == 2
%     if strcmp(tasks_in{1}, 'A')
%         i = 1;
%     elseif strcmp(tasks_in{1}, 'B')
%         i = 2;
%     end
%     tasks = tasks(i);
%     modelstruct = modelstruct(i);
%     param_idx = param_idx(i,:);
% end

if iscell(param_samples)
    param_samples = vertcat(param_samples{:});
end

if ~exist('nSamples', 'var') %|| nSamples > size(param_samples, 1);
    nSamples = size(param_samples, 1);
%     sample_ids = 1:nSamples;
% else
end
    sample_ids = randsample(size(param_samples,1), nSamples, true);
% end


fake_datasets = struct;
for task = 1:length(tasks)
    bins = bin_generator(nBins, 'task', tasks{task});
    
    for s = 1:nSamples
        fake_datasets.(tasks{task}).dataset(s).joint_p = param_samples(sample_ids(s), :)';
        fake_datasets.(tasks{task}).dataset(s).p = param_samples(sample_ids(s), param_idx(task,:))';
        if ~isempty(raw) % if not providing real trials
            fake_datasets.(tasks{task}).dataset(s).raw = trial_generator(fake_datasets.(tasks{task}).dataset(s).p, modelstruct(task), 'model_fitting_data', raw.(tasks{task}), 'attention_manipulation', attention_manipulation);
        else
            fake_datasets.(tasks{task}).dataset(s).raw = trial_generator(fake_datasets.(tasks{task}).dataset(s).p, modelstruct(task), 'attention_manipulation', attention_manipulation);
        end
        if symmetrify && strcmp(tasks{task}, 'B')
            fake_datasets.(tasks{task}).dataset(s).raw.s = abs(fake_datasets.(tasks{task}).dataset(s).raw.s);
        end
        if analyze
            fake_datasets.(tasks{task}).dataset(s).stats = indiv_analysis_fcn(fake_datasets.(tasks{task}).dataset(s).raw, bins, 'output_fields', dep_vars, 'bin_types', bin_types, 'trial_types', trial_types);
        end
    end
    if analyze
        fake_datasets.(tasks{task}).sumstats = sumstats_fcn(fake_datasets.(tasks{task}).dataset, 'fields', dep_vars, 'bootstrap', false);
    end
end