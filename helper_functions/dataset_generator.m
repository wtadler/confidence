function fake_datasets = dataset_generator(model, param_samples, nSamples, varargin)

% this makes a bunch of fake datasets for each model/subject combination, and summarizes them
% the tasks, tasks_in thing here is weird. figure it out. 9/23/15.

raw_A = [];
raw_B = [];
nBins = 13;
raw = [];
tasks_in = [];
dep_vars = {'resp','g','Chat'};
assignopts(who,varargin);

[tasks, modelstruct, param_idx] = submodels_for_analysis(model);

if length(tasks_in) == 1 && length(tasks) == 2
    if strcmp(tasks_in{1}, 'A')
        i = 1;
    elseif strcmp(tasks_in{1}, 'B')
        i = 2;
    end
    tasks = tasks(i);
    modelstruct = modelstruct(i);
    param_idx = param_idx(i,:);
end

if iscell(param_samples)
    param_samples = vertcat(param_samples{:});
end

if ~exist('nSamples', 'var') || nSamples > size(param_samples, 1);
    nSamples = size(param_samples, 1);
    sample_ids = 1:nSamples;
else
    sample_ids = randsample(size(param_samples,1), nSamples);
end


fake_datasets = struct;
for task = 1:length(tasks)
    bins = bin_generator(nBins, 'task', tasks{task});
    
    for s = 1:nSamples
        fake_datasets.(tasks{task}).dataset(s).p = param_samples(sample_ids(s), param_idx(task,:))';
        if ~isempty(raw) % if not providing real trials
            fake_datasets.(tasks{task}).dataset(s).raw = trial_generator(fake_datasets.(tasks{task}).dataset(s).p, modelstruct(task), 'model_fitting_data', raw.(tasks{task}));
        else
            fake_datasets.(tasks{task}).dataset(s).raw = trial_generator(fake_datasets.(tasks{task}).dataset(s).p, modelstruct(task));
        end
        fake_datasets.(tasks{task}).dataset(s).stats = indiv_analysis_fcn(fake_datasets.(tasks{task}).dataset(s).raw, bins, 'output_fields', dep_vars);
    end
    
    fake_datasets.(tasks{task}).sumstats = sumstats_fcn(fake_datasets.(tasks{task}).dataset, 'fields', dep_vars);
end