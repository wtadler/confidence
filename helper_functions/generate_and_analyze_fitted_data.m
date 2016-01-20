function models = generate_and_analyze_fitted_data(models, tasks, varargin)

nBins = 7;
nPlotSamples = 10;
nFakeGroupDatasets = 100;
depvars = {'tf'};
symmetrify = false;
bin_types = {'c_s'};
attention_task = false;
group_plot = false;
real_data = [];
trial_types = {'all'};
assignopts(who, varargin);

nModels = length(models);
for m = 1:nModels
    fprintf('\nAnalyzing generated data from model %i/%i...', m, nModels);

    nSubjects = length(models(m).extracted);
    for dataset = 1:nSubjects
        for task = 1:length(tasks)
            raw.(tasks{task}) = real_data.(tasks{task}).data(dataset).raw;
        end
        
        % for each subject and model, generate and analyze nPlotSamples datasets
        models(m).extracted(dataset).fake_datasets = dataset_generator(models(m),...
            models(m).extracted(dataset).p, nPlotSamples, 'nBins', nBins,...
            'raw', raw, 'tasks', tasks, 'dep_vars', depvars, 'symmetrify', symmetrify,...
            'bin_types', bin_types, 'attention_task', attention_task, 'trial_types', trial_types);
        fprintf('\nGenerating data from model %i/%i for subject %i/%i...', m, nModels, dataset, nSubjects);
    end
    
    if group_plot
        % randomly sample 1 fake dataset from each subject
        % nFakeGroupDatasets times and analyze that grouped dataset
        fake_sumstats = fake_group_datasets_and_stats(models(m), nFakeGroupDatasets, 'fields', depvars);
        tasks = fieldnames(fake_sumstats);
        for t = 1:length(tasks);
            models(m).(tasks{t}).sumstats = fake_sumstats.(tasks{t});
        end
    end
end
