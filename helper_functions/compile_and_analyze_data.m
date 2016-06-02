function real_data = compile_and_analyze_data(root_datadir, varargin)

nBins = 7;
symmetrify = false;
conf_levels = 4;
trial_types = {'all'};
output_fields = {'tf','resp','g','rt','Chat','proportion'};
bin_types = {'c', 's', 'Chat', 'g', 'resp', 'c_s', 'c_C', 'c_Chat', 'c_g', 'c_resp', 'c_prior'};
group_stats = false;
matchstring = '';
assignopts(who, varargin)

datadir = check_datadir(root_datadir);
tasks = fieldnames(datadir);
nTasks = length(tasks);

for task = 1:nTasks;
    fprintf('Task %i/%i: Analyzing subject data...\n', task, nTasks);

    real_data.(tasks{task}) = compile_data('datadir',datadir.(tasks{task}), 'matchstring', matchstring);
    [edges.(tasks{task}), centers.(tasks{task})] = bin_generator(nBins, 'task', tasks{task});
    
    nSubjects = length(real_data.(tasks{task}).data);
    for dataset = 1:nSubjects            
        if symmetrify && strcmp(tasks{task}, 'B')
            real_data.(tasks{task}).data(dataset).raw.s = abs(real_data.(tasks{task}).data(dataset).raw.s);
        end
        real_data.(tasks{task}).data(dataset).stats = indiv_analysis_fcn(real_data.(tasks{task}).data(dataset).raw,...
            edges.(tasks{task}),'conf_levels', conf_levels,...
            'trial_types', trial_types, 'output_fields', output_fields,...
            'bin_types', bin_types);
    end
    

    if group_stats
        real_data.(tasks{task}).sumstats = sumstats_fcn(real_data.(tasks{task}).data, ...
            'fields', output_fields, 'bootstrap', true);
    end
end
