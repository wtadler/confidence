function [ah, models]=plot_group_fits(models, varargin)

nBins = 13; % should be odd
marg_over_s = false; % this doesn't work yet
nPlotSamples = 2;
nHyperplots = 100; % number of times to take a random fake dataset from each subject. has only small effect on computation time.
dep_vars = {'resp','g','Chat','tf'};
root_datadir = '/Users/will/Google Drive/Will - Confidence/Data/v3_all';
tasks = {'A', 'B'};
yaxis = 'task'; % 'depvar', 'task'
gutter = [.013 .04];
margins = [.05 .01 .07 .04]; % L R B T
plot_reliabilities = [2 4 6];
stagger_titles = false;
assignopts(who, varargin);

nTasks = length(tasks);

datadir = check_datadir(root_datadir);

[edges.A, centers.A] = bin_generator(nBins, 'task', 'A');
[edges.B, centers.B] = bin_generator(nBins, 'task', 'B');

if strcmp(yaxis, 'task') && nTasks == 1
    warning('you requested tasks on the y axis but there is only one task')
end

for task = 1:nTasks
    if ~isfield(datadir, tasks{task})
        error('something is wrong')
    end
    
    % load real data
    streal.(tasks{task}) = compile_data('datadir', datadir.(tasks{task}));
    
    nSubjects = length(models(1).extracted);
    if nSubjects ~= length(streal.(tasks{task}).data)
        error('something''s wrong')
    end
    
    % analyze real data
    for subject = 1:nSubjects
        [streal.(tasks{task}).data(subject).stats, streal.(tasks{task}).data(subject).sorted_raw] = indiv_analysis_fcn(streal.(tasks{task}).data(subject).raw, edges.(tasks{task}));
    end
    % real summary stats
    real_sumstats.(tasks{task}) = sumstats_fcn(streal.(tasks{task}).data);
end

t_start = tic;

for m = 1:length(models)
%     model = models(m);
    clear fake
    for subject = 1:nSubjects%, matlabpool('size')) % parallelizes if pool is open
        for task = 1:nTasks
            raw(subject).(tasks{task}) = streal.(tasks{task}).data(subject).raw;
        end
        
        models(m).extracted(subject).fake_datasets = dataset_generator(models(m), models(m).extracted(subject).p, nPlotSamples, ...
            'nBins', nBins, 'raw', raw(subject), 'tasks', tasks, 'dep_vars', dep_vars); % generates fake datasets for both tasks
        warning('should tasks be tasks_in? figure this out!!! 9/23/15')
        
        prop_complete = ((m-1)*nSubjects+subject)/(length(models)*nSubjects);
        secs_remaining = (toc(t_start)/prop_complete - toc(t_start));
        fprintf('%.i%%, %.f secs remaining\n',round(100*prop_complete), secs_remaining)
    end
    
    models(m).fake_sumstats = hyperplot(models(m), nHyperplots, 'fields', dep_vars);
end

ah = show_data('root_datadir', root_datadir, 'real_sumstats', real_sumstats, 'models', models, 'marg_over_s', marg_over_s, ...
    'gutter', gutter, 'margins', margins, 'plot_reliabilities', plot_reliabilities, 'yaxis', yaxis, 'nBins', nBins, 'dep_vars', dep_vars, 'stagger_titles', stagger_titles);
