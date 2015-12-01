function ah=plot_individual_fits(models, varargin)

nBins = 13; % should be odd
marg_over_s = false; % this doesn't work yet
nPlotSamples = 50;
dep_vars = {'resp','g','Chat'};
root_datadir = '~/Google Drive/Will - Confidence/Data/v3_all';
tasks = {'A', 'B'};
yaxis = 'depvar'; % 'depvar', 'task', 'model'.
gutter = [.0175 .025];
plot_reliabilities = [2 4 6];
assignopts(who, varargin);

nTasks = length(tasks);

datadir = check_datadir(root_datadir);

if strcmp(yaxis, 'task') || strcmp(yaxis, 'model')
    dep_vars = {'resp'};
end

if strcmp(yaxis, 'task') && nTasks == 1
    warning('you requested tasks on the y axis but there is only one task')
end

for task = 1:nTasks
    if ~isfield(datadir, tasks{task})
        error('something is wrong')
    end
    
    streal.(tasks{task}) = compile_data('datadir', datadir.(tasks{task}));
    
    nSubjects = length(models(1).extracted);
    if nSubjects ~= length(streal.(tasks{task}).data)
        error('something''s wrong')
    end
    
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
        
        prop_complete = ((m-1)*nSubjects+subject)/(length(models)*nSubjects);
        secs_remaining = (toc(t_start)/prop_complete - toc(t_start));
        fprintf('%.i%%, %.f secs remaining\n',round(100*prop_complete), secs_remaining)

    end
end

if ~strcmp(yaxis, 'model')
    for m = 1:length(models)
    ah=show_data('root_datadir', root_datadir, 'nBins', nBins, 'marg_over_s', marg_over_s, 'models', models(m),...
        'yaxis', yaxis, 'dep_vars', dep_vars, 'tasks', tasks, 'plot_reliabilities', plot_reliabilities, ...
        'gutter', gutter);
    
    uicontrol('Style','text','String',models(m).name, 'Units', 'normalized','Position',[0 0.988 .4 .012])
    
    pause(.1);
    end
else
    ah=show_data('root_datadir', root_datadir, 'nBins', nBins, 'marg_over_s', marg_over_s, 'models', models,...
        'yaxis', yaxis, 'dep_vars', dep_vars, 'tasks', tasks, 'plot_reliabilities', plot_reliabilities, ...
        'gutter', gutter);
end