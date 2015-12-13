function ah = show_data(varargin)
% this function does a lot of things. some options:
% 1. show individual data
% 2. (and individual fits)
% 3. show grouped data
% 4. (and group fits)

% it's gotten unwieldy. maybe break it up to allow more flexible plotting.

root_datadir = '~/Google Drive/Will - Confidence/Data/v3_all';
depvars = {'tf'};%,       'g',        'Chat',     'resp',     'rt'};
nBins = 7;
conf_levels = 4;
% plot_error_bars = true; % could eventually add functionality to just show means. or to not show means
symmetrify = false;
slices = {'c_g', 'c_resp', 'c_s', ''}; % 's', 'c_s', 'c_resp', etc etc etc. figure out how to add a blank
means = {'g',   'resp',   's', 'c'};
mean_color = [0 0 .8];
tasks = {'A','B'};
axis = struct;
axis.col = 'slice'; % 'subject', 'slice', or 'model'. defaults to subject if not doing group plots
axis.row = 'task'; % 'task', 'model', or 'depvar'
axis.fig = 'none'; % 'model', 'task', 'depvar', or 'slice'
trial_type = 'all'; % 'all', 'correct', 'incorrect', etc...
% group_plot = true;
linewidth = 2;
meanlinewidth = 4;
ticklength = .018;
gutter = [.0175 .025];
margins = [.06 .01 .06 .04]; % L R B T
models = [];
nPlotSamples = 10;
nFakeGroupDatasets = 100;
plot_reliabilities = [];
show_legend = false;
s_labels = -8:2:8;
assignopts(who, varargin);

if strcmp(axis.col, 'subject') % in all non-group plots, subjects are along the col axis
    group_plot = false;
else
    group_plot = true;
end

if rem(nBins, 2) == 0; nBins = nBins +1; end % make sure nBins is odd.

% blue to red colormap
map = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
map = map.confchoicemap;
button_colors = map(round(linspace(1,256,8)),:);

nDepvars = length(depvars);
nSlices = length(slices);
nTasks = length(tasks);

ylims = zeros(nDepvars, 2);
for dv = 1:nDepvars
    if strcmp(depvars{dv}, 'tf')
        ylims(dv,:) = [.3 1];
    elseif strcmp(depvars{dv}, 'g')
        ylims(dv,:) = [1 4];
    elseif strcmp(depvars{dv}, 'Chat')
        ylims(dv,:) = [0 1];
    elseif strcmp(depvars{dv}, 'resp')
        ylims(dv,:) = [1 8];
    elseif strcmp(depvars{dv}, 'rt')
        ylims(dv,:) = [0 4];
    end
end

for task = 1:nTasks
    [edges.(tasks{task}), centers.(tasks{task})] = bin_generator(nBins, 'task', tasks{task});
end

real_data = compile_and_analyze_data(root_datadir, 'nBins', nBins,...
    'symmetrify', symmetrify, 'conf_levels', conf_levels, 'trial_types', {trial_type},...
    'output_fields', depvars, 'bin_types', union(slices, means), 'group_plot', group_plot);

nSubjects = length(real_data.(tasks{1}).data);


if isfield(real_data.(tasks{1}).data(1).raw, 'cue_validity') && ~isempty(real_data.(tasks{1}).data(1).raw.cue_validity)
    % attention
    
    nReliabilities = length(unique(real_data.(tasks{1}).data(1).raw.cue_validity_id));
    attention_task = true;
    colors = flipud([.7 0 0;.6 .6 .6;0 .7 0]);
    
else
    nReliabilities = length(unique(real_data.(tasks{1}).data(1).raw.contrast_id));
    attention_task = false;
    
    if isempty(plot_reliabilities); plot_reliabilities = 1:nReliabilities; end
    
    if max(plot_reliabilities) > nReliabilities; error('you requested to plot more reliabilities than there are'); end
        
    hhh = hot(64);
    colors = hhh(round(linspace(1,40,nReliabilities)),:); % black to orange indicate high to low contrast
end

if ~isempty(models)
    show_fits = true;
    nModels = length(models);
    
    models = generate_and_analyze_fitted_data(models, tasks, 'real_data', real_data, 'nBins', nBins, 'nPlotSamples', nPlotSamples,...
        'depvars', depvars, 'symmetrify', symmetrify, 'bin_types', union(slices, means),...
        'attention_task', attention_task, 'group_plot', group_plot);
    
    
%     for m = 1:nModels
%         for dataset = 1:nSubjects
%             for task = 1:nTasks
%                 raw.(tasks{task}) = real_data.(tasks{task}).data(dataset).raw;
%             end
%             models(m).extracted(dataset).fake_datasets = dataset_generator(models(m),...
%                 models(m).extracted(dataset).p, nPlotSamples, 'nBins', nBins,...
%                 'raw', raw, 'tasks', tasks, 'dep_vars', depvars, 'symmetrify', symmetrify,...
%                 'bin_types', union(slices,means), 'attention_task', attention_task);
%             fprintf('\nGenerating data from model %i/%i for subject %i/%i...', m, nModels, dataset, nSubjects);
%         end
% 
%         models(m).fake_sumstats = fake_group_datasets_and_stats(models(m), nFakeGroupDatasets, 'fields', depvars);
%         fprintf('\nAnalyzing generated data from model %i/%i...', m, nModels);
%     end
else
    show_fits = false;
    nModels = 0;
end


n = struct;
plot_axes = {'col', 'row', 'fig'};
for i = 1:3
    switch axis.(plot_axes{i})
        case 'depvar'
            n.(plot_axes{i}) = nDepvars;
        case 'task'
            n.(plot_axes{i}) = nTasks;
        case 'model'
            n.(plot_axes{i}) = nModels;
        case 'slice'
            n.(plot_axes{i}) = nSlices;
        case 'subject'
            n.(plot_axes{i}) = nSubjects;
        case 'none'
            n.(plot_axes{i}) = 1;
    end
end

ah = zeros(n.row, n.col, n.fig);

xticklabels = cell(1, nSlices);
xticks = cell(nTasks, nSlices);
xlabels = cell(1, nSlices);
for slice = 1:nSlices
    if ~isempty(slices{slice})
        label_slice = slices{slice};
    else
        label_slice = means{slice};
    end
    
    switch label_slice
        case {'g', 'c_g'}
            xlabels{slice} = 'confidence';
            [xticklabels{slice}, xticks{:, slice}] = deal(1:conf_levels);
        case {'resp', 'c_resp'}
            xlabels{slice} = 'button press';
            [xticklabels{slice}, xticks{:, slice}] = deal(1:2*conf_levels);
        case {'Chat', 'c_Chat'}
            xlabels{slice} = 'cat. choice';
            [xticklabels{slice}, xticks{:, slice}] = deal(1:2);
        case {'c'}
            [xticklabels{slice}, xticks{:, slice}] = deal(1:nReliabilities);
            if ~attention_task
                xlabels{slice} = 'contrast/eccentricity';
            else
                xlabels{slice} = 'cue validity';
                xticklabels{slice} = {'valid', 'neutral', 'invalid'};
            end
        case {'s', 'c_s'}
            for task = 1:nTasks
                xticks{task, slice} = interp1(centers.(tasks{task}), 1:nBins, s_labels);
            end
            
%             if symmetrify
%                 xlabels{slice} = '|s|';
%                 xticklabels{slice} = abs(s_labels);
%             else
                xlabels{slice} = 's';
                xticklabels{slice} = s_labels;
%             end
    end
end

ylabels = rename_var_labels(depvars); % translate from variable names to something other people can understand.

[depvar, task, model, slice, subject] = deal(1); % update these in the for loop switch below.

%%
for fig = 1:n.fig
    figure(fig)
    clf
    
    for col = 1:n.col
        for row = 1:n.row
            for i = 1:3
                
                switch axis.(plot_axes{i})
                    case 'depvar'
                        depvar = eval(plot_axes{i});
                    case 'task'
                        task = eval(plot_axes{i});
                    case 'model'
                        model = eval(plot_axes{i});
                    case 'slice'
                        slice = eval(plot_axes{i});
                    case 'subject'
                        subject = eval(plot_axes{i});
                end
            end
            
            
            ah(row, col, fig) = tight_subplot(n.row, n.col, row, col, gutter, margins);
            
            if symmetrify && any(strcmp(slices{slice}, {'s', 'c_s'})) && strcmp(tasks{task}, 'B')
                symmetrify_s = true;
            else
                symmetrify_s = false;
            end
            
            shortcutplot = @(data, fake_data, colors, linewidth, plot_reliabilities)...
                single_dataset_plot(data, depvars{depvar},...
                    'fake_data', fake_data, 'group_plot', group_plot, ...
                    'symmetrify', symmetrify_s, 'colors', colors, ...
                    'linewidth', linewidth, ...
                    'plot_reliabilities', plot_reliabilities);
                
                % clean this section up!!
                fake_data = false;
                % plot real sliced data
                if ~isempty(slices{slice})
                    if ~group_plot
                        data = real_data.(tasks{task}).data(subject).stats.(trial_type).(slices{slice});
                    else
                        data = real_data.(tasks{task}).sumstats.(trial_type).(slices{slice});
                    end
                    shortcutplot(data, fake_data, colors, linewidth, plot_reliabilities);
                end
                
                % plot real "mean" data
                if ~isempty(means{slice})
                    if ~group_plot
                        data = real_data.(tasks{task}).data(subject).stats.(trial_type).(means{slice});
                    else
                        data = real_data.(tasks{task}).sumstats.(trial_type).(means{slice});
                    end
                    shortcutplot(data, fake_data, mean_color, meanlinewidth, []);
                end
                
                % plot fitted sliced data
                if show_fits
                    fake_data = true;
                    if ~isempty(slices{slice})
                        if ~group_plot
                            data = models(model).extracted(subject).fake_datasets.(tasks{task}).sumstats.(trial_type).(slices{slice});
                        else
                            data = models(model).fake_sumstats.(tasks{task}).(slices{slice}); % fake_group_datasets_and_stats doesn't have support for trial_type. i think that's okay 12/11/15
                        end
                        try
                        shortcutplot(data, fake_data, colors, linewidth, plot_reliabilities);
                        catch
                            'bla'
                        end
                    end
                    
                    if ~isempty(means{slice})
                        if ~group_plot
                            data = models(model).extracted(subject).fake_datasets.(tasks{task}).sumstats.(trial_type).(means{slice});
                        else
                            data = models(model).fake_sumstats.(tasks{task}).(means{slice});
                        end
                        shortcutplot(data, fake_data, mean_color, meanlinewidth, []);
                    end

                end
                
            % x axis labels for bottom row
            if row == n.row
                xlabel(xlabels{slice})
                set(gca, 'xticklabel', xticklabels{slice})
            else
                xlabel('')
                set(gca, 'xticklabel', '')
            end
                       
                
            set(gca, 'ticklength', [ticklength ticklength], 'xtick', xticks{task, slice})
            if any(strcmp(slices{slice}, {'s', 'c_s'}))
                xlim([.5 nBins+.5])
            else
                xlim([xticks{task, slice}(1)-.5 xticks{task, slice}(end)+.5])
            end
            
            switch depvars{depvar}
                case {'tf','Chat'}
                    plot_halfway_line(.5)
                case 'resp'
                    set(gca, 'ydir', 'reverse')
                    plot_halfway_line(4.5)
            end
            
            % y axis labels for left column
            if col == 1
                yl=ylabel([ylabels{depvar},', Task ' tasks{task}]); % make this more flexible. use: rename_models(model.name)]);
                if ~strcmp(depvars{depvar}, 'resp')
                    set(gca, 'yticklabelmode', 'auto')
                else
                    set(gca, 'clipping', 'off', 'yticklabel', '')
                    xl = get(gca, 'xlim');
                    xrange = diff(xl);
                    for r = 1:8
                        plot(xl(1)-.16*xrange, r, 'square', 'markerfacecolor', button_colors(r,:), 'markeredgecolor', button_colors(r,:), 'markersize', 12)
                    end
                    ylpos = get(yl, 'position');
                    set(yl, 'position', ylpos-[.8 0 0]);
                end % figure out how to do these squares for x
            else
                set(gca, 'yticklabel', '')
            end

            
            % title (and maybe legend) for top row
            if row == 1
                switch axis.col
                    case 'subject'
                        title(real_data.(tasks{task}).data(subject).name);
                    case 'model'
                        title(rename_models(models(model).name));
                end
                
                if col == 1
                    if show_legend
                        legend(labels)
                        
                        if ~group_fits
                            t=title(upper(real_data.(tasks{task}).data(col).name))
                        elseif group_fits
                            t=title(rename_models(model.name));
                            set(gca, 'xticklabel', ori_labels.(tasks{task}))
                        end
                        
                        if col == 1
                            if show_legend
                                warning('add legend functionality')
                            end
                        end
                    end
                    set(gca,'color','none')
                end
            end
        end
    end
end