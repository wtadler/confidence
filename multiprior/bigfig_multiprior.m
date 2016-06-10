function h=bigfig_multiprior(varargin)
%%% FIX LABELING
root_datadir = '~/Google Drive/Will - Confidence/Data/v3_multiprior';
nBins = 5;
% conf_levels = 4;
symmetrify = false;
% mean_color = [0 0 .8];
tasks = {'A','B'};
trial_types = {'prior1', 'prior2', 'prior3'};%, 'Chat1', 'Chat2'}
% linewidth = 2;
gutter = [.013 .085];
margins = [.2 .01 .14 .06]; % L R B T
% show_legend = false;
s_labels = [-8 -4 -2 -1 0 1 2 4 8];
letter_size = 14;
nPlotSamples = 20;
nFakeGroupDatasets = 1000;
model = []; % set to 0 for no model fit. otherwise indicate which model you want to do.
plot_reliabilities = 1:6;
assignopts(who,varargin);

if ~isempty(model)
    if ~model.joint_task_fit
        if model.diff_mean_same_std
            tasks = {'A'};
        else
            tasks = {'B'};
        end
    end
end

A = false;
B = false;

if any(strcmp(tasks, 'A'))
    A = true;
end
if any(strcmp(tasks, 'B'))
    B = true;
end

nTasks = length(tasks);
nRows = 2;
nCols = 3;

for task = 1:nTasks
    [edges.(tasks{task}), centers.(tasks{task})] = bin_generator(nBins, 'task', tasks{task});
end

real_data = compile_and_analyze_data(root_datadir, 'nBins', nBins,...
    'symmetrify', symmetrify, 'trial_types', trial_types,...
    'group_stats', true);

map = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
colors = map.tan_contrast_colors;

figure(1)
clf
set(gcf,'position', [100 100 250*nCols 200*nRows])
%%
if ~isempty(model)
    model = generate_and_analyze_fitted_data(model, tasks, 'real_data', real_data, 'nBins', nBins, 'nPlotSamples', nPlotSamples, ...
        'nFakeGroupDatasets', nFakeGroupDatasets, 'depvars', {'Chat'}, 'symmetrify', symmetrify, ...
        'group_plot', true, 'trial_types', trial_types, 'bin_types', {'c_C'});
    uicontrol('Style', 'text', 'String', rename_models(model.name), 'Units', 'normalized', 'Position', [0 0 .45 .05], 'BackgroundColor', 'w', 'fontsize', 18, 'fontname', 'Helvetica', 'horizontalalignment', 'left')
end
%%

% proportion report cat. 1 vs category and reliability
if A
tight_subplot(nRows, nCols, 1,1, gutter, margins);
crazyplot(real_data, model, 'A', 'prior1', 'c_C', 'Chat', 'label_x', false, 'label_y', true, 'show_legend', true, 'legend_loc', 'southwest', 'plot_reliabilities', plot_reliabilities);
ylabel('prop. report "cat. 1"');
label('a');
title('cat. 1 prior')
% label Task A
yl=get(gca,'ylim');
half=yl(1)+diff(yl)/2;
text(8.5, half, 'Task A', 'horizontalalignment', 'right', 'fontweight','bold', 'fontsize', letter_size+2);
end

if B
tight_subplot(nRows, nCols, 2,1, gutter, margins);
crazyplot(real_data, model, 'B', 'prior1', 'c_C', 'Chat', 'label_x', true, 'label_y', true, 'plot_reliabilities', plot_reliabilities);
ylabel('prop. report "cat. 1"');
label('b')
% label Task B
yl=get(gca,'ylim');
half=yl(1)+diff(yl)/2;
text(8.5, half, 'Task B', 'horizontalalignment', 'right', 'fontweight','bold', 'fontsize', letter_size+2);
end


if A
tight_subplot(nRows, nCols, 1,2, gutter, margins);
crazyplot(real_data, model, 'A', 'prior2', 'c_C', 'Chat', 'label_x', false, 'label_y', false, 'plot_reliabilities', plot_reliabilities);
label('c');
title('neutral prior')
end

if B
tight_subplot(nRows, nCols, 2,2, gutter, margins);
crazyplot(real_data, model, 'B', 'prior2', 'c_C', 'Chat', 'label_x', true, 'label_y', false, 'plot_reliabilities', plot_reliabilities);
label('d')
end

if A
tight_subplot(nRows, nCols, 1,3, gutter, margins);
crazyplot(real_data, model, 'A', 'prior3', 'c_C', 'Chat', 'label_x', false, 'label_y', false, 'plot_reliabilities', plot_reliabilities);
label('e');
title('cat. 2 prior')
end

if B
tight_subplot(nRows, nCols, 2,3, gutter, margins);
crazyplot(real_data, model, 'B', 'prior3', 'c_C', 'Chat', 'label_x', true, 'label_y', false, 'plot_reliabilities', plot_reliabilities);
label('f')
end


% % mean button press vs category and reliability
% if A
% tight_subplot(nRows, nCols, 1,4, gutter, margins);
% crazyplot(real_data, model, 'A', 'prior1', 'c_s', 'resp', 'label_x', false, 'label_y', true);
% label('g');
% yl=ylabel('mean button press');ylpos = get(yl, 'position')
% set(yl, 'position', ylpos-[.25 0 0])
% title('cat. 1 prior')
% end
% 
% if B
% tight_subplot(nRows, nCols, 2,4, gutter, margins);
% crazyplot(real_data, model, 'B', 'prior1', 'c_s', 'resp', 'label_x', true, 'label_y', true);
% label('h')
% yl=ylabel('mean button press');ylpos = get(yl, 'position')
% set(yl, 'position', ylpos-[.25 0 0])
% end
% 
% if A
% tight_subplot(nRows, nCols, 1,5, gutter, margins);
% crazyplot(real_data, model, 'A', 'prior2', 'c_s', 'resp', 'label_x', false, 'label_y', false);
% label('i');
% title('neutral prior')
% end
% 
% if B
% tight_subplot(nRows, nCols, 2,5, gutter, margins);
% crazyplot(real_data, model, 'B', 'prior2', 'c_s', 'resp', 'label_x', true, 'label_y', false);
% label('j')
% end
% 
% if A
% tight_subplot(nRows, nCols, 1,6, gutter, margins);
% crazyplot(real_data, model, 'A', 'prior3', 'c_s', 'resp', 'label_x', false, 'label_y', false);
% label('k');
% title('cat. 2 prior')
% end
% 
% if B
% tight_subplot(nRows, nCols, 2,6, gutter, margins);
% crazyplot(real_data, model, 'B', 'prior3', 'c_s', 'resp', 'label_x', true, 'label_y', false);
% label('l')
% end
%%

    function h=crazyplot(real_data, fake_data, task, trial_type, x, y, varargin)
        if ~isempty(fake_data)
            hold on
            h{2}=single_dataset_plot(fake_data.(task).sumstats.(trial_type).(x), y, x, ...
                'fake_data', true, 'group_plot', true, 's_labels', s_labels,...
                'task', task,  varargin{:});
            line_through_errorbars = false;
        else
            line_through_errorbars = true;
        end
        
        h{1}=single_dataset_plot(real_data.(task).sumstats.(trial_type).(x), y, x, ...
            'fake_data', false, 'group_plot', true, 's_labels', s_labels,...
            'task', task, 'plot_connecting_line', line_through_errorbars,...
            varargin{:});
    end

    function label(letter)
        xl=get(gca,'xlim');
        xrange = diff(xl);
        yl=get(gca,'ylim');
        yrange = diff(yl);
        
        if strcmp(get(gca, 'xdir'), 'reverse')
            xside = 2;
            xsign = -1;
        else
            xside = 1;
            xsign = 1;
        end
        
        if strcmp(get(gca, 'ydir'), 'reverse')
            yside = 1;
            ysign = -1;
        else
            yside = 2;
            ysign = 1;
        end
        
        t=text(xl(xside)-xsign*xrange*.1, yl(yside)+ysign*yrange*.11, letter);
        set(t, 'verticalalignment', 'top', 'horizontalalignment', 'right', 'fontweight', 'bold', 'fontsize', letter_size);
    end
end