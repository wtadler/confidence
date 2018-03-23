function h=bigfig(varargin)
%%% FIX LABELING
root_datadir = '~/Google Drive/Will - Confidence/Data/v3_all';
nBins = 11;
% conf_levels = 4;
symmetrify = false;
% mean_color = [0 0 .8];
tasks = {'A','B'};
trial_types = {'all', 'correct', 'incorrect', 'C1', 'C2'};%, 'Chat1', 'Chat2'}
% linewidth = 2;
gutter = [.04 .07];
margins = [.15 .01 .14 .06]; % L R B T
% show_legend = false;
s_labels = [-8 -4 -2 -1 0 1 2 4 8];
axes_label_fontsize = 16; % axes labels
xy_label_fontsize = 14; % xlabel and ylabel
legend_fontsize = 10;
tick_label_fontsize = 11; % xticklabel and yticklabel
task_label_fontsize = 19;
nPlotSamples = 10;
nFakeGroupDatasets = 100;
model = []; % set to 0 for no model fit. otherwise indicate which model you want to do.
plot_reliabilities = 1:6;
assignopts(who,varargin);

bin_types = {'c_C', 'c_g', 'g', 'c', 's', 'c_s'};
depvars = {'resp', 'tf', 'g', 'Chat', 'proportion'};
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
nRows = 4;
nCols = 4;

for task = 1:nTasks
    [edges.(tasks{task}), centers.(tasks{task})] = bin_generator(nBins, 'task', tasks{task});
end

real_data = compile_and_analyze_data(root_datadir, 'nBins', nBins,...
    'symmetrify', symmetrify, 'trial_types', trial_types,...
    'group_stats', true, 'bin_types', bin_types, 'output_fields', depvars);

map = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
colors = map.tan_contrast_colors;

figure(1)
set(gcf,'position', [100 100 240*nCols 190*nRows])
clf
letter = 1;
%%
if ~isempty(model)
    model = generate_and_analyze_fitted_data(model, tasks, 'real_data', real_data, 'nBins', nBins, 'nPlotSamples', nPlotSamples, ...
        'nFakeGroupDatasets', nFakeGroupDatasets, 'depvars', depvars, 'symmetrify', symmetrify, ...
        'group_plot', true, 'trial_types', trial_types, 'bin_types', bin_types);
    uicontrol('Style', 'text', 'String', rename_models(model.name), 'Units', 'normalized', 'Position', [0 0 .45 .05], 'BackgroundColor', 'w', 'fontsize', 18, 'fontname', 'Helvetica', 'horizontalalignment', 'left')
end

delta = .02;
leftshift1 = 0;
leftshift2 = leftshift1+delta;

captionletterpropleft1 = .11;
captionletterpropleft2 = .05;

% proportion report cat. 1 vs category and reliability
if A
tight_subplot(nRows, nCols, 1,1, gutter, margins);
crazyplot(real_data, model, 'A', 'all', 'c_C', 'Chat', true, leftshift1, captionletterpropleft1, 'label_x', false, 'label_y', true, 'show_legend', true, 'legend_loc', 'northwest');
ylabel('prop. report "cat. 1"', 'fontsize', xy_label_fontsize);
% label Task A
xl=get(gca,'xlim');
task_x=xl(1)+diff(xl)/2;
yl=get(gca,'ylim');
task_y = yl(2)+(diff(yl))*.2;

text(task_x, task_y, 'Task A', 'horizontalalignment', 'center', 'fontweight','bold', 'fontsize', task_label_fontsize);
end

if B
tight_subplot(nRows, nCols, 1,2, gutter, margins);
crazyplot(real_data, model, 'B', 'all', 'c_C', 'Chat', true, leftshift2, captionletterpropleft2, 'label_x', false, 'label_y', false);
% label Task B
xl=get(gca,'xlim');
task_x=xl(1)+diff(xl)/2;
yl=get(gca,'ylim');
task_y = yl(2)+(diff(yl))*.2;
text(task_x, task_y, 'Task B', 'horizontalalignment', 'center', 'fontweight','bold', 'fontsize', task_label_fontsize);
end


% mean button press vs category and reliability
if A
tight_subplot(nRows, nCols, 2,1, gutter, margins);
crazyplot(real_data, model, 'A', 'all', 'c_C', 'resp', true, leftshift1, captionletterpropleft1, 'label_x', true, 'label_y', true, 'nRespSquares', 4);
yl=ylabel('mean button press', 'fontsize', xy_label_fontsize);
ylpos = get(yl, 'position');
set(yl,'position',ylpos+[.45 0 0]);
end

if B
tight_subplot(nRows, nCols, 2,2, gutter, margins);
crazyplot(real_data, model, 'B', 'all', 'c_C', 'resp', true, leftshift2, captionletterpropleft2, 'label_x', true, 'label_y', false, 'nRespSquares', 4);
end

if A
% histogram of confidence ratings by category
tight_subplot(nRows, nCols, 3,1, gutter, margins);
ylim([0 .6])
crazyplot(real_data, model, 'A', 'C1', 'g', 'proportion', false, 0, 0, 'label_x', true, 'label_y', true, 'color', map.cat1);
crazyplot(real_data, model, 'A', 'C2', 'g', 'proportion', true, leftshift1, captionletterpropleft1, 'label_x', true, 'label_y', true, 'color', map.cat2);
ylabel('prop. of total', 'fontsize', xy_label_fontsize)
end

if B
tight_subplot(nRows, nCols, 3,2, gutter, margins);
ylim([0 .6])
crazyplot(real_data, model, 'B', 'C1', 'g', 'proportion', false, 0, 0, 'label_x', true, 'label_y', false, 'color', map.cat1);
crazyplot(real_data, model, 'B', 'C2', 'g', 'proportion', true, leftshift2, captionletterpropleft2, 'label_x', true, 'label_y', false, 'color', map.cat2);
end

if nTasks > 1
% proportion correct vs confidence
tight_subplot(nRows, nCols, 4,1, gutter, margins);
apos = get(gca, 'position');
delta_x = (apos(3) + gutter(1) - delta)/2;
set(gca, 'position', apos + [delta_x 0 0 0]);
ylim([.45 .93])

a=crazyplot(real_data, model, 'A', 'all', 'g', 'tf', false, 0, 0, 'label_x', true, 'label_y', true, 'color', map.taskA);
b=crazyplot(real_data, model, 'B', 'all', 'g', 'tf', true, leftshift1, captionletterpropleft1, 'label_x', true, 'label_y', true, 'color', map.taskB);
ylabel('prop. correct', 'fontsize', xy_label_fontsize);
l=legend([a{1}(1),b{1}(1)],'Task A','Task B', 'fontsize', legend_fontsize);
set(l,'box','off','location','northwest');
end



if A
% mean confidence vs binned orientation conditioned on correctness
tight_subplot(nRows, nCols, 1,3, gutter, margins);
correct=crazyplot(real_data, model, 'A', 'correct', 'c', 'g', false, 0, 0, 'label_x', false, 'label_y', true, 'color', map.correct);
incorrect=crazyplot(real_data, model, 'A', 'incorrect', 'c', 'g',  true, leftshift1, captionletterpropleft1, 'label_x', true, 'label_y', true, 'color', map.incorrect);
l=legend([correct{1}(1),incorrect{1}(1)],'correct','incorrect', 'fontsize', legend_fontsize);
set(l,'box','off','location','northwest')
ylabel('mean confidence', 'fontsize', xy_label_fontsize)
% label Task A
xl=get(gca,'xlim');
task_x=xl(1)+diff(xl)/2;
yl=get(gca,'ylim');
task_y = yl(2)+(diff(yl))*.2;

text(task_x, task_y, 'Task A', 'horizontalalignment', 'center', 'fontweight','bold', 'fontsize', task_label_fontsize);

end

if B
tight_subplot(nRows, nCols, 1,4, gutter, margins);
crazyplot(real_data, model, 'B', 'correct', 'c', 'g', false, 0, 0, 'label_x', false, 'label_y', false, 'color', map.correct);
crazyplot(real_data, model, 'B', 'incorrect', 'c', 'g', true, leftshift2, captionletterpropleft2, 'label_x', true, 'label_y', false, 'color', map.incorrect);
% label Task B
xl=get(gca,'xlim');
task_x=xl(1)+diff(xl)/2;
yl=get(gca,'ylim');
task_y = yl(2)+(diff(yl))*.2;

text(task_x, task_y, 'Task B', 'horizontalalignment', 'center', 'fontweight','bold', 'fontsize', task_label_fontsize);
end

if A
% mean confidence vs binned orientation conditioned on correctness
tight_subplot(nRows, nCols, 2,3, gutter, margins);
correct=crazyplot(real_data, model, 'A', 'correct', 's', 'g', false, 0, 0, 'label_x', false, 'label_y', true, 'color', map.correct);
incorrect=crazyplot(real_data, model, 'A', 'incorrect', 's', 'g', true, leftshift1, captionletterpropleft1, 'label_x', false, 'label_y', true, 'color', map.incorrect);
ylabel('mean confidence', 'fontsize', xy_label_fontsize)

end

if B
tight_subplot(nRows, nCols, 2,4, gutter, margins);
crazyplot(real_data, model, 'B', 'correct', 's', 'g', false, 0, 0, 'label_x', false, 'label_y', false, 'color', map.correct);
crazyplot(real_data, model, 'B', 'incorrect', 's', 'g', true, leftshift2, captionletterpropleft2, 'label_x', false, 'label_y', false, 'color', map.incorrect);
end


if A
% choice vs orientation for all reliabilities
tight_subplot(nRows, nCols, 3,3, gutter, margins);
h=crazyplot(real_data, model, 'A', 'all', 'c_s', 'Chat', true, leftshift1, captionletterpropleft1, 'label_x', false, 'label_y', true, 'plot_reliabilities', plot_reliabilities, 'show_legend', true, 'legend_loc', 'southwest');
ylabel('prop. report "cat. 1"', 'fontsize', xy_label_fontsize)
end

if B
tight_subplot(nRows, nCols, 3,4, gutter, margins);
h=crazyplot(real_data, model, 'B', 'all', 'c_s', 'Chat', true, leftshift2, captionletterpropleft2, 'label_x', false, 'label_y', false,  'plot_reliabilities', plot_reliabilities);
end


if A
% resp vs orientation for all reliabilities
tight_subplot(nRows, nCols, 4,3, gutter, margins);
crazyplot(real_data, model, 'A', 'all', 'c_s', 'resp', true, leftshift1, captionletterpropleft1, 'label_x', true, 'label_y', true, 'plot_reliabilities', plot_reliabilities, 'nRespSquares', 6)
yl=ylabel('mean button press', 'fontsize', xy_label_fontsize)
ylpos = get(yl, 'position');
set(yl,'position',ylpos-[1 0 0]);

end

if B
tight_subplot(nRows, nCols, 4,4, gutter, margins);
crazyplot(real_data, model, 'B', 'all', 'c_s', 'resp', true, leftshift2, captionletterpropleft2, 'label_x', true, 'label_y', false, 'plot_reliabilities', plot_reliabilities, 'nRespSquares', 6)
end

%%

    function h=crazyplot(real_data, fake_data, task, trial_type, x, y, label_axes, left_shift, prop_left, varargin)
        if ~all(get(gca, 'ylim')==[0 1])
            % assume custom limits were set
            custom_ylims = true;
            ylimit = get(gca, 'ylim');
        else
            custom_ylims = false;
        end
        
        if ~isempty(fake_data)
            hold on
            h{2}=single_dataset_plot(fake_data.(task).sumstats.(trial_type).(x), y, x, ...
                'fake_data', true, 'group_plot', true, 's_labels', s_labels,...
                'task', task, 'tick_label_fontsize', tick_label_fontsize, 'xy_label_fontsize', xy_label_fontsize,...
                'legend_fontsize', legend_fontsize,...
                varargin{:});
            line_through_errorbars = false;
        else
            line_through_errorbars = true;
        end
        
        h{1}=single_dataset_plot(real_data.(task).sumstats.(trial_type).(x), y, x, ...
            'fake_data', false, 'group_plot', true, 's_labels', s_labels,...
            'task', task, 'plot_connecting_line', line_through_errorbars,...
            'tick_label_fontsize', tick_label_fontsize, 'xy_label_fontsize', xy_label_fontsize,...
            'legend_fontsize', legend_fontsize,...
            varargin{:});
        if custom_ylims
            ylim(ylimit)
        end
        
        if label_axes
            letter = axeslabel(letter, 'letter_size', axes_label_fontsize', 'prop_left', prop_left, 'prop_above', .15);
        end
        
        set(gca, 'position', get(gca, 'position')-[left_shift 0 0 0])
                    
    end


end