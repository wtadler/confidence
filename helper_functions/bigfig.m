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
gutter = [.045 .055];
margins = [.15 .01 .14 .06]; % L R B T
% show_legend = false;
s_labels = [-8 -4 -2 -1 0 1 2 4 8];
letter_size = 14;
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
set(gcf,'position', [100 100 250*nCols 200*nRows])
clf
letter = 1;
%%
if ~isempty(model)
    model = generate_and_analyze_fitted_data(model, tasks, 'real_data', real_data, 'nBins', nBins, 'nPlotSamples', nPlotSamples, ...
        'nFakeGroupDatasets', nFakeGroupDatasets, 'depvars', depvars, 'symmetrify', symmetrify, ...
        'group_plot', true, 'trial_types', trial_types, 'bin_types', bin_types);
    uicontrol('Style', 'text', 'String', rename_models(model.name), 'Units', 'normalized', 'Position', [0 0 .45 .05], 'BackgroundColor', 'w', 'fontsize', 18, 'fontname', 'Helvetica', 'horizontalalignment', 'left')
end

% proportion report cat. 1 vs category and reliability
if A
tight_subplot(nRows, nCols, 1,1, gutter, margins);
crazyplot(real_data, model, 'A', 'all', 'c_C', 'Chat', 'label_x', false, 'label_y', true, 'show_legend', true, 'legend_loc', 'northwest');
ylabel('prop. report "cat. 1"');
letter = axeslabel(letter);
% label Task A
yl=get(gca,'ylim');
half=yl(1)+diff(yl)/2;
text(8.3, half, 'Task A', 'horizontalalignment', 'right', 'fontweight','bold', 'fontsize', letter_size+2);
end

if B
tight_subplot(nRows, nCols, 2,1, gutter, margins);
crazyplot(real_data, model, 'B', 'all', 'c_C', 'Chat', 'label_x', true, 'label_y', true);
ylabel('prop. report "cat. 1"');
letter = axeslabel(letter);
% label Task B
yl=get(gca,'ylim');
half=yl(1)+diff(yl)/2;
text(8.3, half, 'Task B', 'horizontalalignment', 'right', 'fontweight','bold', 'fontsize', letter_size+2);
end


% mean button press vs category and reliability
if A
tight_subplot(nRows, nCols, 1,2, gutter, margins);
crazyplot(real_data, model, 'A', 'all', 'c_C', 'resp', 'label_x', false, 'label_y', true, 'nRespSquares', 6);
yl=ylabel('mean button press');
ylpos = get(yl, 'position');
set(yl,'position',ylpos+[.4 0 0]);
letter = axeslabel(letter);
end

if B
tight_subplot(nRows, nCols, 2,2, gutter, margins);
crazyplot(real_data, model, 'B', 'all', 'c_C', 'resp', 'label_x', true, 'label_y', true, 'nRespSquares', 6);
yl=ylabel('mean button press')
ylpos = get(yl, 'position');
set(yl,'position',ylpos+[.4 0 0]);
letter = axeslabel(letter);
end

if A
% histogram of confidence ratings by category
tight_subplot(nRows, nCols, 1,3, gutter, margins);
crazyplot(real_data, model, 'A', 'C1', 'g', 'proportion', 'label_x', false, 'label_y', true, 'color', map.cat1);
crazyplot(real_data, model, 'A', 'C2', 'g', 'proportion', 'label_x', false, 'label_y', true, 'color', map.cat2);
ylim([0 .6])
letter = axeslabel(letter);
ylabel('prop. of total')
end

if B
tight_subplot(nRows, nCols, 2,3, gutter, margins);
crazyplot(real_data, model, 'B', 'C1', 'g', 'proportion', 'label_x', true, 'label_y', true, 'color', map.cat1);
crazyplot(real_data, model, 'B', 'C2', 'g', 'proportion', 'label_x', true, 'label_y', true, 'color', map.cat2);
ylim([0 .6])
letter = axeslabel(letter);
ylabel('prop. of total')
end

if nTasks > 1
% proportion correct vs confidence
tight_subplot(nRows, nCols, 1,4, gutter, margins);
apos = get(gca, 'position');
delta_y = (apos(4) + gutter(2))/2;
set(gca, 'position', apos - [0 delta_y 0 0]);
a=crazyplot(real_data, model, 'A', 'all', 'g', 'tf', 'label_x', true, 'label_y', true, 'color', map.taskA);
b=crazyplot(real_data, model, 'B', 'all', 'g', 'tf', 'label_x', true, 'label_y', true, 'color', map.taskB);
ylabel('prop. correct');
ylim([.45 .93])
l=legend([a{1}(1),b{1}(1)],'Task A','Task B');
set(l,'box','off','location','northwest');
letter = axeslabel(letter);
end

if A
% mean confidence vs binned orientation conditioned on correctness
tight_subplot(nRows, nCols, 3,1, gutter, margins);
correct=crazyplot(real_data, model, 'A', 'correct', 'c', 'g', 'label_x', false, 'label_y', true, 'color', map.correct);
incorrect=crazyplot(real_data, model, 'A', 'incorrect', 'c', 'g', 'label_x', false, 'label_y', true, 'color', map.incorrect);
l=legend([correct{1}(1),incorrect{1}(1)],'correct','incorrect');
set(l,'box','off','location','northwest')
letter = axeslabel(letter);
ylabel('mean confidence')
% label Task A
yl=get(gca,'ylim');
half=yl(1)+diff(yl)/2;
text(8.3, half, 'Task A', 'horizontalalignment', 'right', 'fontweight','bold', 'fontsize', letter_size+2);

end

if B
tight_subplot(nRows, nCols, 4,1, gutter, margins);
crazyplot(real_data, model, 'B', 'correct', 'c', 'g', 'label_x', false, 'label_y', true, 'color', map.correct);
crazyplot(real_data, model, 'B', 'incorrect', 'c', 'g', 'label_x', true, 'label_y', true, 'color', map.incorrect);
letter = axeslabel(letter);
ylabel('mean confidence')
% label Task B
yl=get(gca,'ylim');
half=yl(1)+diff(yl)/2;
text(8.3, half, 'Task B', 'horizontalalignment', 'right', 'fontweight','bold', 'fontsize', letter_size+2);
end

if A
% mean confidence vs binned orientation conditioned on correctness
tight_subplot(nRows, nCols, 3,2, gutter, margins);
correct=crazyplot(real_data, model, 'A', 'correct', 's', 'g', 'label_x', false, 'label_y', true, 'color', map.correct);
incorrect=crazyplot(real_data, model, 'A', 'incorrect', 's', 'g', 'label_x', false, 'label_y', true, 'color', map.incorrect);
letter = axeslabel(letter);
end

if B
tight_subplot(nRows, nCols, 4,2, gutter, margins);
crazyplot(real_data, model, 'B', 'correct', 's', 'g', 'label_x', false, 'label_y', true, 'color', map.correct);
crazyplot(real_data, model, 'B', 'incorrect', 's', 'g', 'label_x', true, 'label_y', true, 'color', map.incorrect);
letter = axeslabel(letter);
end


if A
% choice vs orientation for all reliabilities
tight_subplot(nRows, nCols, 3,3, gutter, margins);
h=crazyplot(real_data, model, 'A', 'all', 'c_s', 'Chat', 'label_x', false, 'label_y', true, 'plot_reliabilities', plot_reliabilities, 'show_legend', true, 'legend_loc', 'southwest');
letter = axeslabel(letter);
ylabel('prop. report "cat. 1"')
end

if B
tight_subplot(nRows, nCols, 4,3, gutter, margins);
h=crazyplot(real_data, model, 'B', 'all', 'c_s', 'Chat', 'label_x', true, 'label_y', true,  'plot_reliabilities', plot_reliabilities);
letter = axeslabel(letter);
ylabel('prop. report "cat. 1"')
end


if A
% resp vs orientation for all reliabilities
tight_subplot(nRows, nCols, 3,4, gutter, margins);
crazyplot(real_data, model, 'A', 'all', 'c_s', 'resp', 'label_x', false, 'label_y', true, 'plot_reliabilities', plot_reliabilities)
letter = axeslabel(letter);
yl=ylabel('mean button press');
ylpos = get(yl, 'position');
set(yl,'position',ylpos-[.8 0 0]);
end

if B
tight_subplot(nRows, nCols, 4,4, gutter, margins);
crazyplot(real_data, model, 'B', 'all', 'c_s', 'resp', 'label_x', true, 'label_y', true, 'plot_reliabilities', plot_reliabilities)
letter = axeslabel(letter);
yl=ylabel('mean button press');
ylpos = get(yl, 'position');
set(yl,'position',ylpos-[.8 0 0]);
end

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


end