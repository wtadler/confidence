function ah = show_data(varargin)

root_datadir = '~/Google Drive/Will - Confidence/Data/v3_all';
dep_vars = {'tf'};%,       'g',        'Chat',     'resp',     'rt'};
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
errorbarwidth = .7; % matlab errorbar is silly. errorbar width can't be set, is 2% of total range. so we make a dummy point. only applies to marg_over_s case
gutter = [.0175 .025];
margins = [.06 .01 .06 .04]; % L R B T
models = [];
plot_reliabilities = [];
show_legend = false;
stagger_titles = false;
s_labels = -8:2:8;
assignopts(who, varargin);

if strcmp(axis.col, 'subject') % in all non-group plots, subjects are along the col axis
    group_plot = false;
else
    group_plot = true;
end

if rem(nBins, 2) == 0; nBins = nBins +1; end % make sure nBins is odd.

% blue to red colormap
map = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat')
map = map.confchoicemap;
button_colors = map(round(linspace(1,256,8)),:);
datadir = check_datadir(root_datadir);

if ~isempty(models)
    show_fits = true;
    nModels = length(models);
else
    show_fits = false;
    nModels = 0;
end

nDepvars = length(dep_vars)
nSlices = length(slices);
nTasks = length(tasks);
for task = 1:nTasks
    real_st.(tasks{task}) = compile_data('datadir',datadir.(tasks{task}));
    [edges.(tasks{task}), centers.(tasks{task})] = bin_generator(nBins, 'task', tasks{task});
    
    nSubjects = length(real_st.(tasks{task}).data);
    for dataset = 1:nSubjects
        if symmetrify
            real_st.(tasks{task}).data(dataset).raw.s = abs(real_st.(tasks{task}).data(dataset).raw.s);
        end
        real_st.(tasks{task}).data(dataset).stats = indiv_analysis_fcn(real_st.(tasks{task}).data(dataset).raw,...
            edges.(tasks{task}),'conf_levels', conf_levels,...
            'trial_types', {trial_type}, 'output_fields', dep_vars,...
            'bin_types', union(slices,means));
    end
    
    if group_plot
        real_st.(tasks{task}).sumstats = sumstats_fcn(real_st.(tasks{task}).data, ...
            'fields', dep_vars);
    end
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

ylims = nan(nDepvars,2);
for dv = 1:nDepvars
    switch dep_vars{dv}
        case 'tf'
            ylims(dv,:) = [.3 1];
        case 'g'
            ylims(dv,:) = [1 4];
        case 'Chat'
            ylims(dv,:) = [0 1];
        case 'resp'
            ylims(dv,:) = [1 8];
        case 'rt'
            ylims(dv,:) = [.3 4];
        case 'proportion'
            ylims(dv,:) = [0 .5];
    end
end

if isfield(real_st.(tasks{1}).data(1).raw, 'cue_validity') && ~isempty(real_st.(tasks{1}).data(1).raw.cue_validity)
    % attention
    
    nReliabilities = length(unique(real_st.(tasks{1}).data(1).raw.cue_validity_id));
    attention_task = true;
    colors = flipud([.7 0 0;.6 .6 .6;0 .7 0]);
    
else
    nReliabilities = length(unique(real_st.(tasks{1}).data(1).raw.contrast_id));
    attention_task = false;
    
    if isempty(plot_reliabilities); plot_reliabilities = 1:nReliabilities; end
    
    if max(plot_reliabilities) > nReliabilities; error('you requested to plot more reliabilities than there are'); end
    
    %     contrasts = unique(real_st.(tasks{1}).data(1).raw.contrast);
    %     strings = strsplit(sprintf('%.1f%% ', contrasts*100), ' ');
    %     labels = fliplr(strings(1:end-1));
    %     xl = 'contrast/eccentricity';
    
    hhh = hot(64);
    colors = hhh(round(linspace(1,40,nReliabilities)),:); % black to orange indicate high to low contrast
    
    %     if length(plot_reliabilities) == 3;
    %         colors = [30 95 47;181 172 68; 208 208 208];
    %         colors = kron(colors,ones(2,1))/255;
    %     end
end

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
            
            if symmetrify
                xlabels{slice} = '|s|';
                xticklabels{slice} = abs(s_labels);
            else
                xlabels{slice} = 's';
                xticklabels{slice} = s_labels;
            end
    end
end

ylabels = rename_var_labels(dep_vars); % translate from variable names to something other people can understand.

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
            
            if symmetrify && any(strcmp(slices{slice}, {'s', 'c_s'}))
                symmetrify_s = true;
            else
                symmetrify_s = false;
            end
            
            shortcutplot = @(data, fake_data, colors, linewidth, plot_reliabilities)...
                single_dataset_plot(data, dep_vars{depvar},...
                    'fake_data', fake_data, 'group_plot', group_plot, ...
                    'symmetrify', symmetrify_s, 'colors', colors, ...
                    'linewidth', linewidth, 'errorbarwidth', errorbarwidth,...
                    'plot_reliabilities', plot_reliabilities);
                
                if ~isempty(slices{slice})
                    fake_data = false;
                    if ~group_plot
                        data = real_st.(tasks{task}).data(subject).stats.(trial_type).(slices{slice});
                    else
                        data = real_st.(tasks{task}).sumstats.(trial_type).(slices{slice});
                    end
                    shortcutplot(data, fake_data, colors, linewidth, plot_reliabilities);
                end
                
                if show_fits
                    fake_data = true;
                    if ~group_plot
                        data = models(model).extracted(subject).fake_datasets.(tasks{task}).sumstats.(trial_type).(slices{slice});
                    else
                        data = models(model).fake_sumstats.(trial_type).(slices{slice});
                    end
                    shortcutplot(data, fake_data, colors, linewidth, plot_reliabilities);
                elseif ~isempty(means{slice})
                    fake_data = false;
                    if ~group_plot
                        data = real_st.(tasks{task}).data(subject).stats.(trial_type).(means{slice});
                    else
                        data = real_st.(tasks{task}).sumstats.(trial_type).(means{slice});
                    end
                    shortcutplot(data, fake_data, mean_color, meanlinewidth, []);
                end

                        
                        
            % x axis labels for bottom row
            if row == n.row
                xlabel(xlabels{slice})
                set(gca, 'xticklabel', xticklabels{slice})
            else
                xlabel('')
                set(gca, 'xticklabel', '')
            end
            
            % y axis labels for left column
            if col == 1
                yl=ylabel([ylabels{depvar},', Task ' tasks{task}]); % make this more flexible. use: rename_models(model.name)]);
                if ~strcmp(dep_vars{depvar}, 'resp')
                    set(gca, 'yticklabelmode', 'auto')
                else
                    set(gca, 'clipping', 'off')
                    set(gcf,'units','normalized','outerposition',[0 0 1 1])
                    %                     ar = pbaspect;
                    %                     ar = ar(2)/ar(1);
                    range_ratio = diff(get(gca,'xlim')) / diff(get(gca, 'ylim'));
                    square = .5;
                    width = square*range_ratio;
                    %                     curv = 0;
                    for r = 1:8
                        rectangle('position',[-1-width, r-square/2, width, square], 'facecolor',button_colors(r,:), 'edgecolor','none');
                    end
                end % figure out how to do this for x
            else
                set(gca, 'yticklabel', '')
            end
            
            switch dep_vars{depvar}
                case {'tf','Chat'}
                    plot_halfway_line(.5)
                case 'resp'
                    set(gca, 'ydir', 'reverse')
                    plot_halfway_line(4.5)
            end
            
            set(gca, 'ticklength', [ticklength ticklength], 'box', 'off', 'tickdir', 'out', 'xtick', xticks{task, slice}, 'ylim', ylims(depvar,:))
            
            
            % title (and maybe legend) for top row
            if row == 1
                switch axis.col
                    case 'subject'
                        t=title(real_st.(tasks{task}).data(subject).name);
                    case 'model'
                        t=title(rename_models(models(model).name));
                end
                
                if col == 1
                    if show_legend
                        legend(labels)
                        
                        if ~group_fits
                            t=title(upper(real_st.(tasks{task}).data(col).name))
                        elseif group_fits
                            t=title(rename_models(model.name));
                            set(gca, 'xticklabel', ori_labels.(tasks{task}))
                        end
                        
                        if col == 1
                            if show_legend
                                warning('add legend functionality')
                            end
                        end
                        
                        %                 if stagger_titles && mod(col,2)==0 % every other column. this needs to be after set(gca, 'ydir', 'reverse')
                        %                     tpos = get(t, 'position');
                        %                     yrange = ylims(row,2)-ylims(row,1);
                        %                     set(t, 'position', tpos+[0 .04*yrange 0])
                        %                 end
                        
                    end
                    set(gca,'color','none')
                end
            end
        end
    end
end