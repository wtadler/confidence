function ah = show_data(varargin)

root_datadir = '/Users/will/Google Drive/Will - Confidence/Data/attention1';
dep_vars = {'tf',       'g',        'Chat',     'resp',     'rt'};
nBins = 7; % make this odd
% plot_error_bars = true; % could eventually add functionality to just show means. or to not show means
symmetrify = false;
marg_over_s = false; %marginalize over s to just show effects of reliability
tasks = {'B'};
yaxis = 'depvar'; % 'depvar' or 'task' or 'model';
linewidth = 2;
ticklength = .018;
errorbarwidth = .5; % matlab errorbar is silly. errorbar width can't be set, is 2% of total range. so we make a dummy point. only applies to marg_over_s case
gutter = [.0175 .025];
margins = [.06 .01 .04 .04]; % L R B T
models = [];
plot_reliabilities = [];
show_legend = false;
real_sumstats = [];
stagger_titles = false;
assignopts(who, varargin);

% blue to red colormap
map = load('/Users/will/Google Drive/MATLAB/utilities/MyColorMaps.mat')
map = map.confchoicemap;
button_colors = map(round(linspace(1,256,8)),:);
datadir = check_datadir(root_datadir);

% if nTasks > 1
%     if ~all(strcmp(dep_vars, 'resp'))
%         warning('only showing responses, since you requested multiple tasks')
%     end
%     dep_vars = {'resp', 'resp'};
%     nRows = 2;
% else
%     nRows = length(dep_vars);
% end

if ~isempty(models)
    show_fits = true;
    nModels = length(models);
else
    show_fits = false;
end

switch yaxis
    case 'depvar'
        nRows = length(dep_vars);
        nFigs = length(tasks); % should be length(tasks)*length(models)?
    case 'task'
        if ~all(strcmp(dep_vars, 'resp'))
            warning('only showing responses, since you requested multiple tasks')
        end
        dep_vars = {'resp', 'resp'};
        nRows = 2;
        tasks = {'A', 'B'};
        nFigs = 1;
    case 'model'
        if ~all(strcmp(dep_vars, 'resp'))
            warning('only showing responses, since you requested multiple models')
        end
        dep_vars = repmat(dep_vars,1,nModels);
        nRows = nModels;
        nFigs = length(tasks);
end

nTasks = length(tasks);

ylims = [];
for dv = 1:length(dep_vars)
    if strcmp(dep_vars{dv}, 'tf')
        ylims = [ylims;.5 1];
    elseif strcmp(dep_vars{dv}, 'g')
        ylims = [ylims;1 4];
    elseif strcmp(dep_vars{dv}, 'Chat')
        ylims = [ylims;0 1];
    elseif strcmp(dep_vars{dv}, 'resp')
        ylims = [ylims;1 8];
    elseif strcmp(dep_vars{dv}, 'rt')
        ylims = [ylims;.3 4];
    end
end


for task = 1:nTasks
    real_st.(tasks{task}) = compile_data('datadir',datadir.(tasks{task}));
end
nSubjects = length(real_st.(tasks{1}).data);

if ~isempty(real_sumstats);
    real_stats = real_sumstats;
    group_fits = true;
    nCols = nModels;
else
    group_fits = false;
    nCols = nSubjects;
end

% attention and confidence experiment
if isfield(real_st.(tasks{1}).data(1).raw, 'cue_validity') && ~isempty(real_st.(tasks{1}).data(1).raw.cue_validity)
    attention_task = true;
    nReliabilities = length(unique(real_st.(tasks{1}).data(1).raw.cue_validity_id));
    
    labels = {'valid', 'neutral', 'invalid'};
    xl = 'cue validity';
    
    colors = flipud([.7 0 0;.6 .6 .6;0 .7 0]);
    
    %confidence only experiment
else
    attention_task = false;
    nReliabilities = length(unique(real_st.(tasks{1}).data(1).raw.contrast_id));
    
    contrasts = unique(real_st.(tasks{1}).data(1).raw.contrast);
    strings = strsplit(sprintf('%.1f%% ', contrasts*100), ' ');
    labels = fliplr(strings(1:end-1));
    xl = 'contrast/eccentricity';
    
    hhh = hot(64);
    colors = hhh(round(linspace(1,40,nReliabilities)),:); % black to orange indicate high to low contrast
    
    if length(plot_reliabilities) == 3;
        colors = [30 95 47;181 172 68; 208 208 208];
        colors = kron(colors,ones(2,1))/255;
    end
end

% if strcmp(yaxis, 'task')
%     [edges.A, centers.A] = bin_generator(nBins, 'task', 'B'); % in mixed case, more important to have everything on the same axis than to have both tasks
%     edges.B = edges.A;
%     centers.B = centers.A;
% else
[edges.A, centers.A] = bin_generator(nBins, 'task', 'A');
[edges.B, centers.B] = bin_generator(nBins, 'task', 'B');
% end


for task = 1:nTasks
    if ~marg_over_s
        % tick mark placement
        
        
        
        % v4
        ori_labels.(tasks{task}) = [-8:2:8];%[-9 -7 -5:5 7 9];
        
        % v3
%         ori_labels.(tasks{task}) = unique(round(centers.(tasks{task})));
%         ori_labels.(tasks{task}) = ori_labels.(tasks{task})(2:end-1);
        
        % v2
%         maxcenter = max(abs(centers.(tasks{task})));
%         pts = floor(linspace(2, maxcenter, (nBins-1)/2));
%         ori_labels.(tasks{task}) = [-fliplr(pts) 0 pts];

        % v1
        %     ori_labels = [-14 -5 -2 0 2 5 14]; % make sure that this only has nBins entries or fewer. also has to have smaller absolute values than centers
        %     if length(ori_labels) > nBins
        %         error('more labels than bins')
        %     end
        xticklabels.(tasks{task}) = interp1(centers.(tasks{task}), 1:nBins, ori_labels.(tasks{task}));
        if symmetrify
            ori_labels.(tasks{task}) = abs(ori_labels.(tasks{task}));
        end
    elseif marg_over_s
        xticklabels.(tasks{task}) = 1:nReliabilities;
    end
end

dep_var_labels = rename_var_labels(dep_vars); % translate from variable names to something other people can understand.

for fig = 1:nFigs
    figure
    clf
    
    for col = 1:nCols
        if ~group_fits
            subject = col;
            if show_fits
                model = models(fig); % might be dependent on task too...add this feature later?
            end
        elseif group_fits
            model = models(col);
        end
        
        if ~group_fits && (strcmp(yaxis, 'depvar') || strcmp(yaxis, 'model'))
            task = fig;
            raw = real_st.(tasks{task}).data(subject).raw;
            if symmetrify
                raw.s = abs(raw.s);
            end
            real_stats.(tasks{task}) = indiv_analysis_fcn(raw, edges.(tasks{task}));
        end
        
        for row = 1:nRows
            if strcmp(yaxis, 'task')
                task = row;
                if ~group_fits
                    raw = real_st.(tasks{task}).data(subject).raw;
                    if symmetrify
                        raw.s = abs(raw.s);
                    end
                    real_stats.(tasks{task}) = indiv_analysis_fcn(raw, edges.(tasks{task}));
                end
            end
            
            ah.(tasks{task})(row, col) = tight_subplot(nRows, nCols, row, col, gutter, margins);
            
            if show_fits
                if strcmp(yaxis, 'model')
                    model = models(row);
                end
                
                if group_fits
                    single_dataset_plot(model.fake_sumstats.(tasks{task}), dep_vars{row}, marg_over_s, xticklabels.(tasks{task}), ylims(row,:), 'fill_instead_of_errorbar', true, ...
                        'symmetrify', symmetrify, 'colors', colors, 'linewidth', linewidth, 'errorbarwidth', errorbarwidth, 'ticklength', ticklength, 'fake_datasets', true, ...
                        'plot_reliabilities', plot_reliabilities)
                elseif ~group_fits
                    single_dataset_plot(model.extracted(col).fake_datasets.(tasks{task}).sumstats.all, dep_vars{row}, marg_over_s, xticklabels.(tasks{task}), ylims(row,:), 'fill_instead_of_errorbar', true, ...
                        'symmetrify', symmetrify, 'colors', colors, 'linewidth', linewidth, 'errorbarwidth', errorbarwidth, 'ticklength', ticklength, 'fake_datasets', true, ...
                        'plot_reliabilities', plot_reliabilities)
                end
            end
            
            single_dataset_plot(real_stats.(tasks{task}).all, dep_vars{row}, marg_over_s, xticklabels.(tasks{task}), ylims(row,:), 'fill_instead_of_errorbar', false, ...
                'symmetrify', symmetrify, 'colors', colors, 'linewidth', linewidth, 'errorbarwidth', errorbarwidth, 'ticklength', ticklength, 'std_beta_dist_instead_of_sem', false, ...
                'plot_reliabilities', plot_reliabilities)
            
                % x axis labels for bottom row
            if row == nRows
                if ~marg_over_s
                    if symmetrify
                        xlabel('|s|')
                    else
                        xlabel('s')
                    end
                    set(gca, 'xticklabel', ori_labels.(tasks{task}))
                elseif marg_over_s
                    xlabel(xl)
                    set(gca, 'xticklabel', labels)
                end
            end
            
            % y axis labels for left column
            if col == 1
                if ~strcmp(yaxis, 'model') || group_fits
                    yl=ylabel([dep_var_labels{row} ', Task ' tasks{task}]);
                else
                    yl=ylabel([dep_var_labels{row} ', Task ' tasks{task} ', ' rename_models(model.name)]);
                end
                
                if ~strcmp(dep_vars{row}, 'resp')
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
                    
                    ylabel_pos = get(yl, 'position');
                    set(yl, 'position', ylabel_pos-[3*square 0 0]);
                    
                end
                    
            end
            
            switch dep_vars{row}
                case {'tf','Chat'}
                    plot_halfway_line(.5)
                case 'resp'
                    set(gca, 'ydir', 'reverse')
                    plot_halfway_line(4.5)
            end
            
            % title (and maybe legend) for top row
            if row == 1
                if ~group_fits
                    t=title(real_st.(tasks{task}).data(col).name)
                elseif group_fits
                    t=title(rename_models(model.name));
                    set(gca, 'xticklabel', ori_labels.(tasks{task}))
                end
                
                if col == 1
                    if ~marg_over_s && show_legend
                        legend(labels)
                    end
                end
                if stagger_titles && mod(col,2)==0 % every other column. this needs to be after set(gca, 'ydir', 'reverse')
                    tpos = get(t, 'position');
                    yrange = ylims(row,2)-ylims(row,1);
                    set(t, 'position', tpos+[0 .04*yrange 0])
                end

            end
            set(gca,'color','none')
        end
    end
end
end

function plot_halfway_line(y)
xl = get(gca, 'xlim');
plot(xl, [y y], '-', 'color', [0 0 0], 'linewidth', 1)
end