function handle=single_dataset_plot(binned_stats, y_name, x_name, varargin)
% plot data or fits in a "smart" way. adjusts ylim and labels according to y_name and x_name.
len = size(binned_stats.mean.(y_name), 2);
nReliabilities = size(binned_stats.mean.(y_name), 1);
plot_reliabilities = [];
hhh = hot;
colors = hhh(round(linspace(1, 40, nReliabilities)),:);
linewidth = 2;
symmetrify = false;
fill_alpha = .5;
fake_data = false;
group_plot = false;
errorbarwidth = 1.7; % arbitrary unit
label_x = true;
label_y = true;
attention_task = false;
task = 'A';
s_labels = -8:2:8;
resp_square_offset = .05;
plot_connecting_line = true;
assignopts(who, varargin);

if isempty(plot_reliabilities)
    plot_reliabilities = 1:nReliabilities;
end 

for c = plot_reliabilities
    color = colors(c,:);
    
    if ~strcmp(y_name, 'proportion') && isfield(binned_stats, 'trial_weighted_mean')
        m = binned_stats.trial_weighted_mean.(y_name)(c, :);
    else
        m = binned_stats.mean.(y_name)(c, :);
    end
    
    if group_plot
        if fake_data
            errorbarheight = binned_stats.std.(y_name)(c, :); % std of means of fake group datasets. very close to .mean_sem.
        else % real data
            errorbarheight = binned_stats.edgar_sem2.(y_name)(c, :); % this is hardly diff than sem
        end        
        
    else % individual data
        if fake_data
            errorbarheight = binned_stats.std.(y_name)(c, :); % should this be std or edgar_sem?
        else % real data
            errorbarheight = binned_stats.std.(y_name)(c, :); % this is hardly diff than sem
        end
    end
    
    if symmetrify
        m(1:ceil(len/2)-1) = fliplr(m(ceil(len/2)+1:end));
        errorbarheight(1:ceil(len/2)-1) = fliplr(errorbarheight(ceil(len/2)+1:end));
    end
        
    if ~fake_data
        % errorbar is stupid. to customize width, have to plot a dummy point, with no connecting line. and then plot a line.
        
        dummy_point = 1+len*errorbarwidth; % this method makes all bars equally wide, regardless of how many points there are.
%         dummy_point = 1 + errorbarwidth; % this makes the total width of the bars constant, but requires that errorbarwidth be higher
        
        errorbar([1:len dummy_point], [m -100], [errorbarheight 0], 'marker', 'none', 'linestyle', 'none', 'linewidth', linewidth, 'color', color)
        hold on
        handle(c)=plot(1:len, m, '-', 'linewidth', linewidth, 'color', color);
        if ~plot_connecting_line
            set(handle(c), 'visible', 'off') % this is weird, but have to do it to show up in legend
        end
    else
        x = [1:len fliplr(1:len)];
        y = [m + errorbarheight, fliplr(m - errorbarheight)];
        handle(c) = fill(x, y, color);
        set(handle(c), 'edgecolor', 'none', 'facealpha', fill_alpha);
    end
        
    hold on
end

yl.tf = [.3 1];
yt.tf = [.25:.25:1];
yl.g  = [1 4];
yt.g = 1:4;
yl.Chat = [0 1];
yt.Chat = 0:.25:1;
yl.resp = [1 8];
yt.resp = 1:8;
yl.rt = [0 4];
yt.rt = 0:4;
yl.proportion = [0 .5];
yt.proportion = 0:.1:.5;

set(gca, 'box', 'off',...
    'tickdir', 'out', 'ylim', yl.(y_name),...
    'ytick', yt.(y_name),...
    'xlim', [.5 len+.5], 'ticklength', [.018 .018],...
    'yticklabel', '', 'xticklabel', '', 'color', 'none');

if label_y
    if ~strcmp(y_name, 'resp')
        set(gca, 'yticklabelmode', 'auto')
    else
        set(gca, 'clipping', 'off')
        % blue to red colormap
        map = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
        button_colors = map.button_colors;

        for r = 1:8
            plot(.5-resp_square_offset*len, r, 'square', 'markerfacecolor', button_colors(r,:), 'markersize', 12, 'markeredgecolor','none')
        end
    end
end

if any(strcmp(x_name, {'s', 'c_s'}))
    [~, centers] = bin_generator(len, 'task', task);

    set(gca, 'xtick', interp1(centers, 1:len, s_labels));
else
    set(gca, 'xtick', 1:len)
end

if label_x
    set(gca, 'xticklabelmode', 'auto')
    switch x_name
        case {'g', 'c_g'}
            xlabel('confidence');
        case {'resp', 'c_resp'}
            xlabel('button press');
        case {'Chat', 'c_Chat'}
            xlabel('cat. choice');
        case 'c'
            if attention_task
                xlabel('cue validity')
            else
                xlabel('reliability')
            end
            xtl = cell(1,len);
            xtl{1} = 'high';
            xtl{end} = 'low';
            set(gca, 'xticklabel', xtl);
        case {'s', 'c_s'}
            if symmetrify
                xlabel('abs. orientation')
                set(gca, 'xticklabel', abs(s_labels))
            else
                xlabel('orientation')
                set(gca, 'xticklabel', s_labels)
            end
            
    end
end

if strcmp(x_name, 'c')
    set(gca, 'xdir', 'reverse')
end

switch y_name
    case {'tf','Chat'}
        plot_horizontal_line(.5, '-')
    case 'resp'
        set(gca, 'ydir', 'reverse')
        plot_horizontal_line(4.5, '-')
end