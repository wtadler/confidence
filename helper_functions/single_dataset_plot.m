function handle=single_dataset_plot(binned_stats, y_name, x_name, varargin)
% plot data or fits in a "smart" way. adjusts ylim and labels according to y_name and x_name.
plot_reliabilities = [];
colors = [];
linewidth = 2;
symmetrify = false;
fill_alpha = .5;
fake_data = false;
group_plot = false;
errorbarwidth = 1.7; % arbitrary unit
errorbarwidthForBarplot = 3.3;
barwidth = 0.5;
label_x = true;
label_y = true;
attention_task = false;
task = 'A';
s_labels = -8:2:8;
resp_square_offset = .06;
plot_bar = false;
plot_connecting_line = true;
nRespSquares = 8;
respSquareSize = 12;
show_legend = false;
legend_loc = 'northwest';
bootstrap = false;
xy_label_fontsize = 10;
legend_fontsize = 10;
tick_label_fontsize = 10;
ticklength = .02;
label_s_bin_centers = true;
assignopts(who, varargin);

if ~isempty(colors)
    input_colors = colors;
else
    input_colors = [];
end

if plot_bar
    plot_connecting_line = false;
end

if (strcmp(x_name, 'c') || ~isempty(strfind(x_name, 'c_'))) && ~strcmp(x_name, 'c_s')
    reliability_x_axis = true;
    set(gca, 'xdir', 'reverse');
    
    % transpose everything
    fields = setdiff(fieldnames(binned_stats), 'bin_counts');
    for f = 1:length(fields)
        binned_stats.(fields{f}).(y_name) = permute(binned_stats.(fields{f}).(y_name), [2 1 3]);
    end
    
    nRows = size(binned_stats.mean.(y_name), 1);
    nCols = size(binned_stats.mean.(y_name), 2);
    
    plot_rows = 1:nRows;
else
    reliability_x_axis = false;
    
    nRows = size(binned_stats.mean.(y_name), 1);
    nCols = size(binned_stats.mean.(y_name), 2);
    if ~isempty(plot_reliabilities)
        plot_rows = plot_reliabilities;
    else
        plot_rows = 1:nRows;
    end
end

try
    map = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
catch
    map = load('MyColorMaps.mat');
end
set(gca, 'xticklabelmode', 'auto')
if (strcmp(x_name, 'c') || ~isempty(strfind(x_name, 'c_'))) && ~strcmp(x_name, 'c_s')
    xtl = cell(1,nCols);
    if attention_task
        xlabel('cue validity')
        xtl{1} = 'valid';
        xtl{2} = 'neutral';
        xtl{3} = 'invalid';
    else
        xlabel('reliability')
        xtl{1} = 'highest';
        xtl{end} = 'lowest';
    end
    
    set(gca, 'xticklabel', xtl);
    if strcmp(x_name, 'c')
        colors = [0 0 0];
        show_legend = false;
    elseif strcmp(x_name, 'c_prior')
        colors = [map.cat1; .3 .3 .3; map.cat2];
        labels = {'cat. 1 prior', 'neutral prior', 'cat. 2 prior'};
    elseif strcmp(x_name, 'c_Chat')
        colors = [map.cat1; map.cat2];
        labels = {'"cat. 1"', '"cat. 2"'};
    elseif strcmp(x_name, 'c_resp')
        colors = map.button_colors;
        labels = {'high conf. cat. 1', 'high conf. cat. 2'};
    else
        colors = [map.cat1; map.cat2];
        labels = {'cat. 1', 'cat. 2'};
    end
    
elseif strcmp(x_name, 'g')
    xlabel('confidence');
    colors = rand(10,3);
elseif strcmp(x_name, 'resp')
    xlabel('button press');
    colors = [0 0 0];
    plot_rows = 1;
    show_legend = false;
elseif strcmp(x_name, 'Chat')
    xlabel('cat. choice');
    colors = rand(10,3);
elseif any(strcmp(x_name, {'s', 'c_s'}))
    [~, centers] = bin_generator(nCols, 'task', task);
    
    if label_s_bin_centers
        xticks = round(centers);
    else
        xticks = interp1(centers, 1:nCols, s_labels)
        set(gca, 'xtick', xticks);
    end
        
    if symmetrify
        xlabel('abs. orientation (°)')
        set(gca, 'xticklabel', abs(xticks))
    else
        xlabel('orientation (°)')
        set(gca, 'xticklabel', xticks)
    end
    
    if attention_task
        colors = map.attention_colors;
        labels = {'valid', 'neutral', 'invalid'};
    else
        colors = map.tan_contrast_colors;
        labels = {'highest rel.', 'lowest rel.'};
    end
    
elseif strcmp(x_name, 'C_s')
    colors = [map.cat1; map.cat2];
    labels = {'cat. 1', 'cat. 2'};
    
else
    colors = zeros(10,3);
end

if ~(any(strcmp(x_name, {'s', 'c_s'})) && ~label_s_bin_centers)
    set(gca, 'xtick', 1:nCols);
end

if ~isempty(input_colors);
    colors = input_colors;
end

if ~label_x % clear out
    xlabel('')
    set(gca, 'xticklabels', '')
end

for row = plot_rows
    color = colors(row,:);
    
    if bootstrap
        quantiles = binned_stats.CI.(y_name)(row, :, :);
        errorbarheight = diff(quantiles, [], 3)/2;
        m = mean(quantiles, 3);
    else
        if ~strcmp(y_name, 'proportion') && isfield(binned_stats, 'trial_weighted_mean')
            m = binned_stats.trial_weighted_mean.(y_name)(row, :);
        else
            m = binned_stats.mean.(y_name)(row, :);
        end
        if group_plot
            if fake_data
                errorbarheight = binned_stats.std.(y_name)(row, :); % std of means of fake group datasets. very close to .mean_sem.
            else % real data
                errorbarheight = binned_stats.sem.(y_name)(row, :); % sem or edgar_sem2 to take account of within subject variability.
            end
        else % individual data
            if fake_data || ~any(strcmp(y_name, {'g', 'resp'}))
                errorbarheight = binned_stats.std.(y_name)(row, :);
            else % integer response and real data
                warning(sprintf('using SEM for %s', y_name))
                errorbarheight = binned_stats.sem.(y_name)(row, :);
            end
        end
    end
    
    if symmetrify
        m(1:ceil(nCols/2)-1) = fliplr(m(ceil(nCols/2)+1:end));
        errorbarheight(1:ceil(nCols/2)-1) = fliplr(errorbarheight(ceil(nCols/2)+1:end));
    end
    
    if ~fake_data
        if plot_bar
            errorbarwidth = errorbarwidthForBarplot;
        end
        hold on
        
        % errorbar is stupid. to customize width, have to plot a dummy point, with no connecting line. and then plot a line.
        dummy_point = 1+nCols*errorbarwidth; % this method makes all bars equally wide, regardless of how many points there are.
        %         dummy_point = 1 + errorbarwidth; % this makes the total width of the bars constant, but requires that errorbarwidth be higher
        
        if attention_task && strcmp(x_name,'c')
            for i = 1:nCols
                errorbar([i dummy_point], [m(i) -100], [errorbarheight(i) 0], 'marker', 'none', 'linestyle', 'none', 'linewidth', linewidth, 'color', map.attention_colors(i,:))
            end
        else
            h = errorbar([1:nCols dummy_point], [m -100], [errorbarheight 0], 'marker', 'none', 'linestyle', 'none', 'linewidth', linewidth, 'color', color);
        end
        handle(row)=plot(1:nCols, m, '-', 'linewidth', linewidth, 'color', color);
        if ~plot_connecting_line
            set(handle(row), 'visible', 'off') % this is weird, but have to do it to show up in legend
        end
        
        if plot_bar
            if attention_task && strcmp(x_name,'c')
                for i = 1:nCols
%                     bar(i, m(i), 'BarWidth', barwidth, 'FaceColor', adjustHSV(map.attention_colors(i,:),.3,.8), 'EdgeColor', 'none');
                    handle(i) = bar(i, m(i), 'BarWidth', barwidth, 'FaceColor', map.attention_colors(i,:));
                end
            else
                handle = bar(1:nCols, m, 'BarWidth', barwidth, 'FaceColor', color);
            end
            set(handle, 'edgecolor', 'none', 'facealpha', fill_alpha);
        end
    else
        x = [1:nCols fliplr(1:nCols)];
        y = [m + errorbarheight, fliplr(m - errorbarheight)];
        % nans in fill will fail. set to mean
        if any(isnan(y))
            warning('empty fill bins found. filling some stuff in...');
            y(isnan(y))=nanmean(y);
        end
        handle(row) = fill(x, y, color);
        set(handle(row), 'edgecolor', 'none', 'facealpha', fill_alpha);
    end
    
    hold on
end

yl.tf = [.4 1];% [.3 1]; 
yt.tf = 0:.1:1;
yl.g  = [1 4];
yt.g = 1:4;
yl.Chat = [0 1];
yt.Chat = 0:.25:1;
%%
yl.resp = [5-nRespSquares/2, 4+nRespSquares/2];
yt.resp = (5-nRespSquares/2):(4+nRespSquares/2);
if nRespSquares ~= 8
    extra = .3; % to indicate that data continues beyond the axis
    yl.resp = [yl.resp(1)-extra yl.resp(2)+extra];
end

%%
yl.rt = [0 2.2]; %[0 4];
yt.rt = 0:.5:2; % 0:4;
yl.proportion = [0 .5];
yt.proportion = 0:.1:.5;

set(gca, 'box', 'off',...
    'tickdir', 'out', 'ylim', yl.(y_name),...
    'ytick', yt.(y_name),...
    'xlim', [.5 nCols+.5], 'ticklength', [ticklength ticklength],...
    'yticklabel', '', 'color', 'none', 'fontsize', tick_label_fontsize);

l = get(gca, 'xlabel');
set(l, 'fontsize', xy_label_fontsize);

l = get(gca, 'ylabel');
set(l, 'fontsize', xy_label_fontsize);

if label_y
    if ~strcmp(y_name, 'resp')
        set(gca, 'yticklabelmode', 'auto')
    else
        set(gca, 'clipping', 'off')
        % blue to red colormap
        try
            map = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
        catch
            map = load('MyColorMaps.mat');
        end
        if reliability_x_axis
            square_x = nCols + .5+ resp_square_offset*nCols;
        else
            square_x = .5-resp_square_offset*nCols;
        end
        
        for r = (5-nRespSquares/2):(4+nRespSquares/2) % 1:8 or 3:6 is typical
            plot(square_x, r, 'square', 'markerfacecolor', map.button_colors(r,:), 'markersize', respSquareSize, 'markeredgecolor','none')
        end
    end
end


% flip plotting order
ax = gca;
ax.Children = flipud(ax.Children);

switch y_name
    case {'tf','Chat'}
        chanceline=plot_horizontal_line(.5);
    case 'resp'
        set(gca, 'ydir', 'reverse')
        chanceline=plot_horizontal_line(4.5);
end

if exist('chanceline', 'var') && isvalid(chanceline)
    uistack(chanceline, 'top')
end

if show_legend
    if strcmp(x_name, 'c_prior') | attention_task
        [l, lobj]=legend(handle, labels);
    else
        try
            [l, lobj]=legend(handle([1 end]), labels, 'fontsize', legend_fontsize);
        catch
            [l, lobj]=legend(handle([2 end]), labels, 'fontsize', legend_fontsize);
        end
    end
    % do stuff with lobj odd-numbered line objects, with xdata, to shorten line.
    set(l, 'edgecolor', [1 1 1], 'location', legend_loc, 'box', 'off')
end