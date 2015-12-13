function single_dataset_plot(binned_stats, stat_name, varargin)

len = size(binned_stats.mean.(stat_name), 2);
nReliabilities = size(binned_stats.mean.(stat_name), 1);
plot_reliabilities = [];
hhh = hot;
colors = hhh(round(linspace(1, 40, nReliabilities)),:);
linewidth = 2;
symmetrify = false;
fill_alpha = .5;
fake_data = false;
group_plot = false;
errorbarwidth = 1.7; % arbitrary unit
assignopts(who, varargin);

if isempty(plot_reliabilities)
    plot_reliabilities = 1:nReliabilities;
end 

for c = plot_reliabilities
    color = colors(c,:);
    
    m = binned_stats.mean.(stat_name)(c, :);
    
    if ~group_plot
        if fake_data
            errorbarheight = binned_stats.std.(stat_name)(c, :); % is this right?
        else
            errorbarheight = binned_stats.std.(stat_name)(c, :);
        end 
    else
        if fake_data
            errorbarheight = binned_stats.std.(stat_name)(c, :);
        else
            errorbarheight = binned_stats.edgar_sem.(stat_name)(c, :); % this is hardly diff than sem
        end
    end
    
    if symmetrify
        m(1:ceil(len/2)-1) = fliplr(m(ceil(len/2)+1:end));
        errorbarheight(1:ceil(len/2)-1) = fliplr(errorbarheight(ceil(len/2)+1:end));
    end
        
    if ~fake_data
        % errorbar is stupid. to customize width, have to plot a dummy point, with no connecting line. and then plot a line.
        dummy_point = len*errorbarwidth;
        errorbar([dummy_point 1:len], [-100 m], [0 errorbarheight], '.', 'linewidth', linewidth, 'color', color)
        hold on
        plot(1:len, m, '-', 'linewidth', linewidth, 'color', color);
    else
        x = [1:len fliplr(1:len)];
        y = [m + errorbarheight, fliplr(m - errorbarheight)];
        f = fill(x, y, color);
        set(f, 'edgecolor', 'none', 'facealpha', fill_alpha);
    end
        
    hold on
end

yl.tf = [.3 1];
yl.g  = [1 4];
yl.Chat = [0 1];
yl.resp = [1 8];
yl.rt = [.3 4];
yl.proportion = [0 .5];

set(gca, 'box', 'off', 'tickdir', 'out', 'ylim', yl.(stat_name));