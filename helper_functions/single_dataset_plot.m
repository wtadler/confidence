function single_dataset_plot(binned_stats, stat_name, varargin)

len = size(binned_stats.mean.(stat_name), 2);
nReliabilities = size(binned_stats.mean.(stat_name), 1);
plot_reliabilities = [];
hhh = hot;
colors = hhh(round(linspace(1, 40, nReliabilities)),:);
linewidth = 2;
symmetrify = false;
fill_alpha = .7;
fake_data = false;
group_plot = false;
assignopts(who, varargin);

if isempty(plot_reliabilities)
    plot_reliabilities = 1:nReliabilities;
end 

for c = plot_reliabilities
    color = colors(c,:);
    
    m = binned_stats.mean.(stat_name)(c, :);
    
    if ~group_plot
        if fake_data
            errorbarheight = binned_stats.sem.(stat_name)(c, :); % is this right?
        else
            errorbarheight = binned_stats.std.(stat_name)(c, :);
        end 
    else
        if fake_data
            errorbarheight = binned_stats.sem.(stat_name)(c, :); % is this right?
        else
            errorbarheight = binned_stats.edgar_sem.(stat_name)(c, :);
        end
    end
    
    if symmetrify
            m(1:ceil(len/2)-1) = fliplr(m(ceil(len/2)+1:end));
            errorbarheight(1:ceil(len/2)-1) = fliplr(errorbarheight(ceil(len/2)+1:end));
    end
        
    if ~fake_data
        errorbar(1:len, m, errorbarheight, 'linewidth', linewidth, 'color', color)
    else
        x = [1:len fliplr(1:len)];
        y = [m + errorbarheight, fliplr(m - errorbarheight)];
        f = fill(x, y, color);
        set(f, 'edgecolor', 'none', 'facealpha', fill_alpha);
    end
    
%         errorbarwidth = .5;
%         
%         dummy_point = xticklabels(c) + errorbarwidth * 50;
%         
%         if ~fill_instead_of_errorbar
%             errorbar([dummy_point xticklabels(c)], [0 m], [0 errorbarheight], '.', 'linewidth', linewidth, 'color', color)
%         else
%             boxwidth = errorbarwidth * .65;
%             x = [xticklabels(c) - boxwidth, xticklabels(c) + boxwidth];
%             x = [x fliplr(x)];
%             y = [m - errorbarheight, m - errorbarheight, m + errorbarheight, m + errorbarheight];
%             f = fill(x, y, color);
%             set(f, 'edgecolor', 'none', 'facealpha', fill_alpha);
%         end
    
    hold on
    
    % axes stuff for every plot
    
%     set(gca,'box', 'off', 'ylim', ylim, 'ticklength', [ticklength ticklength], 'tickdir','out', 'xtick', xticklabels, 'xticklabel', '', 'yticklabel', '')
%     if ~marginalized_over_s
%         xlim([0 len+1])
%     else
%         xlim([xticklabels(1)-.5 xticklabels(end)+.5])
%     end
end