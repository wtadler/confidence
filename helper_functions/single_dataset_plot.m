function single_dataset_plot(binned_stats, stat_name, marginalized_over_s, xticklabels, ylims, varargin)
% TO ADD: smart default values for xticklabels and ylims.

if ~marginalized_over_s
    nBins = size(binned_stats.mean.(stat_name), 2);
    nReliabilities = size(binned_stats.mean.(stat_name), 1);
else
    nReliabilities = length(binned_stats.mean_marg_over_s.(stat_name));
end
plot_reliabilities = [];
hhh = hot;
colors = hhh(round(linspace(1, 40, nReliabilities)),:);
ticklength = .025;
linewidth = 2;
symmetrify = false;
fill_instead_of_errorbar = false;
    fill_alpha = .7;
fake_datasets = false;
assignopts(who, varargin);

if isempty(plot_reliabilities)
    plot_reliabilities = 1:nReliabilities;
end 


for c = plot_reliabilities
    color = colors(c,:);
    
    if ~marginalized_over_s
        m = binned_stats.mean.(stat_name)(c, :);
        if ~fake_datasets && (strcmp(stat_name, 'tf') || strcmp(stat_name, 'Chat')) 
            try
                errorbarheight = binned_stats.std_beta_dist.(stat_name)(c, :);
            catch
                errorbarheight = binned_stats.std.(stat_name)(c, :);
            end
        elseif ~fake_datasets
            errorbarheight = binned_stats.sem.(stat_name)(c, :); % maybe this line should be edgar_sem
        elseif fake_datasets
            errorbarheight = binned_stats.std.(stat_name)(c, :);
        end
        
        if symmetrify
            m(1:ceil(nBins/2)-1) = fliplr(m(ceil(nBins/2)+1:end));
            errorbarheight(1:ceil(nBins/2)-1) = fliplr(errorbarheight(ceil(nBins/2)+1:end));
        end
        
        if ~fill_instead_of_errorbar
            errorbar(1:nBins, m, errorbarheight, 'linewidth', linewidth, 'color', color)
        else
            x = [1:nBins fliplr(1:nBins)];
            y = [m + errorbarheight, fliplr(m - errorbarheight)];
            f = fill(x, y, color);
            set(f, 'edgecolor', 'none', 'facealpha', fill_alpha);
        end
        
    else
        m = binned_stats.mean_marg_over_s.(stat_name)(c);
        if ~strcmp(stat_name, 'tf') && ~strcmp(stat_name, 'Chat')
            errorbarheight = binned_stats.sem_marg_over_s.(stat_name)(c);
        else
            errorbarheight = binned_stats.std_beta_dist_over_s.(stat_name)(c);
        end
        
        errorbarwidth = .5;
        
        dummy_point = xticklabels(c) + errorbarwidth * 50;
        
        if ~fill_instead_of_errorbar
            errorbar([dummy_point xticklabels(c)], [0 m], [0 errorbarheight], '.', 'linewidth', linewidth, 'color', color)
        else
            boxwidth = errorbarwidth * .65;
            x = [xticklabels(c) - boxwidth, xticklabels(c) + boxwidth];
            x = [x fliplr(x)];
            y = [m - errorbarheight, m - errorbarheight, m + errorbarheight, m + errorbarheight];
            f = fill(x, y, color);
            set(f, 'edgecolor', 'none', 'facealpha', fill_alpha);
        end
    end
    
    hold on
    
    % axes stuff for every plot
    
    set(gca,'box', 'off', 'ylim', ylims, 'ticklength', [ticklength ticklength], 'tickdir','out', 'xtick', xticklabels, 'xticklabel', '', 'yticklabel', '')
    if ~marginalized_over_s
        xlim([0 nBins+1])
    else
        xlim([xticklabels(1)-.5 xticklabels(end)+.5])
    end
end