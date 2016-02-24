function plot_logposterior_hist(logposterior, varargin)

load default_colororder
colors = [default_colororder; default_colororder; default_colororder];
linewidth = 1.5;
true_logposterior = [];
nBinEdges = 100;
show_legend = true;
show_labels = true;
lp_lim = 0;
assignopts(who, varargin);

chains = [];
for c = 1:length(logposterior)
    if ~isempty(logposterior{c})
        chains = [chains c];
    end
end
nChains = length(logposterior);

all_logposteriors = vertcat(logposterior{:});
logposterior_lim = [min(all_logposteriors) max(all_logposteriors)];
lp_bins = linspace(logposterior_lim(1), logposterior_lim(2), nBinEdges);

for c = chains
    color = colors(c, :);
    
    bin_count=histc(logposterior{c}, lp_bins);
    bin_count = bin_count(1:end-1);
    bin_centers = lp_bins(1:end-1)+diff(lp_bins)/2;
    
    normalized_logposterior_p = bin_count/sum(bin_count);
    plot(bin_centers,normalized_logposterior_p,'color',color,'linewidth',linewidth)
    hold on
    
    lp_lim = max(lp_lim,max(normalized_logposterior_p));
    ylim([0 lp_lim])
    
    if c==chains(end)
        xlabel('log posterior')
        ylabel('p(log posterior)')
        yl=get(gca,'ylim');
        if ~isempty(true_logposterior)
            plot([true_logposterior true_logposterior],yl,'k-')
        end
        xlim([min(vertcat(logposterior{:})) max(vertcat(logposterior{:}))]);
        set(gca,'box','off','tickdir','out')
        
        if show_legend
            l = legend(strread(num2str(chains),'%s'));
            set(l,'box','off','location','best');
        end
        
        if ~show_labels
            set(gca, 'visible','off','xtick', [], 'ytick', [], 'xlabel', [], 'ylabel', [])
        end

    end

end