function plot_logposterior_over_samples(logposterior, varargin)

load default_colororder
colors = default_colororder;
linewidth = 1.5;
true_logposterior = [];
show_legend = true;
show_labels = true;
assignopts(who, varargin);

chains = [];
for c = 1:length(logposterior)
    if ~isempty(logposterior{c})
        chains = [chains c];
    end
end
nChains = length(logposterior);

for c = chains
    color = colors(c,:);
    plot(logposterior{c},'color', color, 'linewidth', linewidth)
    hold on
    
    if c==chains(end)
        xl=get(gca,'xlim');
        if ~isempty(true_logposterior)
            plot(xl, [true_logposterior true_logposterior],'k-')
        end
        ylim([min(vertcat(logposterior{:})) max(vertcat(logposterior{:}))]);
        xlabel('sample')
        ylabel('log posterior')
        set(gca, 'box','off','tickdir','out')
        
        if show_legend
            l = legend(strread(num2str(chains),'%s'));
            set(l,'box','off','location','best');
        end
        
        if ~show_labels
            set(gca, 'visible','off','xtick', [], 'ytick', [], 'xlabel', [], 'ylabel', [])
        end
    end
end