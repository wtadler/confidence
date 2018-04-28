function [group_mean, group_sem] = mybar(x, varargin)
cla
% just pass a 2d matrix and you'll get back a nicer bar plot that is more customizable than MATLAB's built-in bar.m.
group_gutter = .05; % percentage of total width between each group
bar_gutter = .01; % percentage of group width between each bar, and to the L/R of the side bars

% inter_group_gutter=.2;
% intra_group_gutter= 0.02;
fontname = 'Helvetica';
barnames = [];
% barname_ypos = 0; % come up with automatic way to do this
show_mean = true;
show_errorbox = true;
mark_grate_ellipse = false;
bootstrap = true;
CI = .95;
fontsize = 10;

data_lims = [min(x(:)) max(x(:))];
data_range = diff(data_lims);
yl = [data_lims(1)-.05*data_range data_lims(2)+.05*data_range];
region_color = 'b';
region_alpha = .4;
fig_orientation = 'vert';
assignopts(who, varargin)

barname_alignment = 'left'; % 'right' or 'left'
switch barname_alignment
    case 'left'
        barname_ypos = 0+diff(yl)/38;
    case 'right'
        barname_ypos = 0-diff(yl)/38;
    otherwise
        error('barname_alignment not recognized')
end

nGroups = size(x,1);
nBarsPerGroup = size(x,2);

%%
group_width = (1 - group_gutter * (nGroups-1)) / nGroups;
bar_width = (group_width - (1 + nBarsPerGroup) * bar_gutter) / nBarsPerGroup;

if mark_grate_ellipse
    warning('this grate-ellipse marking is not smart')
    grate_ellipse_order = [1 2 4 5 7 3 6 8 9 10 11];
    barnames = barnames(grate_ellipse_order);
    x = x(:, grate_ellipse_order);
    barcolor = [repmat([0 0 0],5,1); repmat([.5 .5 .5], 6, 1)];
else
    barcolor = repmat([0 0 0], nBarsPerGroup, 1);
end
%%
if bootstrap
    bootstat = bootstrp(1e4, @mean, x'); % 1e4 is sufficient
    group_quantiles = quantile(bootstat, [.5 - CI/2, .5, .5 + CI/2]);
    if size(group_quantiles, 1) == 1
        group_quantiles = group_quantiles';
    end
    group_mean = group_quantiles(2,:); % median
    group_quantiles = group_quantiles([1 3],:);

else
    group_mean = mean(x, 2);
end
group_sem = std(x, [], 2)./sqrt(nBarsPerGroup);


for g = 1:nGroups
    group_start = (g-1)*(group_width+group_gutter);
    for b = 1:nBarsPerGroup
        y = x(g,b);
        start = group_start + bar_gutter*b + bar_width*(b-1);
        %         start = g-.5*(1-inter_group_gutter) + (barwidth+intra_group_gutter)*(b-1);
        
        f=fill([start start+bar_width start+bar_width start], [0 0 y y], barcolor(b, :), 'EdgeColor', 'none');
        
        hold on
        
        % subject name
        if ~isempty(barnames)
            name = barnames{b};
            textx = start+bar_width/2;
            t = text(textx, barname_ypos, name);
            %             if length(name) <= 1
            %                 rot = 0;
            %             else
            if strcmp(fig_orientation, 'vert')
                rot = 90;
            elseif strcmp(fig_orientation, 'horz');
                rot = 0;
            end
            %             end
            set(t, 'horizontalalignment', barname_alignment, 'rot',rot, 'fontsize', fontsize, 'fontname', fontname, 'color', [.6 .6 .6],'fontweight','bold')
        end
    end
    
    
    startpt = group_start + bar_gutter;
    endpt = group_start + group_width - bar_gutter;
    if show_mean        
        plot([startpt endpt],[group_mean(g) group_mean(g)],'-','color',region_color,'linewidth',3)
    end
    if show_errorbox
        if bootstrap
            error_box = [group_quantiles(1, g) group_quantiles(1, g) group_quantiles(2, g) group_quantiles(2, g)];
        else
            error_box = [group_mean(g)-group_sem(g) group_mean(g)-group_sem(g) group_mean(g)+group_sem(g) group_mean(g)+group_sem(g)];
        end
        
        f = fill([startpt endpt endpt startpt], error_box,...
            region_color,'edgecolor','none', 'facealpha', region_alpha);
    end
end

set(gca,'box','off','tickdir','out','xtick','')
if strcmp(fig_orientation, 'horz')
    set(gca, 'view', [90 -90], 'xdir', 'reverse');
else
    set(gca, 'view', [0 90], 'xdir', 'normal');
end
xl = get(gca,'xlim');
zero_line=plot(xl,[0 0],'k-');
uistack(zero_line,'bottom')

ylim(yl)

%
% line_spacing = 20;
%
% nLines = floor(yrange/line_spacing);
%
% for i = 1:nLines
%     y = round(yl(1),-2) + i*line_spacing;
%     if y == 0
%         color = [0 0 0];
%         linewidth = line_spacing/45;
%     else
%         color = [.8 .8 .8];
%         linewidth = line_spacing/125;
%     end
%     p=plot(xl, [y y],'color', color,'linewidth', linewidth);
%     uistack(p,'bottom');
%     hold on
% end