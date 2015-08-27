function mybar(x, varargin)

% just pass a 2d matrix and you'll get back a nicer bar plot that is more customizable than MATLAB's built-in bar.m.

inter_group_gutter=.2;
intra_group_gutter= 0.02;
fontname = 'Helvetica Neue';
barnames = [];
show_mean = true;
mark_grate_ellipse = false;
assignopts(who, varargin)



figure
clf

nGroups = size(x,1);
nBarsPerGroup = size(x,2);

barwidth = (1 - inter_group_gutter - (nBarsPerGroup-1)*intra_group_gutter)/nBarsPerGroup;

if mark_grate_ellipse
    warning('this grate-ellipse marking is not flexible')
    grate_ellipse_order = [1 2 4 5 7 3 6 8 9 10 11];
    barnames = barnames(grate_ellipse_order);
    x = x(:, grate_ellipse_order);
    barcolor = [repmat([0 0 0],5,1); repmat([.5 .5 .5], 6, 1)];
else
    barcolor = repmat([0 0 0], nBarsPerGroup, 1);
end

for g = 1:nGroups
    for b = 1:nBarsPerGroup
        y = x(g,b);
        start = g-.5*(1-inter_group_gutter) + (barwidth+intra_group_gutter)*(b-1);

        f=fill([start start+barwidth start+barwidth start], [0 0 y y], barcolor(b, :), 'EdgeColor', 'none');
        
        hold on
        
        % subject name
        if ~isempty(barnames)
            name = barnames{b};
            if y >= 0
                color = [0 0 0];
            else
                color = [1 1 1];
            end
            textx = start+barwidth/2;
            t = text(textx, -15, name);
            set(t, 'horizontalalignment', 'right', 'rot',90, 'fontsize', 10, 'fontname', fontname, 'color', color)
        end
    end
    
    if show_mean
        group_mean = mean(x(g,:));
        
        height = 5;
        startpt = g-.5*(1-inter_group_gutter);
        endpt = startpt + (barwidth+intra_group_gutter)*(nBarsPerGroup);
        f = fill([startpt endpt endpt startpt],[group_mean-height group_mean-height group_mean+height group_mean+height],...
            'blue','edgecolor','none');
    end
end

yl = get(gca,'ylim');
xl = get(gca,'xlim');
yrange = diff(yl);

nLines = floor(yrange/100);

for i = 1:nLines
    y = round(yl(1),-2) + i*100;
    if y == 0
        color = [0 0 0];
        linewidth = 2.2;
    else
        color = [.8 .8 .8];
        linewidth = .8;
    end
    p=plot(xl, [y y],'color', color,'linewidth', linewidth);
    uistack(p,'bottom');
    hold on
end