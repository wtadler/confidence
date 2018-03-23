function h=plot_hvu_line(x, orientation, varargin)

if isempty(varargin)
    varargin = {'--', 'color', [0 0 0], 'linewidth', 1};
end

xl = get(gca, 'xlim');
yl = get(gca, 'ylim');

hold on

switch orientation
    case 'horizontal'
        h=plot(xl, [x x], varargin{:});
    case 'vertical'
        h=plot([x x], yl, varargin{:});
    case 'unity'
        lower = max([xl(1), yl(1)]);
        upper = min([xl(2), yl(2)]);
        
        h=plot([lower upper], [lower upper], varargin{:});
end