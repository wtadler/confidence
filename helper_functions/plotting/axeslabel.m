function nextletter = axeslabel(letter, varargin)

prop_left = .1;
prop_above = .11;
letter_size = 14;
assignopts(who, varargin);

if isnumeric(letter)
    nextletter = letter+1;
    letter = char(letter+96);
else
    nextletter = uint8(letter)-95;
end

xl=get(gca,'xlim');
xrange = diff(xl);
yl=get(gca,'ylim');
yrange = diff(yl);

if strcmp(get(gca, 'xdir'), 'reverse')
    xside = 2;
    xsign = -1;
else
    xside = 1;
    xsign = 1;
end

if strcmp(get(gca, 'ydir'), 'reverse')
    yside = 1;
    ysign = -1;
else
    yside = 2;
    ysign = 1;
end

t=text(xl(xside)-xsign*xrange*prop_left, yl(yside)+ysign*yrange*prop_above, letter);
set(t, 'verticalalignment', 'top', 'horizontalalignment', 'right', 'fontweight', 'bold', 'fontsize', letter_size);
end