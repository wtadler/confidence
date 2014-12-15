function [bounds, axis, bins] = bin_generator(bins,varargin)

% binstyle='quantile' produces quantile bins and the quantile centers of those bins for the
% predicted distribution of stimuli in a Qamar-like experiment. For
% instance, bins=3 will produce a vector bounds with points at the
% following quantiles [.333 .666], for 3 bins. It will produce a vector
% axis for plotting those bins, [ .167 .5 .833].


binstyle = 'quantile'; % 'quantile', or some other stuff, or 'rt'
o_boundary = 25;
o_axis = [-6.16 -5.16 -4.16 4.16 5.16 6.16]; % to look at decision boundary
assignopts(who,varargin);

s1 = 3;
s2 = 12;
q = 1 / bins;
bounds = zeros(1, bins - 1);
axis = zeros(1, bins - 1);

if strcmp(binstyle, 'quantile')
    f  = @(b, q) .25*(2+erf(b/(s1*sqrt(2))) + erf(b/(s2*sqrt(2)))) - q; % see quantile_bin_generator.pages for explanation of the equation.

    for i = 1:bins-1;
        bounds(i)  = fzero(@(b) f(b, i*q), 0);
    end
    
    axis_points = 1 : 2 : 2 * bins - 1;
    for i = 1 : length(axis_points);
        axis(i) = fzero(@(b) f(b, axis_points(i)*q/2), 0);
    end

elseif strcmp(binstyle, 'log')
    o_axis_half = logspace(.03, log10(o_boundary + 1), (bins)/2) - 1;
    axis = [-fliplr(o_axis_half) 0 o_axis_half];

elseif strcmp(binstyle, 'lin')
    axis = linspace(-o_boundary, o_boundary, bins);

elseif strcmp(binstyle, 'defined')
    axis = o_axis;
    bins = length(o_axis); % in this case, re-define bins and output
    
elseif strcmp(binstyle, 'rt')
    axis = linspace(0.1,1.5,bins);
end



if ~strcmp(binstyle,'quantile') % bounds for all non-quantile binstyles.
    bounds = axis(1,1:end-1) + diff(axis(1, :)) / 2;
end