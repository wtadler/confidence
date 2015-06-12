function [unique_values, contrast_id] = unique_contrasts(contrast, varargin)
% define defaults
flipsig = true;
assignopts(who, varargin);

[unique_values, ~, contrast_id] = unique(contrast);    % find unique contrast values
nLevels = length(unique_values);
contrast_id = contrast_id'; % makes row instead of col vector
% reverse order so that sigmas go from high to low contrast
if flipsig;
    contrast_id  = - contrast_id + nLevels + 1;
    unique_values        = fliplr(unique_values); % should this be in or outside of the if statement?
end
