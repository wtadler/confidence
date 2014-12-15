function [contrast_values, contrast_id] = unique_contrasts(contrast,varargin)
% define defaults
flipsig = 1;
sig_levels = 6;
assignopts(who, varargin);

[contrast_values, ~, contrast_id] = unique(contrast);    % find unique contrast values
contrast_id = contrast_id'; % makes row instead of col vector
% reverse order so that sigmas go from high to low contrast
if flipsig == 1;
    contrast_id  = - contrast_id + sig_levels + 1;
    contrast_values        = fliplr(contrast_values); % should this be in or outside of the if statement?
end
