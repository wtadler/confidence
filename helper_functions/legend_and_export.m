%LEGEND_AND_EXPORT Adds legend to figure, exports, and closes.
% LEGEND_AND_EXPORT(filename,sig) adds a sig length legend to a plot with
% lines that have different sigma values. Filename extension determines the
% exported image format. For example, end filename with .pdf or .png to
% export as vector or bitmap, respectively.
%
% If plotting k decision boundary, add optional parameter pair
% 'plottingk',true to add one entry to the legend.

% Will Adler 2014

function legend_and_export(filename,sig,varargin)

% assign defaults for optional args
plottingk = false;
percent = false;
assignopts(who,varargin);


legend_array = cell(1,length(sig)); % initialize array

for leg=1:length(sig) % for each value of sig, add an entry to the legend
    if percent == false
        legend_array{leg}=['\sigma=' num2str(sig(leg))];
    elseif percent == true
        legend_array{leg}=['c=' num2str(100*sig(leg),2) '%'];
    end
end

if plottingk==true % if we are plottingk, add k to the legend.
    legend_array{length(sig)+1}='k = ±sqrt(k1/k2)';
end

legend(legend_array)

export_and_reset(filename)
end
