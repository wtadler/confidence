%FADEPLOT Plots a surface that varies in color according to bin
%observations.
% FADEPLOT(axis,structure,trialtype) uses the elements of structure
% structure with name trialtype. It draws a surface on axis, representing
% the mean or std. Color becomes whiter as structure(trialtype).bin_counts
% decreases.
%
% colormap uses the default plotting colors, and the plot only becomes
% whiter at the 75th percentile of structure(trialtype).bin_counts.
% Optionally change this by specifying 'falloff',x where x is a number
% between 0 and 1. If falloff=1, the plot will just be a regular plot,
% because it's easier to add a legend to a regular plot than a surface with
% varying color. Haven't quite figured this one out yet.
%
% Plots structure(trialtype).g_mean by default. Optionally specify
% 'stat','std' to plot structure(trialtype).g_std instead.

% Will Adler 2014

function fadeplot(axis,structure,trialtype,varargin)

% define defaults
falloff=.75;
stat='mean';
assignopts(who,varargin);


index=strcmp({structure.name},trialtype);
bincounts = structure(index).bin_counts(:,1:end-1);

if falloff <0 || falloff > 1
    error('falloff must be between 0 and 1.')
    
elseif falloff == 1; % If fade isn't necessary, use the easier and legend-able plot technique.
    if strcmp(stat,'mean')
        plot(axis,structure(index).g_mean)
    elseif strcmp(stat,'std')
        plot(axis,structure(index).g_std)
    end
else % if using fade
    map = fademap(falloff); % make colormap
    
    
    for i=1:6;
        if strcmp(stat,'mean')
            surface(repmat(axis,2,1),repmat(structure(index).g_mean(i,:),2,1),repmat(bincounts(i,:),2,1),'facecol','no','edgecolor','interp')
        elseif strcmp(stat,'std')
            surface(repmat(axis,2,1),repmat(structure(index).g_std(i,:),2,1),repmat(bincounts(i,:),2,1),'facecol','no','edgecolor','interp')
        end
        colormap(map{i})
        freezeColors
    end
end

if strcmp(stat,'mean') % put these in the above ifs?
    ylim([.5 1])
elseif strcmp(stat,'std')
    ylim([0 .2])
end

    function map = fademap(falloff)
        
        default_colors=get(0,'DefaultAxesColorOrder'); % get default colors
        
        falloff64 = round(falloff*64);
        
        for i=1:length(default_colors);
            map{i}=[linspaceNDim([1 1 1],default_colors(i,:),64-falloff64)';repmat(default_colors(i,:),falloff64,1)];
        end
    end


end