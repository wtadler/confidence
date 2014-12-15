function data = old_MCS_analysis_fcn(all,data,bins,i,varargin)

% define defaults
binstyle='quantile';
o_boundary=25;
o_axis=[]; % this will cause an error if you use a 'defined' binstyle but fail to define the bins.
conflevels = 4;

assignopts(who,varargin);

% this function gets called in big for loops that run for each sigma.

% find indices for this sig level
data(1).indices = 1 : length(all.C); % all trial indices for the particular sig level
data(2).indices = find(all.tf == 1); % correct trial indices
data(3).indices = find(all.tf == 0); % incorrect trial indices
data(4).indices = find(all.Chat == 1); % Chat1 trial indices
data(5).indices = find(all.Chat == 2); % Chat2 trial indices
data(6).indices = find(all.C == 1); % C1 trial indices
data(7).indices = find(all.C == 2); % C2 trial indices

for t=1:length(data) % for all types of trials (all, correct, incorrect, Chat==1, Chat==2 trials),
    % choose bin style. o_axis will be for plotting.
    
    [data(t).o_bins, data(t).o_axis(i,:)] = bin_generator(bins,'binstyle',binstyle);
    
    %I think that all this stuff is junk, and is now in the function.
%     elseif strcmp(binstyle,'log')
%         o_axis_half = logspace(.03,log10(o_boundary+1),(bins)/2)-1;
%         data(t).o_axis(i,:) = [-fliplr(o_axis_half) o_axis_half]; % This is where we will plot the bins
%     elseif strcmp(binstyle,'lin')
%         data(t).o_axis(i,:) = linspace(-o_boundary,o_boundary,bins); % this is not actually logarithmic, duh
%     elseif strcmp(binstyle,'defined')
%         data(t).o_axis(i,:) = o_axis;
%         bins = length(o_axis);
%     end    
% 
%     % find halfway point between bins, for histc.
%     if ~strcmp(binstyle,'quantile') % this is a dirty fix because I changed how quantile axes/bins are made.
%         data(t).o_bins = data(t).o_axis(i,1:end-1) + diff(data(t).o_axis(i,:))/2; % find the average point between the axis ticks, use for binning only.
%     end
    
    data(t).g   = all.g(data(t).indices);
    data(t).tf  = all.tf(data(t).indices);
    data(t).resp{i}= all.g(data(t).indices) + conflevels + (all.Chat(data(t).indices)-2).*(2*all.g(data(t).indices)-1);
    % this works only for our current setup with the conf levels laid out as they are. this might not make sense for the model predictions yet
    % the {i} thing is a real hack. have to come up with a more standard
    % way to store this data for multiple sigmas. rows aren't good because
    % of indexing problems.
    
    if isfield(all,'rt'); % this will only execute with real data. might be worth modeling RT data, though?
        data(t).rt  = all.rt(data(t).indices);
        data(t).Chat = all.Chat(data(t).indices);
    end
    
    
    [n, data(t).bin_numbers]  = histc(all.s(data(t).indices), [-Inf, data(t).o_bins, Inf]); % find the bin IDs and numbers
    data(t).bin_counts(i,:) = n(1:end-1);
    
    for j=1:bins; % for each orientation bin, get average and std for all, correct, incorrect, Chat==1, Chat==2 trials
        data(t).g_mean(i,j)          = mean(data(t).g (data(t).bin_numbers==j));
        data(t).g_std(i,j)           = std (data(t).g (data(t).bin_numbers==j));
        data(t).g_sem(i,j)           = data(t).g_std(i,j) ./ sqrt(data(t).bin_counts(i,j));
        data(t).percent_correct(i,j) = mean(data(t).tf(data(t).bin_numbers==j));
        if isfield(all,'rt');
            data(t).rt_mean(i,j)     = mean(data(t).rt(data(t).bin_numbers==j));
            data(t).Chat_mean(i,j)   = mean(data(t).Chat(data(t).bin_numbers==j)); %why did i put this in this if? computer has Chat too
            data(t).Chat1_prop(i,j)  = 2 - data(t).Chat_mean(i,j); %This is equivalent to the proportion of Chat1 reports. I did the proof.
            %sum(data(t).Chat(data(t).bin_numbers==j) == 1) / length(data(t).Chat(data(t).bin_numbers==j)); % 2nd term should just be bins...
        end
    end
    %Chat_mean(i,j) = mean(data(1).Chat(data(1).bin_numbers==j));
    
    [data(t).g_mean_sort, data(t).sort_index] = sort(data(t).g_mean,2); % this is for the mean vs std graphs. not dep on i? get out of loop?
    
    % sort according to the sort index of <g>.
    data(t).g_std_sort(i,:)           = data(t).g_std           (i,data(t).sort_index(i,:));
    data(t).percent_correct_sort(i,:) = data(t).percent_correct (i,data(t).sort_index(i,:));
    if isfield(all,'rt');
        data(t).rt_sort(i,:)          = data(t).rt_mean         (i,data(t).sort_index(i,:));
        data(t).Chat_sort(i,:)        = data(t).Chat_mean       (i,data(t).sort_index(i,:));
    end
end