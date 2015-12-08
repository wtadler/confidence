function sumstats = sumstats_fcn(data, varargin)

% takes

% define default
% decb_analysis = 0;
% assignopts(who,varargin);

% g_exists  = isfield(data(1).raw, 'g');
% rt_exists = isfield(data(1).raw, 'rt');
trial_types = setdiff(fieldnames(data(1).stats), 'sig_levels');
slices = setdiff(fieldnames(data(1).stats.all), 'index');

% fields = {'bin_counts','percent_correct','Chat1_prop','g_mean','resp_mean'}
fields = {'tf','resp','g','Chat', 'rt'};%'rt'
% eventually, move all the if statement checks to the beginning here
assignopts(who,varargin);

for type = 1 : length(trial_types)
    for slice = 1:length(slices)
        for f = 1:length(fields)
            
            % initialize
            Mean = [];
            STD = [];
            bin_counts = [];
            
            % append
            for dataset = 1 : length(data); % concatenate along 3rd dim for other datasets/subjects
                Mean        = cat(3, Mean,       data(dataset).stats.(trial_types{type}).(slices{slice}).mean.(fields{f}));
                STD         = cat(3, STD,        data(dataset).stats.(trial_types{type}).(slices{slice}).std.(fields{f}));
                bin_counts  = cat(3, bin_counts, data(dataset).stats.(trial_types{type}).(slices{slice}).bin_counts);
            end
            
            % sum, mean, SEM, edgar SEM over subjects
            st.mean = nanmean(Mean, 3); % have to use nanmean and nanstd because there are missing data. for instance, some subjects never say high confidence in certain bins.
            st.std = nanstd(Mean, 0, 3);
%             nDatasets = length(data);
            nDatasets = sum(bin_counts~=0, 3); % how many datasets have data in each bin?
            bin_counts(bin_counts==0) = nan;
            
            st.sem = st.std ./ sqrt(nDatasets);
            st.edgar_sem = sqrt(st.std.^2 ./ nDatasets + ...
                nanmean(STD.^2./(bsxfun(@times, nDatasets, bin_counts)), 3));
            
            stats = fieldnames(st);
            for s = 1:length(stats)
                sumstats.(trial_types{type}).(slices{slice}).(stats{s}).(fields{f}) = st.(stats{s});
            end
        end
        
    end
end