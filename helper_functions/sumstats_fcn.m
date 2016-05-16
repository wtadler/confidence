function sumstats = sumstats_fcn(data, varargin)
bootstrap = true;
% define default
% decb_analysis = 0;
% assignopts(who,varargin);

% g_exists  = isfield(data(1).raw, 'g');
% rt_exists = isfield(data(1).raw, 'rt');
trial_types = setdiff(fieldnames(data(1).stats), 'sig_levels');
slices = setdiff(fieldnames(data(1).stats.(trial_types{1})), {'index', 'subindex_by_c'});

% fields = {'bin_counts','percent_correct','Chat1_prop','g_mean','resp_mean'}
fields = {'tf','resp','g','Chat', 'rt'};%'rt'
% eventually, move all the if statement checks to the beginning here
nBootstrapSamples = 1e3;
bootstrapConfInterval = .95;
assignopts(who,varargin);

for type = 1 : length(trial_types)
    for slice = 1:length(slices)
        for f = 1:length(fields)
            
            % initialize
            st.Mean = [];
            st.STD = [];
            st.bin_counts = [];

            % append
            for dataset = 1 : length(data); % concatenate along 3rd dim for other datasets/subjects
                st.Mean        = cat(3, st.Mean,       data(dataset).stats.(trial_types{type}).(slices{slice}).mean.(fields{f}));
                st.STD         = cat(3, st.STD,        data(dataset).stats.(trial_types{type}).(slices{slice}).std.(fields{f}));
                st.bin_counts  = cat(3, st.bin_counts, data(dataset).stats.(trial_types{type}).(slices{slice}).bin_counts);
            end
            
            % sum, mean, SEM, edgar SEM over subjects
            st.mean = nanmean(st.Mean, 3); % have to use nanmean and nanstd because there are missing data. for instance, some subjects never say high confidence in certain bins.
            st.std = nanstd(st.Mean, 0, 3);
            st.trial_weighted_mean = nansum(st.bin_counts.*st.Mean, 3) ./ nansum(st.bin_counts, 3);
            
%             nDatasets = length(data);
            nDatasets = sum(st.bin_counts~=0, 3); % how many datasets have data in each bin?
            st.bin_counts(st.bin_counts==0) = nan;
            
            st.sem = st.std ./ sqrt(nDatasets);
            edgar_sem = @(var) sqrt(var ./ nDatasets + nanmean(st.STD.^2./(bsxfun(@times, nDatasets, st.bin_counts)), 3));
            
            st.edgar_sem = edgar_sem(st.std.^2);%sqrt(st.std.^2 ./ nDatasets + ...
%                 nanmean(st.STD.^2./(bsxfun(@times, nDatasets, st.bin_counts)), 3)); % January 2015
            
            varpop = 1./(nDatasets - 1) .* nansum(bsxfun(@minus, st.Mean, st.mean).^2, 3) - (nDatasets - 1)./(nDatasets.^2).*nansum(st.STD.^2./st.bin_counts,3);
            
            st.edgar_sem2 = edgar_sem(varpop);
            
            if bootstrap
                dim = size(st.mean);
                bootstat = bootstrp(nBootstrapSamples, @nanmean, permute(st.Mean, [3 1 2]));
                st.mean = mean(bootstat);
                st.mean = reshape(st.mean, dim(1), dim(2));
                
                quantiles = quantile(bootstat, [.5 - bootstrapConfInterval/2, .5 + bootstrapConfInterval/2]);
                quantiles = reshape(quantiles', dim(1), dim(2), 2);
                st.CI = quantiles;
            end
               
            
            stats = fieldnames(st);
            for s = 1:length(stats)
                sumstats.(trial_types{type}).(slices{slice}).(stats{s}).(fields{f}) = st.(stats{s});
            end
        end
        
    end
end