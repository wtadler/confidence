function [stats, sorted_raw] = indiv_analysis_fcn(raw, bins, varargin)
% model data doesn't work right now with the way theory_plots generates
% fake data.
% make this parfor-ated. the bins loop is really slow

% define defaults
conf_levels = 4; % used for discrete variance
sig_levels = length(unique(raw.contrast));
data_type = 'subject';
flipsig = 1;
discrete = 0;
assignopts(who,varargin);

g_exists  = isfield(raw, 'g');
rt_exists = isfield(raw, 'rt');

if strcmp(data_type, 'model')  % this is only for the weird theory_MCS_and_plots way of looping over sigma. deprecated
    sig_levels = 1;
    raw.contrast_id = ones(1,length(raw.C));
    raw.contrast_values = 1;
    raw.rt = -ones(1,length(raw.C)); % fake rt data. better to have this then to ugly up the code below with if statements.
end

stats.all.index       = 1 : length(raw.C); % these indices just go into histc
stats.correct.index   = find(raw.tf == 1);
stats.incorrect.index = find(raw.tf == 0);

% Model data is -1 and 1, and subject data is 1 and 2.
if ismember(2, raw.Chat)
    C1=1;
    C2=2;
elseif ismember(-1, raw.Chat)
    C1=-1;
    C2=1;
end

stats.Chat1.index     = find(raw.Chat == C1);
stats.Chat2.index     = find(raw.Chat == C2);
stats.C1.index        = find(raw.C == C1);
stats.C2.index        = find(raw.C == C2);
idx = stats.Chat1.index + 1;
stats.after_Chat1.index= idx(idx<length(raw.C));
idx = stats.Chat2.index + 1;
stats.after_Chat2.index= idx(idx<length(raw.C));


trial_types = fieldnames(stats); % the field names defined above.

% cut out sig sorting thing here, moved to compile_and_analyze

for type = 1 : length(trial_types)    
    for contrast = 1 : sig_levels;
        % These vectors contain all the trials for a given contrast and
        % trial_type. intersection of two indices.
        
        % sort trials by contrast and trial type
        sorted_raw.(trial_types{type}){contrast}.C    = raw.C    (intersect(find(raw.contrast_id == contrast), stats.(trial_types{type}).index));
        sorted_raw.(trial_types{type}){contrast}.s    = raw.s    (intersect(find(raw.contrast_id == contrast), stats.(trial_types{type}).index));
        if isfield(sorted_raw.(trial_types{type}){contrast}, 'x')
            sorted_raw.(trial_types{type}){contrast}.x    = raw.x    (intersect(find(raw.contrast_id == contrast), stats.(trial_types{type}).index));
        end
        sorted_raw.(trial_types{type}){contrast}.Chat = raw.Chat (intersect(find(raw.contrast_id == contrast), stats.(trial_types{type}).index));
        sorted_raw.(trial_types{type}){contrast}.tf   = raw.tf   (intersect(find(raw.contrast_id == contrast), stats.(trial_types{type}).index));
        if g_exists
            sorted_raw.(trial_types{type}){contrast}.g    = raw.g    (intersect(find(raw.contrast_id == contrast), stats.(trial_types{type}).index));
        end
        if rt_exists
            sorted_raw.(trial_types{type}){contrast}.rt   = raw.rt   (intersect(find(raw.contrast_id == contrast), stats.(trial_types{type}).index));
        end
        
        stats.(trial_types{type}).Chat1_prop_over_c(contrast) = .5 - .5 * mean(sorted_raw.(trial_types{type}){contrast}.Chat);
        stats.(trial_types{type}).percent_correct_over_c(contrast) = mean(sorted_raw.(trial_types{type}){contrast}.tf);
        if g_exists
            for conf_level = 1 : conf_levels
                stats.(trial_types{type}).Chat1_prop_over_c_and_g     (sig_levels + 1 - contrast, conf_level) = .5 - .5 * mean(sorted_raw.(trial_types{type}){contrast}.Chat(sorted_raw.(trial_types{type}){contrast}.g == conf_level));
                stats.(trial_types{type}).percent_correct_over_c_and_g(sig_levels + 1 - contrast, conf_level) = mean(sorted_raw.(trial_types{type}){contrast}.tf(sorted_raw.(trial_types{type}){contrast}.g == conf_level));
            end
            stats.(trial_types{type}).g_mean_over_c(contrast) = mean(sorted_raw.(trial_types{type}){contrast}.g);
            for i = 1:2
                if ismember(2, raw.Chat)
                    stats.(trial_types{type}).g_mean_over_c_and_Chat(sig_levels + 1 - contrast, i) = mean(sorted_raw.(trial_types{type}){contrast}.g(sorted_raw.(trial_types{type}){contrast}.Chat == i));
                elseif ismember(-1, raw.Chat) % this is running in data_plots.m
                    stats.(trial_types{type}).g_mean_over_c_and_Chat(sig_levels + 1 - contrast, i) = mean(sorted_raw.(trial_types{type}){contrast}.g(sorted_raw.(trial_types{type}){contrast}.Chat == 2*i-3));
                end
            end
        end

        
        % bin trials by s
        [n, stats.(trial_types{type}).bin_index] = histc(sorted_raw.(trial_types{type}){contrast}.s, [-Inf, bins, Inf]);
        stats.(trial_types{type}).bin_counts (contrast,:) = n(1 : end - 1);
        
        for bin = 1 : length(bins) + 1; % calc stats for each bin over s
            if g_exists
                stats.(trial_types{type}).g_mean            (contrast,bin) =     mean(sorted_raw.(trial_types{type}){contrast}.g    (stats.(trial_types{type}).bin_index == bin));
                stats.(trial_types{type}).g_std             (contrast,bin) =     std (sorted_raw.(trial_types{type}){contrast}.g    (stats.(trial_types{type}).bin_index == bin));
                if discrete == 1
                    for g = 1 : conf_levels
                        % var(discrete random variable) = sum_g(p_g * (g - <g>)^2
                        p(g) = sum(sorted_raw.(trial_types{type}){contrast}.g (stats.(trial_types{type}).bin_index == bin) == g) ...
                            ./ length(sorted_raw.(trial_types{type}){contrast}.g (stats.(trial_types{type}).bin_index == bin) );
                        p(g) = p(g) .* (g - stats.(trial_types{type}).g_mean(contrast,bin)).^2;
                    end
                    stats.(trial_types{type}).g_discr_variance  (contrast,bin) =      sum(p(g));
                    stats.(trial_types{type}).g_discr_std       (contrast,bin) = sqrt(sum(p(g)));
                    % something like the below is probably right for binomial
                    % confidence...but this isn't very good
                    %stats.(trial_types{type}).exp_val_g         (contrast,bin) = sum(sorted_raw.(trial_types{type}){contrast}.g (stats.(trial_types{type}).bin_index == bin) == 2);
                    %stats.(trial_types{type}).binom_var_g       (contrast,bin) = stats.(trial_types{type}).exp_val_g(contrast,bin) * (1 - stats.(trial_types{type}).exp_val_g(contrast,bin) / length(sorted_raw.(trial_types{type}){contrast}.g(stats.(trial_types{type}).bin_index == bin) ));
                end
            end
                
            stats.(trial_types{type}).percent_correct   (contrast,bin) =     mean(sorted_raw.(trial_types{type}){contrast}.tf   (stats.(trial_types{type}).bin_index == bin));
            stats.(trial_types{type}).Chat1_prop        (contrast,bin) = .5 - .5 * mean(sorted_raw.(trial_types{type}){contrast}.Chat (stats.(trial_types{type}).bin_index == bin));
            if rt_exists     
                stats.(trial_types{type}).rt_mean           (contrast,bin) =     mean(sorted_raw.(trial_types{type}){contrast}.rt   (stats.(trial_types{type}).bin_index == bin));
            end
        end
        
        if rt_exists
            % bin trials by rt. MERGE THIS WITH TOP.
            bins_rt = bin_generator(length(bins)+1,'binstyle','rt'); % need new kinds of bins
            [n, stats.(trial_types{type}).bin_index] = histc(sorted_raw.(trial_types{type}){contrast}.rt, [0, bins_rt, Inf]);
            stats.(trial_types{type}).bin_counts_rt (contrast,:) = n(1 : end - 1);
            
            for bin = 1 : length(bins_rt) + 1; % calc stats for each bin over s
                if g_exists
                    stats.(trial_types{type}).g_mean_rt            (contrast,bin) =     mean(sorted_raw.(trial_types{type}){contrast}.g    (stats.(trial_types{type}).bin_index == bin));
                    stats.(trial_types{type}).g_std_rt             (contrast,bin) =     std (sorted_raw.(trial_types{type}){contrast}.g    (stats.(trial_types{type}).bin_index == bin));
                    if discrete == 1
                        for g = 1 : conf_levels
                            % var(discrete random variable) = sum_g(p_g * (g - <g>)^2
                            p(g) = sum(sorted_raw.(trial_types{type}){contrast}.g (stats.(trial_types{type}).bin_index == bin) == g) ...
                                ./ length(sorted_raw.(trial_types{type}){contrast}.g (stats.(trial_types{type}).bin_index == bin) );
                            p(g) = p(g) .* (g - stats.(trial_types{type}).g_mean(contrast,bin)).^2;
                        end
                        stats.(trial_types{type}).g_discr_variance_rt  (contrast,bin) =      sum(p(g));
                        stats.(trial_types{type}).g_discr_std_rt       (contrast,bin) = sqrt(sum(p(g)));
                        % something like the below is probably right for binomial
                        % confidence...but this isn't very good
                        %stats.(trial_types{type}).exp_val_g         (contrast,bin) = sum(sorted_raw.(trial_types{type}){contrast}.g (stats.(trial_types{type}).bin_index == bin) == 2);
                        %stats.(trial_types{type}).binom_var_g       (contrast,bin) = stats.(trial_types{type}).exp_val_g(contrast,bin) * (1 - stats.(trial_types{type}).exp_val_g(contrast,bin) / length(sorted_raw.(trial_types{type}){contrast}.g(stats.(trial_types{type}).bin_index == bin) ));
                    end
                end
                
                stats.(trial_types{type}).percent_correct_rt   (contrast,bin) =     mean(sorted_raw.(trial_types{type}){contrast}.tf   (stats.(trial_types{type}).bin_index == bin));
                stats.(trial_types{type}).Chat1_prop_rt        (contrast,bin) = .5 - .5 * mean(sorted_raw.(trial_types{type}){contrast}.Chat (stats.(trial_types{type}).bin_index == bin));
                stats.(trial_types{type}).rt_mean_rt           (contrast,bin) =     mean(sorted_raw.(trial_types{type}){contrast}.rt   (stats.(trial_types{type}).bin_index == bin));
            end
        end
        
        
        
        
        
        if g_exists
            [stats.(trial_types{type}).g_mean_sorted(contrast, :), stats.(trial_types{type}).g_mean_sort_index(contrast,:)] = sort(stats.(trial_types{type}).g_mean(contrast,:),2); % sort g_mean to plot against std
            % sort according to sort index of <g>
            stats.(trial_types{type}).g_std_sorted_by_g_mean(contrast, :)           = stats.(trial_types{type}).g_std           (contrast, stats.(trial_types{type}).g_mean_sort_index(contrast,:));
            stats.(trial_types{type}).percent_correct_sorted_by_g_mean(contrast, :) = stats.(trial_types{type}).percent_correct (contrast, stats.(trial_types{type}).g_mean_sort_index(contrast,:));
            stats.(trial_types{type}).Chat1_prop_sorted_by_g_mean(contrast, :)      = stats.(trial_types{type}).Chat1_prop      (contrast, stats.(trial_types{type}).g_mean_sort_index(contrast,:));
        end
        if rt_exists
            stats.(trial_types{type}).rt_mean_sorted_by_g_mean(contrast, :)         = stats.(trial_types{type}).rt_mean         (contrast, stats.(trial_types{type}).g_mean_sort_index(contrast,:));
        end
        
    end
    
    
    
end