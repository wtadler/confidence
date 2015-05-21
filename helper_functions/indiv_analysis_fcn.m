function [stats, sorted_raw] = indiv_analysis_fcn(raw, bins, varargin)
% model data doesn't work right now with the way theory_plots generates
% fake data.
% make this parfor-ated. the bins loop is really slow

% define defaults
conf_levels = 4; % used for discrete variance
if isfield(raw, 'cue_validity_id')
    stats.sig_levels = length(unique(raw.cue_validity_id));
else
    stats.sig_levels = length(unique(raw.contrast));
end

data_type = 'subject';
flipsig = 1;
discrete = 0;
assignopts(who,varargin);

g_exists  = isfield(raw, 'g');
rt_exists = isfield(raw, 'rt');

% if strcmp(data_type, 'model')  % this is only for the weird theory_MCS_and_plots way of looping over sigma. deprecated
%     sig_levels = 1;
%     raw.contrast_id = ones(1,length(raw.C));
%     raw.contrast_values = 1;
%     raw.rt = -ones(1,length(raw.C)); % fake rt data. better to have this then to ugly up the code below with if statements.
% end


% Make sure the choice data is in the right format. -1 and 1 rather than 1 and 2
if ismember(2, raw.Chat)
    raw.Chat(raw.Chat==1)=-1;
    raw.Chat(raw.Chat==2)=1;
end

stats.all.index = true(1,length(raw.C));
% stats.all.index       = 1 : length(raw.C); % these indices just go into histc
% stats.correct.index   = find(raw.tf == 1);
% stats.incorrect.index = find(raw.tf == 0);
% stats.Chat1.index     = find(raw.Chat == -1); % this should be logical indexing
% stats.Chat2.index     = find(raw.Chat == 1);
% stats.C1.index        = find(raw.C == -1);
% stats.C2.index        = find(raw.C == 1);
% idx = stats.Chat1.index + 1;
% stats.after_Chat1.index= idx(idx<length(raw.C));
% idx = stats.Chat2.index + 1;
% stats.after_Chat2.index= idx(idx<length(raw.C));


trial_types = setdiff(fieldnames(stats), 'sig_levels'); % the field names defined above.

% cut out sig sorting thing here, moved to compile_and_analyze

for type = 1 : length(trial_types)    
    st=stats.(trial_types{type});
    for contrast = 1 : stats.sig_levels;
        % These vectors contain all the trials for a given contrast and
        % trial_type. intersection of two indices.
        fields = {'C','s','x','Chat','tf','resp','g','rt'};
        
        % sort trials by reliability (sig) and trial type
        if isfield(raw, 'cue_validity_id')
            sig_idx = raw.cue_validity_id == contrast;
        else
            sig_idx = raw.contrast_id == contrast;
        end
        
        for f = 1:length(fields)
            if isfield(raw,fields{f})
                sr.(fields{f}) = ...
                    raw.(fields{f})(sig_idx & st.index);
            end
        end
        
        
%         st.Chat1_prop_over_c(contrast) = .5 - .5 * mean(sr.Chat);
%         st.percent_correct_over_c(contrast) = mean(sr.tf);
%         st.resp_over_c(contrast) = mean(sr.resp);
%         
%         if g_exists
%             for conf_level = 1 : conf_levels
%                 st.Chat1_prop_over_c_and_g     (sig_levels + 1 - contrast, conf_level) = .5 - .5 * mean(sr.Chat(sr.g == conf_level));
%                 st.percent_correct_over_c_and_g(sig_levels + 1 - contrast, conf_level) = mean(sr.tf(sr.g == conf_level));
%             end
%             for response = 1:conf_levels*2
%                 st.percent_correct_over_c_and_resp(sig_levels + 1 - contrast, response) = mean(sr.tf(sr.resp == response));
%             end
%             st.resp_mean_over_c(contrast) = mean(sr.resp);
%             st.g_mean_over_c(contrast) = mean(sr.g);
%             for i = [1 2]
%                 choice = 2*i-3;
%                 st.g_mean_over_c_and_Chat(sig_levels + 1 - contrast, i) = mean(sr.g(sr.Chat == choice));
%                 st.resp_mean_over_c_and_Chat(sig_levels + 1 - contrast, i) = mean(sr.resp(sr.Chat == choice));
%             end
%         end

        
        % bin trials by s
        [n, st.bin_index] = histc(sr.s, [-Inf, bins, Inf]);
        st.bin_counts (contrast,:) = n(1 : end - 1); % number in each s bin, at this contrast level
        output_fields = {'tf','resp','g','rt'};
        for bin = 1 : length(bins) + 1; % calc stats for each bin over s
            for f = 1:length(output_fields)
                if isfield(sr,output_fields{f})
                    st.mean.(output_fields{f})(contrast,bin) = mean(sr.(output_fields{f})(st.bin_index == bin));
                    st.std.(output_fields{f})(contrast,bin) = std(  sr.(output_fields{f})(st.bin_index == bin));
                    st.sem.(output_fields{f})(contrast,bin) = st.std.(output_fields{f})(contrast,bin)/sqrt(st.bin_counts(contrast,bin));
                end
            end
            st.mean.Chat(contrast,bin) = .5 - .5 * mean(sr.Chat(st.bin_index == bin)); % this is actually chat prop
            st.std.Chat(contrast,bin) = .5 * std(sr.Chat(st.bin_index == bin));
            st.sem.Chat(contrast,bin) = st.std.Chat(contrast,bin) / sqrt(st.bin_counts(contrast,bin)); % this is sem chat prop
            % might want to do sem of beta dist for binary vars like choice or tf??
            %                             if strcmp(dep_vars{dep_var}, 'tf') % standard deviation of the beta distribution instead of SEM
%                 nHits = sum(raw.(dep_vars{dep_var})(idx));
%                 nMisses = nTrials - nHits;
%                 sems (bin, i) = sqrt(nHits*nMisses/((nHits+nMisses)^2*(nHits+nMisses+1)));

        end
        
        % bin trials by reliability
        for f = 1:length(output_fields)
            if isfield(sr,output_fields{f})
                st.mean_marg_over_s.(output_fields{f})(contrast) = mean(sr.(output_fields{f}));
                st.std_marg_over_s.(output_fields{f})(contrast) = std(  sr.(output_fields{f}));
                st.sem_marg_over_s.(output_fields{f})(contrast) = st.std_marg_over_s.(output_fields{f})(contrast)/sqrt(sum(st.bin_counts(contrast,:)));
            end
            st.mean_marg_over_s.Chat(contrast) = .5 - .5 * mean(sr.Chat);
            st.std_marg_over_s.Chat(contrast) = .5 * std(sr.Chat);
            st.sem_marg_over_s.Chat(contrast) = st.std_marg_over_s.Chat(contrast)/sqrt(sum(st.bin_counts(contrast,:)));

        end

        
%         
%         if rt_exists
%             % bin trials by rt. MERGE THIS WITH TOP. also, add resp to this if you want
%             bins_rt = bin_generator(length(bins)+1,'binstyle','rt'); % need new kinds of bins
%             [n, st.bin_index] = histc(sr.rt, [0, bins_rt, Inf]);
%             st.bin_counts_rt (contrast,:) = n(1 : end - 1);
%             
%             for bin = 1 : length(bins_rt) + 1; % calc stats for each bin over s
%                 if g_exists
%                     st.g_mean_rt            (contrast,bin) =     mean(sr.g    (st.bin_index == bin));
%                     st.g_std_rt             (contrast,bin) =     std (sr.g    (st.bin_index == bin));
%                 end
%                 
%                 st.percent_correct_rt   (contrast,bin) =     mean(sr.tf   (st.bin_index == bin));
%                 st.Chat1_prop_rt        (contrast,bin) = .5 - .5 * mean(sr.Chat (st.bin_index == bin));
%                 st.rt_mean_rt           (contrast,bin) =     mean(sr.rt   (st.bin_index == bin));
%             end
%         end
%         
        
%         if g_exists
%             [st.g_mean_sorted(contrast, :), st.g_mean_sort_index(contrast,:)] = sort(st.g_mean(contrast,:),2); % sort g_mean to plot against std
%             % sort according to sort index of <g>
%             st.g_std_sorted_by_g_mean(contrast, :)           = st.g_std           (contrast, st.g_mean_sort_index(contrast,:));
%             st.percent_correct_sorted_by_g_mean(contrast, :) = st.percent_correct (contrast, st.g_mean_sort_index(contrast,:));
%             st.Chat1_prop_sorted_by_g_mean(contrast, :)      = st.Chat1_prop      (contrast, st.g_mean_sort_index(contrast,:));
%         end
%         if rt_exists
%             st.rt_mean_sorted_by_g_mean(contrast, :)         = st.rt_mean         (contrast, st.g_mean_sort_index(contrast,:));
%         end
        
        sorted_raw.(trial_types{type}){contrast} = sr;

    end
    
    stats.(trial_types{type}) = st;

end