function [stats, raw_by_type] = indiv_analysis_fcn(raw, bins, varargin)
% model data doesn't work right now with the way theory_plots generates
% fake data.
% make this parfor-ated? the bins loop is really slow

% define defaults
conf_levels = 4; % used for discrete variance
% data_type = 'subject';
flipsig = 1; % implement this. 12/4/15
% discrete = 0;
trial_types = {'all'};
output_fields = {'tf','resp','g','rt','Chat','proportion'};
bin_types = {'c', 's', 'Chat', 'g', 'resp', 'c_s', 'c_C', 'c_Chat', 'c_g', 'c_resp'};
assignopts(who,varargin);

stats = struct;

% Make sure the choice data is in the right format. -1 and 1 rather than 1 and 2
if ismember(2, raw.Chat)
    raw.Chat(raw.Chat==1)=-1;
    raw.Chat(raw.Chat==2)=1;
end

fields = {};
rawfields = fieldnames(raw);
for f = 1:length(rawfields)
    if ~isempty(raw.(rawfields{f})) && ~any(strcmp(rawfields{f}, {'contrast_values', 'cue_validity_values', 'prior_values'}))
        fields = [fields rawfields{f}];
    end
end

if any(strcmp(fields, 'contrast_id'))
    attention = false;
elseif any(strcmp(fields, 'cue_validity_id'))
    attention = true;
end

if isfield(raw, 'cue_validity_id')
    stats.sig_levels = length(unique(raw.cue_validity_id));
else
    stats.sig_levels = length(unique(raw.contrast));
end

for type = 1:length(trial_types)
    switch trial_types{type}
        case 'all'
            idx = true(1, length(raw.C));
        case 'correct'
            idx = raw.tf == 1;
        case 'incorrect'
            idx = raw.tf == 0;
        case 'Chat1'
            idx = raw.Chat == -1;
        case 'Chat2'
            idx = raw.Chat == 1;
        case 'C1'
            idx = raw.C == -1;
        case 'C2'
            idx = raw.C == 1;
        case 'prior1'
            idx = raw.prior_id == 1;
        case 'prior2'
            idx = raw.prior_id == 2;
        case 'prior3'
            idx = raw.prior_id == 3;
    end
    stats.(trial_types{type}).index = idx;
    
    % sort raw data by type
    for f = 1:length(fields)
        raw_by_type.(trial_types{type}).(fields{f}) = raw.(fields{f})(stats.(trial_types{type}).index);
    end
    
    % sort raw data by type and contrast
    for c = 1 : stats.sig_levels;
        if attention
            sig_idx = raw_by_type.(trial_types{type}).cue_validity_id == c;
        else
            sig_idx = raw_by_type.(trial_types{type}).contrast_id     == c;
        end
        
        stats.(trial_types{type}).subindex_by_c{c} = sig_idx;
        
        for f = 1:length(fields)
            raw_by_type.(trial_types{type}).by_contrast(c).(fields{f}) = raw_by_type.(trial_types{type}).(fields{f})(sig_idx);
        end
        
    end
    
    % find s bins for this type
    [~, s_bin_index] = histc(raw_by_type.(trial_types{type}).s, [-Inf, bins, Inf]);
    try
        rt_bins = 15;
        [~, rt_bin_index] = histc(raw_by_type.(trial_types{type}).rt, linspace(0,2,rt_bins+1));
    end
    
    % bin requested output fields
    for b = 1:length(bin_types)
        if ~isempty(bin_types{b})
            stats.(trial_types{type}).(bin_types{b}) = bin_by_field(raw_by_type.(trial_types{type}), bin_types{b});
            
            % mean rate of response and dirichlet std
            bin_counts = stats.(trial_types{type}).(bin_types{b}).bin_counts;
            stats.(trial_types{type}).(bin_types{b}).mean.proportion = bin_counts./sum(bin_counts(:));
            
            a = bin_counts+1;
            a0 = sum(a(:));
            stats.(trial_types{type}).(bin_types{b}).std.proportion = sqrt(a.*(a0-a) ./ (a0^2 * (a0 + 1)));
        end
    end
end


    function st = bin_by_field(raw, bin_type)
        st = struct;
        if isempty(regexp(bin_type, 'c|c_.*'))
            switch bin_type
                case 's'
                    for bin = 1 : length(bins) + 1
                        idx = s_bin_index == bin;
                        st = compute(st, raw, idx, bin);
                    end
                case 'Chat'
                    for Chat = 1:2
                        idx = raw.Chat == 2*Chat-3; % convert [1 2] to [-1 1]
                        st = compute(st, raw, idx, Chat);
                    end
                case 'g'
                    for g = 1:conf_levels
                        idx = raw.g == g;
                        st = compute(st, raw, idx, g);
                    end
                case 'resp'
                    for resp = 1:2*conf_levels
                        idx = raw.resp == resp;
                        st = compute(st, raw, idx, resp);
                    end
                case 'rt'
                    for bin = 1:rt_bins
                        idx = rt_bin_index==bin;
                        st = compute(st, raw, idx, bin);
                    end
                case 'C_s'
                   for C = 1:2
                       idx = raw.C == 2*C-3;
                       [~, bin_index] = histc(raw.s, [-Inf, bins, Inf]);
                       for bin = 1 : length(bins) + 1
                           st = compute(st, raw, idx & (bin_index == bin), C, bin);
                       end
                   end
                    
                    
            end
            %
        else % bin by contrast, or contrast + something else.
            for contrast = 1 : stats.sig_levels
                switch bin_type
                    case 'c'
                        nTrials = length(raw.by_contrast(contrast).C);
                        idx = true(1, nTrials);
                        st = compute(st, raw.by_contrast(contrast), idx, contrast, 1);
                    case 'c_s'
                        [~, bin_index] = histc(raw.by_contrast(contrast).s, [-Inf, bins, Inf]);
                        for bin = 1 : length(bins)+1
                            idx = bin_index == bin;
                            st = compute(st, raw.by_contrast(contrast), idx, contrast, bin);
                        end
                    case 'c_C'
                        for C = 1:2
                            idx = raw.by_contrast(contrast).C == 2*C-3; % convert [1 2] to [-1 1]
                            st = compute(st, raw.by_contrast(contrast), idx, contrast, C);
                        end
                    case 'c_prior'
                        for prior = unique(raw.by_contrast(contrast).prior_id)
                            idx = raw.by_contrast(contrast).prior_id == prior;
                            st = compute(st, raw.by_contrast(contrast), idx, contrast, prior);
                        end
                        
                    case 'c_Chat'
                        for Chat = 1:2
                            idx = raw.by_contrast(contrast).Chat == 2*Chat-3; % convert [1 2] to [-1 1]
                            st = compute(st, raw.by_contrast(contrast), idx, contrast, Chat);
                        end
                    case 'c_g'
                        for g = 1:conf_levels
                            idx = raw.by_contrast(contrast).g == g;
                            st = compute(st, raw.by_contrast(contrast), idx, contrast, g);
                        end
                    case 'c_resp'
                        for resp = 1:2*conf_levels
                            idx = raw.by_contrast(contrast).resp == resp;
                            st = compute(st, raw.by_contrast(contrast), idx, contrast, resp);
                        end
                end
            end
            
            
        end
    end

    function st2 = compute(st2, raw, idx, i, j)
        if ~exist('j','var')
            st2.bin_counts(i) = sum(idx);
        else
            st2.bin_counts(i,j) = sum(idx);
        end
        for field = 1:length(output_fields)
            dep_var = output_fields{field};
            if strcmp(dep_var, 'proportion'); continue; end
            [Mean, STD, SEM] = mean_and_std(raw, dep_var, idx);
            if ~exist('j','var')
                st2.mean.(dep_var)(i) = Mean;
                st2.std.(dep_var)(i) = STD;
                st2.sem.(dep_var)(i) = SEM;
            else
                st2.mean.(dep_var)(i,j) = Mean;
                st2.std.(dep_var)(i,j) = STD;
                st2.sem.(dep_var)(i,j) = SEM;
            end
        end
    end

    function [Mean, STD, SEM] = mean_and_std(sr, field, idx)
        std_beta_dist = @(a,b) sqrt(a*b/((a+b)^2*(a+b+1)));
        
        switch field
            case 'Chat'
                Mean = .5 - .5 * mean(sr.Chat(idx)); % this is actually Chat prop
                %                 struct.std = .5 * std(sr.(field)(idx)); % regular std
                nChatn1 = sum(sr.Chat(idx)==-1);
                nChat1 = sum(sr.Chat(idx)==1);
                STD = std_beta_dist(nChatn1+1, nChat1+1);
                
            case 'tf'
                Mean = mean(sr.tf(idx));
                nHits = sum(sr.tf(idx)==1);
                nMisses = sum(sr.tf(idx)==0);
                STD = std_beta_dist(nHits+1, nMisses+1); % this seems a reasonable way to add a tiny prior and fix the zero problem
                
            otherwise % should RT's have a different kind of errorbar?
                Mean = mean(sr.(field)(idx));
                STD = std(sr.(field)(idx));
        end
        SEM = STD/sqrt(sum(idx));
    end
end