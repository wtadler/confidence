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
% stats.correct.index   = raw.tf == 1;
% stats.incorrect.index = raw.tf == 0;
% stats.Chat1.index     = raw.Chat == -1;
% stats.Chat2.index     = raw.Chat == 1;
% stats.C1.index        = raw.C == -1;
% stats.C2.index        = raw.C == 1;


trial_types = setdiff(fieldnames(stats), 'sig_levels'); % the field names defined above.


fields = {'C','s','x','Chat','tf','resp','g','rt'};
output_fields = {'tf','resp','g','rt','Chat'};

for type = 1 : length(trial_types)
    st=stats.(trial_types{type});
    
    
    if g_exists
        for f = 1:length(output_fields)
        % BIN BY CONFIDENCE %%%%%%%%%%%%
        for g = 1:conf_levels
            idx = raw.g == g;
            [Mean, STD] = mean_and_std(raw, output_fields{f}, idx);
            st.g.mean.(output_fields{f})(g) = Mean;
            st.g.std.(output_fields{f})(g) = STD;
            if f == 1
                st.g.bin_counts(g) = sum(idx);
            end
        end
        
        % BIN BY RESPONSE %%%%%%%%%%%%
        for resp = 1:2*conf_levels
            idx = raw.resp == resp;
            [Mean, STD] = mean_and_std(raw, output_fields{f}, idx);
            st.resp.mean.(output_fields{f})(resp) = Mean;
            st.resp.std.(output_fields{f})(resp) = STD;
            if f == 1
                st.resp.bin_counts(resp) = sum(idx);
            end
        end
        end
    end
    
    
    for contrast = 1 : stats.sig_levels;
        % These vectors contain all the trials for a given contrast and
        % trial_type. intersection of two indices.
        
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
        
        % bin trials by s
        [n, sr.bin_index] = histc(sr.s, [-Inf, bins, Inf]);
        %         st.bin_counts (contrast,:) = n(1 : end - 1); % number in each s bin, at this contrast level
        
        
        for f = 1:length(output_fields)
            if isfield(sr,output_fields{f})
                
                % BIN BY CONTRAST %%%%%%%%%%%%
                idx = true(1, length(sr.(output_fields{f})));
                [Mean, STD] = mean_and_std(sr, output_fields{f}, idx);
                st.c.mean.(output_fields{f})(contrast) = Mean;
                st.c.std.(output_fields{f})(contrast) = STD;
                if f == 1 % only need to do this once for each bin, so only do it for the first run field
                    st.c.bin_counts(contrast) = sum(idx);
                end
                
                % BIN BY CONTRAST AND S %%%%%%%%%%%%
                for bin = 1 : length(bins) + 1
                    idx = sr.bin_index == bin;
                    [Mean, STD] = mean_and_std(sr, output_fields{f}, idx);
                    st.c_s.mean.(output_fields{f})(contrast, bin) = Mean;
                    st.c_s.std.(output_fields{f})(contrast, bin) = STD;
                    if f == 1
                        st.c_s.bin_counts(contrast, bin) = sum(idx);
                    end
                end
                
                % BIN BY CONTRAST AND CHOICE %%%%%%%%%%%%
                for Chat = 1:2
                    idx = sr.Chat == 2*Chat-3; % convert [1 2] to [-1 1]
                    [Mean, STD] = mean_and_std(sr, output_fields{f}, idx);
                    st.c_Chat.mean.(output_fields{f})(contrast, Chat) = Mean;
                    st.c_Chat.std.(output_fields{f})(contrast, Chat) = STD;
                    if f == 1
                        st.c_Chat.bin_counts(contrast, Chat) = sum(idx);
                    end
                end
                
                
                if g_exists                    
                    % BIN BY CONTRAST AND CONFIDENCE %%%%%%%%%%%%
                    for g = 1:conf_levels
                        idx = sr.g == g;
                        [Mean, STD] = mean_and_std(sr, output_fields{f}, idx);
                        st.c_g.mean.(output_fields{f})(contrast, g) = Mean;
                        st.c_g.std.(output_fields{f})(contrast, g) = STD;
                        if f == 1
                            st.c_g.bin_counts(contrast, g) = sum(idx);
                        end
                    end
                    
                    % BIN BY CONTRAST AND RESPONSE %%%%%%%%%%%%
                    for resp = 1:2*conf_levels
                        idx = sr.resp == resp;
                        [Mean, STD] = mean_and_std(sr, output_fields{f}, idx);
                        st.c_resp.mean.(output_fields{f})(contrast, resp) = Mean;
                        st.c_resp.std.(output_fields{f})(contrast, resp) = STD;
                        if f == 1
                            st.c_resp.bin_counts(contrast, resp) = sum(idx);
                        end
                    end
                end
            end
        end
        sorted_raw.(trial_types{type}){contrast} = sr;
    end
    stats.(trial_types{type}) = st;
end


    function [Mean, STD] = mean_and_std(sr, field, idx)
        std_beta_dist = @(a,b) sqrt(a*b/((a+b)^2*(a+b+1)));
        
        switch field
            case 'Chat'
                Mean = .5 - .5 * mean(sr.Chat(idx)); % this is actually Chat prop
                %                 struct.std = .5 * std(sr.(field)(idx)); % regular std
                nChatn1 = sum(sr.Chat(idx)==-1);
                nChat1 = sum(sr.Chat(idx)==1);
                STD = std_beta_dist(nChatn1, nChat1);
                
            case 'tf'
                Mean = mean(sr.tf(idx));
                nHits = sum(sr.tf(idx)==1);
                nMisses = sum(sr.tf(idx)==0);
                STD = std_beta_dist(nHits, nMisses);
                
            otherwise
                Mean = mean(sr.(field)(idx));
                STD = std(sr.(field)(idx));
                
        end
    end
end