function [stats, sorted_raw] = indiv_analysis_fcn(raw, bins, varargin)
% model data doesn't work right now with the way theory_plots generates
% fake data.
% make this parfor-ated. the bins loop is really slow

% define defaults
conf_levels = 4; % used for discrete variance
% data_type = 'subject';
flipsig = 1; % implement this 12/4/15
% discrete = 0;
trial_types = {'all'};
output_fields = {'tf','resp','g','rt','Chat'};
assignopts(who,varargin);

if isfield(raw, 'cue_validity_id')
    stats.sig_levels = length(unique(raw.cue_validity_id));
else
    stats.sig_levels = length(unique(raw.contrast));
end

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

% will only use the indices specified in trial_types
stats.all.index = true(1,length(raw.C));
stats.correct.index   = raw.tf == 1;
stats.incorrect.index = raw.tf == 0;
stats.Chat1.index     = raw.Chat == -1;
stats.Chat2.index     = raw.Chat == 1;
stats.C1.index        = raw.C == -1;
stats.C2.index        = raw.C == 1;

fields = {'C','s','x','Chat','tf','resp','g','rt','contrast_id'};

for type = 1:length(trial_types)
    % sort raw data
    for f = 1:length(fields)
        if isfield(raw, fields{f})
            raw_by_type.(trial_types{type}).(fields{f}) = raw.(fields{f})(stats.(trial_types{type}).index);
        end
    end
%     
%     % bin requested output fields
%     for f = 1:length(output_fields)
%         % bin, marginalizing over contrast
%         stats = bin_by_field(stats, raw_by_type.(trial_types{type}), output_fields{f});
%     end

end
% 
% % not sure if this is the right approach, 12/4/15
%     function stats = bin_by_field(stats, raw, output_field)
%         switch output_field
%             case 'tf'
%             case 'Chat'
%             case 'g'
%             case 'resp'
%             case 'rt'
%         end
%     end
%         
        

% refactor this!!
for type = 1 : length(trial_types)
    st=stats.(trial_types{type});
    
    % sort trials by type
    for f = 1:length(fields)
        if isfield(raw, fields{f})
            raw_type.(fields{f}) = raw.(fields{f})(st.index);
        end
    end
    
    [~, bin_index] = histc(raw_type.s, [-Inf, bins, Inf]);
        
    for f = 1:length(output_fields)
        if isfield(raw_type, output_fields{f})
            if g_exists
                % BIN BY CONFIDENCE %%%%%%%%%%%%
                for g = 1:conf_levels
                    idx = raw_type.g == g;
                    [Mean, STD] = mean_and_std(raw_type, output_fields{f}, idx);
                    st.g.mean.(output_fields{f})(g) = Mean;
                    st.g.std.(output_fields{f})(g) = STD;
                    if f == 1
                        st.g.bin_counts(g) = sum(idx);
                    end
                end
                
                % BIN BY RESPONSE %%%%%%%%%%%%
                for resp = 1:2*conf_levels
                    idx = raw_type.resp == resp;
                    [Mean, STD] = mean_and_std(raw_type, output_fields{f}, idx);
                    st.resp.mean.(output_fields{f})(resp) = Mean;
                    st.resp.std.(output_fields{f})(resp) = STD;
                    if f == 1
                        st.resp.bin_counts(resp) = sum(idx);
                    end
                end
            end
            
            % BIN BY S %%%%%%%%%%%%
            for bin = 1 : length(bins) + 1
                idx = bin_index == bin;
                [Mean, STD] = mean_and_std(raw_type, output_fields{f}, idx);
                st.s.mean.(output_fields{f})(bin) = Mean;
                st.s.std.(output_fields{f})(bin) = STD;
                if f == 1
                    st.s.bin_counts(bin) = sum(idx);
                end
            end
            
            % BIN BY CHAT %%%%%%%%
            for Chat = 1:2
                idx = raw_type.Chat == 2*Chat-3; % convert [1 2] to [-1 1]
                [Mean, STD] = mean_and_std(raw_type, output_fields{f}, idx);
                st.Chat.mean.(output_fields{f})(Chat) = Mean;
                st.Chat.std.(output_fields{f})(Chat) = STD;
                if f == 1
                    st.Chat.bin_counts(Chat) = sum(idx);
                end
            end
        end
    end
    
    
    
    
    for contrast = 1 : stats.sig_levels;
        % These vectors contain all the trials for a given contrast and
        % trial_type. intersection of two indices.
        
        % trials are already sorted by type. sort them by reliability too
        if isfield(raw_type, 'cue_validity_id')
            sig_idx = raw_type.cue_validity_id == contrast;
        else
            sig_idx = raw_type.contrast_id == contrast;
        end
        
        for f = 1:length(fields)
            if isfield(raw,fields{f})
                sr.(fields{f}) = ...
                    raw_type.(fields{f})(sig_idx);
            end
        end
        
        % bin trials by s
        [~, sr.bin_index] = histc(sr.s, [-Inf, bins, Inf]);        
        
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
                STD = std_beta_dist(nChatn1+1, nChat1+1);
                
            case 'tf'
                Mean = mean(sr.tf(idx));
                nHits = sum(sr.tf(idx)==1);
                nMisses = sum(sr.tf(idx)==0);
                STD = std_beta_dist(nHits+1, nMisses+1); % this seems a reasonable way to add a tiny prior and fix the zero problem
                
            otherwise
                Mean = mean(sr.(field)(idx));
                STD = std(sr.(field)(idx));
                
        end
    end
end