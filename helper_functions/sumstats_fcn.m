function sumstats = sumstats_fcn(data, varargin)

% takes

% define default
decb_analysis = 0;
assignopts(who,varargin);

g_exists  = isfield(data(1).raw, 'g');
rt_exists = isfield(data(1).raw, 'rt');
trial_types = fieldnames(data(1).stats);

for type = 1 : length(trial_types)
    % load data for first subject into multidim array
    sumstats.(trial_types{type}).bin_counts_md      = data(1).stats.(trial_types{type}).bin_counts;
    sumstats.(trial_types{type}).percent_correct_md = data(1).stats.(trial_types{type}).percent_correct;
    sumstats.(trial_types{type}).Chat1_prop_md      = data(1).stats.(trial_types{type}).Chat1_prop;
    if g_exists
        sumstats.(trial_types{type}).g_mean_md          = data(1).stats.(trial_types{type}).g_mean;
    end
    if rt_exists
        sumstats.(trial_types{type}).rt_mean_md         = data(1).stats.(trial_types{type}).rt_mean;
    end
    
    if decb_analysis
        if g_exists
            sumstats.(trial_types{type}).g_decb_mean_md     =(data(1).stats.(trial_types{type}).g_mean(:,2) + data(1).stats.(trial_types{type}).g_mean(:,5)) / 2;
        end
        sumstats.(trial_types{type}).Chat1_decb_prop_mean_md = (data(1).stats.(trial_types{type}).Chat1_prop(:,2) + data(1).stats.(trial_types{type}).Chat1_prop(:,5)) / 2;
    end
    
    if rt_exists
        sumstats.(trial_types{type}).bin_counts_md_rt      = data(1).stats.(trial_types{type}).bin_counts_rt;
        sumstats.(trial_types{type}).percent_correct_md_rt = data(1).stats.(trial_types{type}).percent_correct_rt;
        sumstats.(trial_types{type}).Chat1_prop_md_rt      = data(1).stats.(trial_types{type}).Chat1_prop_rt;
        if g_exists
            sumstats.(trial_types{type}).g_mean_md_rt          = data(1).stats.(trial_types{type}).g_mean_rt;
        end
        sumstats.(trial_types{type}).rt_mean_md_rt         = data(1).stats.(trial_types{type}).rt_mean_rt;
        
        if decb_analysis
            if g_exists
                sumstats.(trial_types{type}).g_decb_mean_md_rt     =(data(1).stats.(trial_types{type}).g_mean_rt(:,2) + data(1).stats.(trial_types{type}).g_mean_rt(:,5)) / 2;
            end
            sumstats.(trial_types{type}).Chat1_decb_prop_mean_md_rt = (data(1).stats.(trial_types{type}).Chat1_prop_rt(:,2) + data(1).stats.(trial_types{type}).Chat1_prop_rt(:,5)) / 2;
        end
    end
    
    
    
    for subject = 2 : length(data); % concatenate along 3rd dim for other subjects
        sumstats.(trial_types{type}).bin_counts_md      = cat(3, sumstats.(trial_types{type}).bin_counts_md,        data(subject).stats.(trial_types{type}).bin_counts);
        sumstats.(trial_types{type}).percent_correct_md = cat(3, sumstats.(trial_types{type}).percent_correct_md,   data(subject).stats.(trial_types{type}).percent_correct);
        sumstats.(trial_types{type}).Chat1_prop_md      = cat(3, sumstats.(trial_types{type}).Chat1_prop_md,        data(subject).stats.(trial_types{type}).Chat1_prop);
        if g_exists
            sumstats.(trial_types{type}).g_mean_md          = cat(3, sumstats.(trial_types{type}).g_mean_md,            data(subject).stats.(trial_types{type}).g_mean);
        end
        if isfield(data(1).stats.(trial_types{type}), 'rt_mean')
            sumstats.(trial_types{type}).rt_mean_md         = cat(3, sumstats.(trial_types{type}).rt_mean_md,           data(subject).stats.(trial_types{type}).rt_mean);
        end
        
        if decb_analysis
            if g_exists
                sumstats.(trial_types{type}).g_decb_mean_md     = cat(3, sumstats.(trial_types{type}).g_decb_mean_md,      (data(subject).stats.(trial_types{type}).g_mean(:,2) + data(subject).stats.(trial_types{type}).g_mean(:,5)) / 2);
            end
            sumstats.(trial_types{type}).Chat1_decb_prop_md = cat(3, sumstats.(trial_types{type}).Chat1_decb_prop_mean_md, (data(subject).stats.(trial_types{type}).Chat1_prop(:,2) + data(subject).stats.(trial_types{type}).Chat1_prop(:,5)) / 2);
        end

        if rt_exists
            sumstats.(trial_types{type}).bin_counts_md_rt      = cat(3, sumstats.(trial_types{type}).bin_counts_md_rt,        data(subject).stats.(trial_types{type}).bin_counts_rt);
            sumstats.(trial_types{type}).percent_correct_md_rt = cat(3, sumstats.(trial_types{type}).percent_correct_md_rt,   data(subject).stats.(trial_types{type}).percent_correct_rt);
            sumstats.(trial_types{type}).Chat1_prop_md_rt      = cat(3, sumstats.(trial_types{type}).Chat1_prop_md_rt,        data(subject).stats.(trial_types{type}).Chat1_prop_rt);
            if g_exists
                sumstats.(trial_types{type}).g_mean_md_rt          = cat(3, sumstats.(trial_types{type}).g_mean_md_rt,            data(subject).stats.(trial_types{type}).g_mean_rt);
            end
            if isfield(data(1).stats.(trial_types{type}), 'rt_mean')
                sumstats.(trial_types{type}).rt_mean_md_rt         = cat(3, sumstats.(trial_types{type}).rt_mean_md_rt,           data(subject).stats.(trial_types{type}).rt_mean_rt);
            end
            
            if decb_analysis
                if g_exists
                    sumstats.(trial_types{type}).g_decb_mean_md_rt     = cat(3, sumstats.(trial_types{type}).g_decb_mean_md_rt,      (data(subject).stats.(trial_types{type}).g_mean_rt(:,2) + data(subject).stats.(trial_types{type}).g_mean_rt(:,5)) / 2);
                end
                sumstats.(trial_types{type}).Chat1_decb_prop_md_rt = cat(3, sumstats.(trial_types{type}).Chat1_decb_prop_mean_md_rt, (data(subject).stats.(trial_types{type}).Chat1_prop_rt(:,2) + data(subject).stats.(trial_types{type}).Chat1_prop_rt(:,5)) / 2);
            end
        end
        
    end
    
    if g_exists
        sumstats.(trial_types{type}).g_mean = ...
            mean(sumstats.(trial_types{type}).g_mean_md, 3);
        sumstats.(trial_types{type}).g_sem = ...
            std(sumstats.(trial_types{type}).g_mean_md, 0, 3) / sqrt(length(data));
    end
    
    sumstats.(trial_types{type}).percent_correct_mean = ...
        mean(sumstats.(trial_types{type}).percent_correct_md, 3);
    sumstats.(trial_types{type}).percent_correct_sem = ...
        std(sumstats.(trial_types{type}).percent_correct_md, 0, 3) / sqrt(length(data));
    
    sumstats.(trial_types{type}).Chat1_prop_mean = ...
        mean(sumstats.(trial_types{type}).Chat1_prop_md, 3);
    sumstats.(trial_types{type}).Chat1_prop_sem = ...
        std(sumstats.(trial_types{type}).Chat1_prop_md, 0, 3) / sqrt(length(data));
    
    if rt_exists
        sumstats.(trial_types{type}).rt_mean = ...
            mean(sumstats.(trial_types{type}).rt_mean_md, 3);
        sumstats.(trial_types{type}).rt_sem = ...
            std(sumstats.(trial_types{type}).rt_mean_md, 0, 3) / sqrt(length(data));
    end
    
    if decb_analysis
        if g_exists
            sumstats.(trial_types{type}).g_decb_mean = ...
                mean(sumstats.(trial_types{type}).g_decb_mean_md, 3);
            sumstats.(trial_types{type}).g_decb_sem = ...
                std(sumstats.(trial_types{type}).g_decb_mean_md, 0, 3) / sqrt(length(data));
        end
        
        sumstats.(trial_types{type}).Chat1_decb_prop = ...
            mean(sumstats.(trial_types{type}).Chat1_decb_prop_md, 3);
        sumstats.(trial_types{type}).Chat1_decb_prop_sem = ...
            std(sumstats.(trial_types{type}).Chat1_decb_prop_md, 0, 3) / sqrt(length(data));
    end
    
    if rt_exists
        if g_exists
            sumstats.(trial_types{type}).g_mean_rt = ...
                mean(sumstats.(trial_types{type}).g_mean_md_rt, 3);
            sumstats.(trial_types{type}).g_sem_rt = ...
                std(sumstats.(trial_types{type}).g_mean_md_rt, 0, 3) / sqrt(length(data));
        end
        
        sumstats.(trial_types{type}).percent_correct_mean_rt = ...
            mean(sumstats.(trial_types{type}).percent_correct_md_rt, 3);
        sumstats.(trial_types{type}).percent_correct_sem_rt = ...
            std(sumstats.(trial_types{type}).percent_correct_md_rt, 0, 3) / sqrt(length(data));
        
        sumstats.(trial_types{type}).Chat1_prop_mean_rt = ...
            mean(sumstats.(trial_types{type}).Chat1_prop_md_rt, 3);
        sumstats.(trial_types{type}).Chat1_prop_sem_rt = ...
            std(sumstats.(trial_types{type}).Chat1_prop_md_rt, 0, 3) / sqrt(length(data));
        
        sumstats.(trial_types{type}).rt_mean_rt = ...
            mean(sumstats.(trial_types{type}).rt_mean_md_rt, 3);
        sumstats.(trial_types{type}).rt_sem_rt = ...
            std(sumstats.(trial_types{type}).rt_mean_md_rt, 0, 3) / sqrt(length(data));
        
        if decb_analysis
            if g_exists
                sumstats.(trial_types{type}).g_decb_mean_rt = ...
                    mean(sumstats.(trial_types{type}).g_decb_mean_md_rt, 3);
                sumstats.(trial_types{type}).g_decb_sem_rt = ...
                    std(sumstats.(trial_types{type}).g_decb_mean_md_rt, 0, 3) / sqrt(length(data));
            end
            
            sumstats.(trial_types{type}).Chat1_decb_prop_rt = ...
                mean(sumstats.(trial_types{type}).Chat1_decb_prop_md_rt, 3);
            sumstats.(trial_types{type}).Chat1_decb_prop_sem_rt = ...
                std(sumstats.(trial_types{type}).Chat1_decb_prop_md_rt, 0, 3) / sqrt(length(data));
        end
    end
    
    
end