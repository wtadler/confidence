function sumstats = sumstats_fcn(data, varargin)

% takes

% define default
% decb_analysis = 0;
% assignopts(who,varargin);

% g_exists  = isfield(data(1).raw, 'g');
% rt_exists = isfield(data(1).raw, 'rt');
trial_types = fieldnames(data(1).stats);
% fields = {'bin_counts','percent_correct','Chat1_prop','g_mean','resp_mean'}
fields = {'tf','resp','g','Chat'};%'rt'
% eventually, move all the if statement checks to the beginning here

for type = 1 : length(trial_types)
    for f = 1:length(fields)
        % initialize
        sst.md.mean.(fields{f}) = [];%data(1).stats.(trial_types{type}).mean.(fields{f});
        sst.md.std.(fields{f}) = [];%data(1).stats.(trial_types{type}).std.(fields{f});
        sst.md.bin_counts = [];%data(1).stats.(trial_types{type}).bin_counts;
        
        % append
        for subject = 1 : length(data); % concatenate along 3rd dim for other subjects
            sst.md.mean.(fields{f}) = cat(3, sst.md.mean.(fields{f}), data(subject).stats.(trial_types{type}).mean.(fields{f}));
            sst.md.std.(fields{f}) = cat(3, sst.md.std.(fields{f}), data(subject).stats.(trial_types{type}).std.(fields{f}));
            sst.md.bin_counts = cat(3, sst.md.bin_counts, data(subject).stats.(trial_types{type}).bin_counts);
        end
        
        % sum, mean, SEM, edgar SEM over subjects
        sst.mean.(fields{f}) = mean(sst.md.mean.(fields{f}),3); % mean of each subject's binned mean
        sst.std.(fields{f}) = std(sst.md.mean.(fields{f}),0,3); % standard deviation of the mean across subjects
        sst.sem.(fields{f}) = sst.std.(fields{f}) / sqrt(length(data)); % standard error of the mean across subjects
        sst.edgar_sem.(fields{f}) = sqrt(sst.std.(fields{f}).^2 ./ length(data) +...
            mean(sst.md.std.(fields{f}).^2./(length(data)*sst.md.bin_counts),3));
        
    end
    sumstats.(trial_types{type}) = sst;
end

end
% for type = 1 : length(trial_types)
    % load data for first subject into multidim array
    
%     for f = 1:length(fields)
% %         if isfield(data(1).stats.(trial_types{type}),fields{f})
% %             sumstats.(trial_types{type}).md.(fields{f}) = data(1).stats.(trial_types{type}).(fields{f});
% %         end
%         sumstats.(trial_types{type}).md.mean.(fields{f}) = data(1).stats.(trial_types{type}).mean.(fields{f});
%         sumstats.(trial_types{type}).md.std.(fields{f}) = data(1).stats.(trial_types{type}).std.(fields{f});
%         sumstats.(trial_types{type}).md.bin_counts = data(1).stats.(trial_types{type}).bin_counts;
%     end
% 
%     
%     
%     
%     sumstats.(trial_types{type}).bin_counts_md      = data(1).stats.(trial_types{type}).bin_counts;
%     sumstats.(trial_types{type}).percent_correct_md = data(1).stats.(trial_types{type}).percent_correct;
%     sumstats.(trial_types{type}).Chat1_prop_md      = data(1).stats.(trial_types{type}).Chat1_prop;
%     if g_exists
%         sumstats.(trial_types{type}).g_mean_md          = data(1).stats.(trial_types{type}).g_mean;
%     end
%     if rt_exists
%         sumstats.(trial_types{type}).rt_mean_md         = data(1).stats.(trial_types{type}).rt_mean;
%     end
    
%     if decb_analysis
%         if g_exists
%             sumstats.(trial_types{type}).g_decb_mean_md     =(data(1).stats.(trial_types{type}).g_mean(:,2) + data(1).stats.(trial_types{type}).g_mean(:,5)) / 2;
%         end
%         sumstats.(trial_types{type}).Chat1_decb_prop_mean_md = (data(1).stats.(trial_types{type}).Chat1_prop(:,2) + data(1).stats.(trial_types{type}).Chat1_prop(:,5)) / 2;
%     end
%     
%     if rt_exists
%         sumstats.(trial_types{type}).bin_counts_md_rt      = data(1).stats.(trial_types{type}).bin_counts_rt;
%         sumstats.(trial_types{type}).percent_correct_md_rt = data(1).stats.(trial_types{type}).percent_correct_rt;
%         sumstats.(trial_types{type}).Chat1_prop_md_rt      = data(1).stats.(trial_types{type}).Chat1_prop_rt;
%         if g_exists
%             sumstats.(trial_types{type}).g_mean_md_rt          = data(1).stats.(trial_types{type}).g_mean_rt;
%         end
%         sumstats.(trial_types{type}).rt_mean_md_rt         = data(1).stats.(trial_types{type}).rt_mean_rt;
%         
%         if decb_analysis
%             if g_exists
%                 sumstats.(trial_types{type}).g_decb_mean_md_rt     =(data(1).stats.(trial_types{type}).g_mean_rt(:,2) + data(1).stats.(trial_types{type}).g_mean_rt(:,5)) / 2;
%             end
%             sumstats.(trial_types{type}).Chat1_decb_prop_mean_md_rt = (data(1).stats.(trial_types{type}).Chat1_prop_rt(:,2) + data(1).stats.(trial_types{type}).Chat1_prop_rt(:,5)) / 2;
%         end
%     end
    
    
% % % %     
% % % % %     for subject = 2 : length(data); % concatenate along 3rd dim for other subjects
% % % % %         for f = 1:length(fields)
% % % % % %             if isfield(data(1).stats.(trial_types{type}), fields{f})
% % % % % %                 sumstats.(trial_types{type}).md.(fields{f}) = cat(3, sumstats.(trial_types{type}).md.(fields{f}),   data(subject).stats.(trial_types{type}).(fields{f}));
% % % % % %             end
% % % % %               sumstats.(trial_types{type}).md.mean.(fields{f}) = cat(3, sumstats.(trial_types{type}).md.mean.(fields{f}), data(subject).stats.(trial_types{type}).mean.(fields{f}));
% % % % %               sumstats.(trial_types{type}).md.std.(fields{f}) = cat(3, sumstats.(trial_types{type}).md.std.(fields{f}), data(subject).stats.(trial_types{type}).std.(fields{f}));
% % % % %               sumstats.(trial_types{type}).md.bin_counts = cat(3, sumstats.(trial_types{type}).md.bin_counts, data(subject).stats.(trial_types{type}).bin_counts);
% % % % %         end
% % % % %         
% % % % %         
%         sumstats.(trial_types{type}).bin_counts_md      = cat(3, sumstats.(trial_types{type}).bin_counts_md,        data(subject).stats.(trial_types{type}).bin_counts);
%         sumstats.(trial_types{type}).percent_correct_md = cat(3, sumstats.(trial_types{type}).percent_correct_md,   data(subject).stats.(trial_types{type}).percent_correct);
%         sumstats.(trial_types{type}).Chat1_prop_md      = cat(3, sumstats.(trial_types{type}).Chat1_prop_md,        data(subject).stats.(trial_types{type}).Chat1_prop);
%         if g_exists
%             sumstats.(trial_types{type}).g_mean_md          = cat(3, sumstats.(trial_types{type}).g_mean_md,            data(subject).stats.(trial_types{type}).g_mean);
%         end
%         if isfield(data(1).stats.(trial_types{type}), 'rt_mean')
%             sumstats.(trial_types{type}).rt_mean_md         = cat(3, sumstats.(trial_types{type}).rt_mean_md,           data(subject).stats.(trial_types{type}).rt_mean);
%         end
        
%         if decb_analysis
%             if g_exists
%                 sumstats.(trial_types{type}).g_decb_mean_md     = cat(3, sumstats.(trial_types{type}).g_decb_mean_md,      (data(subject).stats.(trial_types{type}).g_mean(:,2) + data(subject).stats.(trial_types{type}).g_mean(:,5)) / 2);
%             end
%             sumstats.(trial_types{type}).Chat1_decb_prop_md = cat(3, sumstats.(trial_types{type}).Chat1_decb_prop_mean_md, (data(subject).stats.(trial_types{type}).Chat1_prop(:,2) + data(subject).stats.(trial_types{type}).Chat1_prop(:,5)) / 2);
%         end
% 
%         if rt_exists
%             sumstats.(trial_types{type}).bin_counts_md_rt      = cat(3, sumstats.(trial_types{type}).bin_counts_md_rt,        data(subject).stats.(trial_types{type}).bin_counts_rt);
%             sumstats.(trial_types{type}).percent_correct_md_rt = cat(3, sumstats.(trial_types{type}).percent_correct_md_rt,   data(subject).stats.(trial_types{type}).percent_correct_rt);
%             sumstats.(trial_types{type}).Chat1_prop_md_rt      = cat(3, sumstats.(trial_types{type}).Chat1_prop_md_rt,        data(subject).stats.(trial_types{type}).Chat1_prop_rt);
%             if g_exists
%                 sumstats.(trial_types{type}).g_mean_md_rt          = cat(3, sumstats.(trial_types{type}).g_mean_md_rt,            data(subject).stats.(trial_types{type}).g_mean_rt);
%             end
%             if isfield(data(1).stats.(trial_types{type}), 'rt_mean')
%                 sumstats.(trial_types{type}).rt_mean_md_rt         = cat(3, sumstats.(trial_types{type}).rt_mean_md_rt,           data(subject).stats.(trial_types{type}).rt_mean_rt);
%             end
%             
%             if decb_analysis
%                 if g_exists
%                     sumstats.(trial_types{type}).g_decb_mean_md_rt     = cat(3, sumstats.(trial_types{type}).g_decb_mean_md_rt,      (data(subject).stats.(trial_types{type}).g_mean_rt(:,2) + data(subject).stats.(trial_types{type}).g_mean_rt(:,5)) / 2);
%                 end
%                 sumstats.(trial_types{type}).Chat1_decb_prop_md_rt = cat(3, sumstats.(trial_types{type}).Chat1_decb_prop_mean_md_rt, (data(subject).stats.(trial_types{type}).Chat1_prop_rt(:,2) + data(subject).stats.(trial_types{type}).Chat1_prop_rt(:,5)) / 2);
%             end
%         end
        
%     end
    
%     for f = 1:length(fields)
% %         if isfield(sumstats.(trial_types{type}).md, fields{f})
%             sumstats.(trial_types{type}).mean.(fields{f}) = mean(sumstats.(trial_types{type}).md.mean.(fields{f}),3); % mean of each subject's binned mean
%             sumstats.(trial_types{type}).std.(fields{f}) = std(sumstats.(trial_types{type}).md.mean.(fields{f}),0,3); % standard deviation of the mean across subjects
%             sumstats.(trial_types{type}).sem.(fields{f}) = sumstats.(trial_types{type}).std.(fields{f}) / sqrt(length(data)); % standard error of the mean across subjects
%             sumstats.(trial_types{type}).edgar_sem.(fields{f}) = sumstats.(trial_types{type}).std.(fields{f}).^2 ./length(data)+...
%                 mean(sumstats.(trial_types{type}).md.std.(fields{f}).^2./(length(data)*sumstats.trial_types
% %         end
%     end
% %     
% %     if g_exists
% %         sumstats.(trial_types{type}).g_mean = ...
% %             mean(sumstats.(trial_types{type}).g_mean_md, 3);
% %         sumstats.(trial_types{type}).g_sem = ...
% %             std(sumstats.(trial_types{type}).g_mean_md, 0, 3) / sqrt(length(data));
% %     end
% %     
% %     sumstats.(trial_types{type}).percent_correct_mean = ...
% %         mean(sumstats.(trial_types{type}).percent_correct_md, 3);
% %     sumstats.(trial_types{type}).percent_correct_sem = ...
% %         std(sumstats.(trial_types{type}).percent_correct_md, 0, 3) / sqrt(length(data));
% %     
% %     sumstats.(trial_types{type}).Chat1_prop_mean = ...
% %         mean(sumstats.(trial_types{type}).Chat1_prop_md, 3);
% %     sumstats.(trial_types{type}).Chat1_prop_sem = ...
% %         std(sumstats.(trial_types{type}).Chat1_prop_md, 0, 3) / sqrt(length(data));
% %     
% % %     if rt_exists
% % %         sumstats.(trial_types{type}).rt_mean = ...
% % %             mean(sumstats.(trial_types{type}).rt_mean_md, 3);
% %         sumstats.(trial_types{type}).rt_sem = ...
% %             std(sumstats.(trial_types{type}).rt_mean_md, 0, 3) / sqrt(length(data));
% %     end
% %     
% %     if decb_analysis
% %         if g_exists
% %             sumstats.(trial_types{type}).g_decb_mean = ...
% %                 mean(sumstats.(trial_types{type}).g_decb_mean_md, 3);
% %             sumstats.(trial_types{type}).g_decb_sem = ...
% %                 std(sumstats.(trial_types{type}).g_decb_mean_md, 0, 3) / sqrt(length(data));
% %         end
% %         
% %         sumstats.(trial_types{type}).Chat1_decb_prop = ...
% %             mean(sumstats.(trial_types{type}).Chat1_decb_prop_md, 3);
% %         sumstats.(trial_types{type}).Chat1_decb_prop_sem = ...
% %             std(sumstats.(trial_types{type}).Chat1_decb_prop_md, 0, 3) / sqrt(length(data));
% %     end
% %     
% %     if rt_exists
% %         if g_exists
% %             sumstats.(trial_types{type}).g_mean_rt = ...
% %                 mean(sumstats.(trial_types{type}).g_mean_md_rt, 3);
% %             sumstats.(trial_types{type}).g_sem_rt = ...
% %                 std(sumstats.(trial_types{type}).g_mean_md_rt, 0, 3) / sqrt(length(data));
% %         end
% %         
% %         sumstats.(trial_types{type}).percent_correct_mean_rt = ...
% %             mean(sumstats.(trial_types{type}).percent_correct_md_rt, 3);
% %         sumstats.(trial_types{type}).percent_correct_sem_rt = ...
% %             std(sumstats.(trial_types{type}).percent_correct_md_rt, 0, 3) / sqrt(length(data));
% %         
% %         sumstats.(trial_types{type}).Chat1_prop_mean_rt = ...
% %             mean(sumstats.(trial_types{type}).Chat1_prop_md_rt, 3);
% %         sumstats.(trial_types{type}).Chat1_prop_sem_rt = ...
% %             std(sumstats.(trial_types{type}).Chat1_prop_md_rt, 0, 3) / sqrt(length(data));
% %         
% %         sumstats.(trial_types{type}).rt_mean_rt = ...
% %             mean(sumstats.(trial_types{type}).rt_mean_md_rt, 3);
% %         sumstats.(trial_types{type}).rt_sem_rt = ...
% %             std(sumstats.(trial_types{type}).rt_mean_md_rt, 0, 3) / sqrt(length(data));
% %         
% %         if decb_analysis
% %             if g_exists
% %                 sumstats.(trial_types{type}).g_decb_mean_rt = ...
% %                     mean(sumstats.(trial_types{type}).g_decb_mean_md_rt, 3);
% %                 sumstats.(trial_types{type}).g_decb_sem_rt = ...
% %                     std(sumstats.(trial_types{type}).g_decb_mean_md_rt, 0, 3) / sqrt(length(data));
% %             end
% %             
% %             sumstats.(trial_types{type}).Chat1_decb_prop_rt = ...
% %                 mean(sumstats.(trial_types{type}).Chat1_decb_prop_md_rt, 3);
% %             sumstats.(trial_types{type}).Chat1_decb_prop_sem_rt = ...
% %                 std(sumstats.(trial_types{type}).Chat1_decb_prop_md_rt, 0, 3) / sqrt(length(data));
% %         end
% %     end
%     
%     
% end