initial = 'sr';
new_subject_flag = 'n';
first_task_letter = 'B';

room_letter = '1139_hires_rig';
category_types = {'diff_mean_same_std','same_mean_diff_std'};

if strcmp(first_task_letter,'B')
    category_types = fliplr(category_types);
end




for i = 1:2
    categorical_decision(category_types{i}, initial, new_subject_flag, room_letter, i, 2, first_task_letter)
end 


return

%%
figure
for i = 1:3
    mean(mean(Test.responses{i}.tf))
    subplot(1,3,i)
    hist(Test.responses{i}.conf(:))
end

%%
all_resp = [Test.responses{1}.c(:); Test.responses{2}.c(:); Test.responses{3}.c(:)]
all_s = [Test.R.draws{1}(:); Test.R.draws{2}(:); Test.R.draws{3}(:)]


plot(all_s,all_resp+.2*rand(size(all_resp)),'.')