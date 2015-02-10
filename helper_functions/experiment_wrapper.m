initial = 'wjshort';
new_subject_flag = 'y';
first_task_letter = 'A';

room_letter = '1139_hires_rig';
category_types = {'diff_mean_same_std','same_mean_diff_std'};

if strcmp(first_task_letter,'B')
    category_types = fliplr(category_types);
end




for i = 1:2
    categorical_decision(category_types{i}, initial, new_subject_flag, room_letter, i, 2, first_task_letter)
end