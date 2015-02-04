initial = 'wa';
new_subject_flag = 1; 
first_cat = 1;

room_letter = '1139';
category_types = {'diff_mean_same_std','same_mean_diff_std'};
if first_cat == 2
    category_types = fliplr(category_types);
end

for i = 1:2
    categorical_decision(category_types{i}, initial, new_subject_flag, room_letter, i, 2)
end
