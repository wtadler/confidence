initial = 'washort';
new_subject_flag = 'y'; 
first_cat = 2;

room_letter = '1139_hires_rig';
category_types = {'diff_mean_same_std','same_mean_diff_std'};

if first_cat == 2
    
    
    category_types = fliplr(category_types);
end



for i = 1:2
    categorical_decision(category_types{i}, initial, new_subject_flag, room_letter, i, 2)
end

%what happens when not new?
% keep demo and conf training?
% lower contrasts
% remind which task, often!