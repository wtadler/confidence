function run_categorical_decision(initial)

% initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
user = 'rachel';

new_subject_flag = 'y';
room_letter = 'mbp'; % 'mbp','Carrasco_L1'

category_type = 'same_mean_diff_std'; % 'same_mean_diff_std','sym_uniform'
attention_manipulation = true;
exp_number = 1;
nExperiments = 1;
first_task_letter = 'Attention';



categorical_decision(category_type, initial, new_subject_flag, ...
    room_letter, exp_number, nExperiments, first_task_letter, ...
    attention_manipulation, user)
