% run_categorical_decision

initial = 'rdshortnotrain'; % 'rdshortnotrain'
user = 'rachel';

new_subject_flag = 'n';
room_letter = 'mbp';

category_type = 'sym_uniform'; % 'same_mean_diff_std'
attention_manipulation = true;
exp_number = 1;
nExperiments = 1;
first_task_letter = 'Attention';



categorical_decision(category_type, initial, new_subject_flag, ...
    room_letter, exp_number, nExperiments, first_task_letter, ...
    attention_manipulation, user)
