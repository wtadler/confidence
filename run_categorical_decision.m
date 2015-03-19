function run_categorical_decision(initial)
cd C:\GitHub\Confidence-Theory

% initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'

new_subject_flag = 'y';
room_letter = '1139'; % 'mbp','Carrasco_L1'

category_type = 'sym_uniform'; % 'same_mean_diff_std','sym_uniform'
attention_manipulation = true;

categorical_decision(category_type, initial, new_subject_flag, ...
    room_letter, attention_manipulation)