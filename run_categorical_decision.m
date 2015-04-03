function run_categorical_decision(initial)
% cd C:\GitHub\Confidence-Theory

% initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
% initial = 'testfast';

if nargin==0
    % initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
    initial = 'shortfast';
end

new_subject = false;
room_letter = 'mbp'; % 'mbp','Carrasco_L1','1139'

category_type = 'same_mean_diff_std'; % 'same_mean_diff_std','sym_uniform'
attention_manipulation = true;
eye_tracking = false;

categorical_decision(category_type, initial, new_subject, ...
    room_letter, attention_manipulation, eye_tracking)