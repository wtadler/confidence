function run_categorical_decision(initial)
% cd C:\GitHub\Confidence-Theory
% initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
% initial = 'testfast';

if nargin==0
    % initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
    initial = 'sj';
end

exp_type = 'attention'; %'attention' or 'AB'
new_subject = false;

switch exp_type
    case 'attention'
        room_letter = '1139'; % 'mbp','Carrasco_L1','1139'
        category_type = 'same_mean_diff_std'; % 'same_mean_diff_std','sym_uniform'
        eye_tracking = false;
        nStimuli = 4;

        category_type = 'same_mean_diff_std'; % 'same_mean_diff_std','sym_uniform'
        stim_type = 'grate';
        
        categorical_decision(category_type, initial, new_subject, ...
            room_letter, nStimuli, eye_tracking, stim_type)
    case 'AB'
        cd('C:\GitHub\Confidence-Theory')
        stim_type = 'ellipse';
        room_letter = '1139';
        nStimuli = 1;
        eye_tracking = false;
        first_task_letter = 'A';
        category_types = {'diff_mean_same_std', 'same_mean_diff_std'};
        if strcmp(first_task_letter, 'B')
            category_types = fliplr(category_types);
        end
        for i = 1:2
            categorical_decision(category_types{i}, initial, new_subject, ...
                room_letter, nStimuli, eye_tracking, stim_type, i, 2)
        end

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