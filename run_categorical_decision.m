function run_categorical_decision(initial)
% cd C:\GitHub\Confidence-Theory

% initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
% initial = 'testfast';

if nargin==0
    % initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
    initial = 'shortfast';
end

exp_type = 'AB'; %'attention' or 'AB'
new_subject = true;

switch exp_type
    case 'attention'
        room_letter = 'Carrasco_L1'; % 'mbp','Carrasco_L1','1139'
        category_type = 'same_mean_diff_std'; % 'same_mean_diff_std','sym_uniform'
        eye_tracking = false;

        attention_manipulation = true;
        
        categorical_decision(category_type, initial, new_subject, ...
            room_letter, attention_manipulation, eye_tracking)
    case 'AB'
        room_letter = '1139';
        attention_manipulation = false;
        eye_tracking = false;
        first_task_letter = 'B';
        category_types = {'diff_mean_same_std', 'same_mean_diff_std'};
        if strcmp(first_task_letter, 'B')
            category_types = fliplr(category_types);
        end
        for i = 1:2
            categorical_decision(category_types{i}, initial, new_subject, ...
                room_letter, attention_manipulation, eye_tracking, i, 2)
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
>>>>>>> 189400d8d26578b0ee3c7027b0d84534da2d262a

