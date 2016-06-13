function run_categorical_decision(initial)
% cd C:\GitHub\Confidence-Theory
% initial = 'testfast';

if nargin==0
    % initial = 'rd_p1_run02_notrain'; % 'rdshortnotrain'
    initial = 'test';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE SURE THIS IS SET CORRECTLY %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
session = 3;

if session == 1
    new_subject = true;
    staircase = true;
    choice_only = true;
    
    opt_c = [];
    psybayes_struct = [];
elseif session == 2
    new_subject = false;
    staircase = true;
    choice_only = true;

    % SET THIS FOR 2ND SESSION
    old = load('/Users/purplab/Desktop/Rachel/Confidence/confidence/data/mjo_20160608_180004.mat');
    % [extra files here: lma_recovered_20160610_160107.mat]
    
    psybayes_struct = old.psybayes_struct;
    opt_c = [];
     
elseif session > 2
    new_subject = false;
    staircase = false;
    choice_only = false;
    
    % SET THIS FOR 3RD+ SESSION
    opt_c = exp(-2.3167);
    psybayes_struct = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

exp_type = 'attention'; %'attention' or 'AB'

switch exp_type
    case 'attention'
        eye_tracking = true;
        nTestBlocks = 3;
        
        room_letter = 'Carrasco_L1'; % 'mbp','Carrasco_L1','1139'
        cd ~/Desktop/Rachel/Confidence/confidence
        category_type = 'same_mean_diff_std'; % 'same_mean_diff_std','sym_uniform'
        nStimuli = 4;
        stim_type = 'grate';
        
        categorical_decision(category_type, initial, new_subject, ...
            room_letter, nStimuli, eye_tracking, stim_type, [], [], ...
            choice_only, false, false, staircase, psybayes_struct, opt_c, ...
            nTestBlocks)
    case 'AB'
        cd('C:\GitHub\Confidence-Theory')
        test_feedback = false;
        two_response = true;
        
        stim_type = 'ellipse';
        room_letter = '1139';
        nStimuli = 1;
        eye_tracking = true;

        first_task_letter = 'A';
        category_types = {'diff_mean_same_std', 'same_mean_diff_std'};
        if strcmp(first_task_letter, 'B')
            category_types = fliplr(category_types);
        end
        for i = 1:2
            categorical_decision(category_types{i}, initial, new_subject, ...
                room_letter, nStimuli, eye_tracking, stim_type, i, 2, [], ...
                two_response, test_feedback)
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