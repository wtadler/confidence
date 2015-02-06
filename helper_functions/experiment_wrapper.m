initial = 'washort';
new_subject_flag = 'y';  % this is weird. doesn't call in experimenter at right times. fix.
first_cat = 2;

room_letter = '1139_hires_rig';
category_types = {'diff_mean_same_std','same_mean_diff_std'};


if first_cat == 2
    
    
    category_types = fliplr
    (category_types);
end




for i = 1:2
    categorical_decision(category_types{i}, initial, new_subject_flag, room_letter, i, 2)
end

%what happens when not new?
% keep demo and conf training?
% lower contrasts
% remind which task, often!



% confidence training text was too low on screen, then this error
%Index exceeds matrix dimensions.

%Error in run_exp (line 26)
%            cval = R.trial_order{blok}(section,
%            trial); %class

%Error in categorical_decision (line 384)
%            [Training.confidence.responses, flag] =
%            run_exp(Training.confidence.n,Training.confidence.R,Test.t,scr,color,P,'Confidence
%            Training',k, new_subject_flag);
 
