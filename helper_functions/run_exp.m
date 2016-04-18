function [responses, flag, psybayes_struct] = run_exp(n, R, t, scr, color, P, type, blok, new_subject, task_str, final_task, subject_name, choice_only, two_response, test_feedback, psybayes_struct)

if ~exist('test_feedback','var')
    test_feedback = false;
end

if ~exist('choice_only', 'var') || isempty(choice_only)
    if (strcmp(type, 'Category Training') && ~test_feedback) || strcmp(type, 'Attention Training')
        choice_only = true;
    elseif test_feedback
        choice_only = false;
    end
end

if ~exist('two_response','var')
    two_response = false;
end

if ~exist('psybayes_struct','var') || isempty(psybayes_struct)
    staircase = false;
else
    staircase = true;
end

nStimuli = size(R.draws{blok}, 3);

if nStimuli == 4
    cross_rot = 45;
else
    cross_rot = 0;
end

flag = 0;
responses.c = zeros(n.sections, n.trials); % cat response
responses.conf = zeros(n.sections, n.trials); % conf response
responses.rt = zeros(n.sections, n.trials); % rt
if two_response
    responses.rtConf = zeros(n.sections, n.trials);
end
%bool matrix of correct/incorrect responses
responses.tf = zeros(n.sections, n.trials);

%%% Text before trials %%%
switch type
    case 'Category Training'
        str=['Coming up: ' task_str type];
    case 'Confidence Training'
        str=['Let''s get some quick practice with confidence ratings.\n\n'...
            'Coming up: ' task_str type];
    case 'Confidence and Atention Training'
        str=['Let''s get some practice with confidence ratings and attentional cues.\n\n'...
            'Coming up: ' task_str type];
    case 'Attention Training'
        str=['Let''s get some practice with attentional cues.\n\n'...
            'Coming up: ' task_str type];
    case 'Testing'
        str=['Coming up: ' task_str type ' Block ' num2str(blok) ' of ' num2str(n.blocks)];
end

if P.eye_tracking
    Eyelink('message', 'Subject code: %s', subject_name);
end
[~,ny]=center_print(str,'center');
Screen('TextSize', scr.win, scr.fontsize); % reset fontsize if made small for attention training.
flip_key_flip(scr,'begin',ny,color, false);

priors = unique(R.prior{blok});
multi_prior = length(priors) ~= 1; % true if there are multiple priors

%%%Run trials %%%
try
    for section = 1:n.sections
        if P.eye_tracking
            block = [type ' Block ' num2str(blok) ', Section ' num2str(section)];
            Eyelink('message', str);
        end
        
        n_trials = n.trials;
        trial_order = 1:n.trials;
        trial_counter = 1;
        
        % for trial = 1:n.trials;
        while trial_counter <= n_trials
            %             responses.c %%% print for debugging
            %             trial_order
            
            % Current trial number
            i_trial = trial_counter; % the trial we're on now
            trial = trial_order(i_trial); % the current index into trials
            trial_counter = trial_counter+1; % update the trial counter so that we will move onto the next trial, even if there is a fixation break
            
            if multi_prior
                prior_id = find(R.prior{blok}(section, trial) == priors);
                
                switch prior_id
                    case 1
                        fixation_cross = scr.crossR;
                    case 2
                        fixation_cross = scr.cross;
                    case 3
                        fixation_cross = scr.crossL;
                end
            else
                fixation_cross = scr.cross;
            end
            
            Screen('DrawTexture', scr.win, fixation_cross, [], [], cross_rot);
            t0 = Screen('Flip', scr.win);
            
            % Check fixation hold
            if P.eye_tracking
                % prints EyeLink message TRIAL_START, SYNC_TIME
                drift_corrected = rd_eyeLink('trialstart', scr.win, {scr.el, i_trial, scr.cx, scr.cy, scr.rad});
                
                if drift_corrected
                    % restart trial
                    Screen('DrawTexture', scr.win, fixation_cross, [], [], cross_rot);
                    t0 = Screen('Flip', scr.win);
                end
            end
            
            stim = struct;
            if staircase
                switch R.cue_validity{blok}(section, trial)
                    case 1
                        condition = 'valid';
                    case 0
                        condition = 'neutral';
                    case -1
                        condition = 'invalid';
                end
                [next_contrast, psybayes_struct.(condition)] = psybayes(psybayes_struct.(condition), psybayes_struct.(condition).method, psybayes_struct.(condition).vars, psybayes_struct.(condition).trial_contrast, psybayes_struct.(condition).trial_correct);
                psybayes_struct.(condition).trial_contrast = next_contrast;
                R.sigma{blok}(section, trial, :) = exp(next_contrast);
            end
            
            for i = 1:nStimuli
               stim(i).ort =       R.draws{blok}(section, trial, i);  %orientation
               stim(i).phase =     R.phase{blok}(section, trial, i);  %phase (not needed by ellipse)
               stim(i).cur_sigma = R.sigma{blok}(section, trial, i);  %contrast (pregenerated when not using staircase)
            end
            
            if nStimuli == 1
                WaitSecs(t.betwtrials/1000);
            elseif nStimuli >= 2
                %                 stim(2).ort = R2.draws{blok}(section, trial);
                %                 stim(2).cur_sigma = R2.sigma{blok}(section, trial);
                %                 stim(2).phase = R2.phase{blok}(section, trial);
                
                % DISPLAY SPATIAL ATTENTION CUE
                
                if nStimuli == 2
                    if R.cue{blok}(section, trial) == 0
                        cue = scr.cueLR;
                        rot = 0;
                    elseif R.cue{blok}(section, trial) == 1
                        cue = scr.cueL;
                        rot = 0;
                    elseif R.cue{blok}(section, trial) == 2
                        cue = scr.cueL;
                        rot = 180;
                    end
                elseif nStimuli == 4
                    if R.cue{blok}(section, trial) == 0
                        cue = scr.cueLRUD;
                        rot = 45;
                    else
                        cue = scr.cueL;
                        rot = 225 - 90 * R.cue{blok}(section, trial);
                    end
                else
                    warning('No support for nStimuli other than 2 or 4')
                end
                
                Screen('DrawTexture', scr.win, cue, [], [], rot);
                
                %                 if R.cue{blok}(section, trial) == -1
                %                     Screen('DrawTexture', scr.win, scr.cueL);
                %                 elseif R.cue{blok}(section, trial) == 0
                %                     Screen('DrawTexture', scr.win, scr.cueLR);
                %                 elseif R.cue{blok}(section, trial) == 1
                %                     Screen('DrawTexture', scr.win, scr.cueR);
                %                 end
                
                t_cue = Screen('Flip', scr.win, t0 + t.betwtrials/1000);
                if P.eye_tracking
                    Eyelink('Message', 'EVENT_CUE');
                end
                Screen('DrawTexture', scr.win, fixation_cross, [], [], cross_rot);
                t_cue_off = Screen('Flip', scr.win, t_cue + t.cue_dur/1000);
                
                %%% should make this timing exact by interfacing with grate
                if P.eye_tracking
                    fixation = 1;
                    while GetSecs - t_cue_off < t.cue_target_isi/1000 - P.eye_slack && fixation
                        WaitSecs(.01);
                        % prints EyeLink message FIX_CHECK, and potentially BROKE_FIXATION
                        fixation = rd_eyeLink('fixcheck', scr.win, {scr.cx, scr.cy, scr.rad});
                    end
                    if ~fixation
                        [trial_order, n_trials] = fixationBreakTasks(scr.win, color.bg, trial_order, i_trial, n_trials);
                        continue
                    end
                else
                    WaitSecs(t.cue_target_isi/1000);
                end
            end
            
            if P.eye_tracking
                Eyelink('Message', 'EVENT_TARGET');
            end
            if strcmp(P.stim_type, 'gabor')
                r_gabor(P, scr, t, stim); % haven't yet added phase info to this function
            elseif strcmp(P.stim_type, 'grate')
                grate(P, scr, t, stim);
            elseif strcmp(P.stim_type, 'ellipse')
                ellipse(P, scr, t, stim); % ellipse doesn't need phase info
            end
            
            
            if nStimuli >= 2
                % DISPLAY RESPONSE CUE (i.e. probe)
                %%% should make this timing exact by interfacing with grate
                Screen('DrawTexture', scr.win, fixation_cross, [], [], cross_rot);
                t_target_off = Screen('Flip', scr.win);
                
                if nStimuli == 2
                    if R.probe{blok}(section, trial)     == 1
                        rot = 0;
                    elseif R.probe{blok}(section, trial) == 2
                        rot = 180;
                    end
                elseif nStimuli == 4
                    rot = 225 - 90 * R.probe{blok}(section, trial);
                end
                
                Screen('DrawTexture', scr.win, scr.resp_cueL, [], [], rot);
                
                %                 if R.probe{blok}(section, trial) == -1
                %                     Screen('DrawTexture', scr.win, scr.resp_cueL);
                %                 elseif R.probe{blok}(section, trial) == 1
                %                     Screen('DrawTexture', scr.win, scr.resp_cueR);
                %                 end
                
                if P.eye_tracking
                    fixation = 1;
                    while GetSecs - t_target_off < t.cue_target_isi/1000 - P.eye_slack && fixation
                        WaitSecs(.01);
                        % prints EyeLink message FIX_CHECK, and potentially BROKE_FIXATION
                        fixation = rd_eyeLink('fixcheck', scr.win, {scr.cx, scr.cy, scr.rad});
                    end
                    if ~fixation
                        [trial_order, n_trials] = fixationBreakTasks(...
                            scr.win, color.bg, trial_order, i_trial, n_trials);
                        continue
                    end
                end
                
                t_resp_cue = Screen('Flip', scr.win, t_target_off + t.cue_target_isi/1000);
                if P.eye_tracking
                    Eyelink('Message', 'EVENT_RESPCUE');
                end
            end
            
            %clc;
            fprintf('blok %g, section %g, trial %g\n\n',blok,section,trial)
            
            %subject input
            t0 = GetSecs;
            Chat = 0;
            
            C = R.trial_order{blok}(section, trial); % true category
            
            while Chat == 0
                [tCatResp, keyCode] = KbWait(-1, 2); % 2 waits for a release then press
                
                if keyCode(scr.keyinsert)%% && keyCode(scr.keyenter) && sum(keyCode)==2
                    error('You cancelled the script by pressing insert.')%% the insert and enter keys simultaneously.')
                end
                
                if choice_only
                    if keyCode(scr.key5) % cat 1
                        Chat = 1;
                    elseif keyCode(scr.key6) % cat 2
                        Chat = 2;
                    else
                        continue
                    end
                    
                else % if collecting confidence responses
                    if two_response
                        if keyCode(scr.keyC1)
                            Chat = 1;
                        elseif keyCode(scr.keyC2)
                            Chat = 2;
                        else
                            continue
                        end
                        
                        [~,ny] = center_print('Confidence?', scr.cy-30);
                        t1=Screen('Flip', scr.win);
                        WaitSecs(0.1);
                        
                        conf = 0;
                        while conf == 0
                            [tConfResp, keyCode] = KbWait(-1,2); % 2 waits for a release then press
                            if keyCode(scr.key1)
                                conf = 1;
                            elseif keyCode(scr.key2)
                                conf = 2;
                            elseif keyCode(scr.key3)
                                conf = 3;
                            elseif keyCode(scr.key4)
                                conf = 4;
                            else
                                continue
                            end
                        end
                        
                    elseif ~two_response
                        if keyCode(scr.key1)
                            conf = 4;
                            Chat = 1;
                        elseif keyCode(scr.key2)
                            conf = 3;
                            Chat = 1;
                        elseif keyCode(scr.key3)
                            conf = 2;
                            Chat = 1;
                        elseif keyCode(scr.key4)
                            conf = 1;
                            Chat = 1;
                        elseif keyCode(scr.key7)
                            conf = 1;
                            Chat = 2;
                        elseif keyCode(scr.key8)
                            conf = 2;
                            Chat = 2;
                        elseif keyCode(scr.key9)
                            conf = 3;
                            Chat = 2;
                        elseif keyCode(scr.key10)
                            conf = 4;
                            Chat = 2;
                        else
                            continue
                        end
                    end
                    
                    confstrings = {'VERY LOW', 'SOMEWHAT LOW', 'SOMEWHAT HIGH', 'VERY HIGH'};
                    confstr = confstrings{conf};
                end
            end
            
            %record 1 if correct, 0 if incorrect
            %             fprintf('cat %d - ACC %d\n', resp, resp==cval) % for debugging
            responses.tf(section, trial) = (Chat == C);
            if staircase
                psybayes_struct.(condition).trial_correct = responses.tf(section, trial);
            end
            responses.c (section, trial) = Chat;
            if ~choice_only
                responses.conf(section, trial) = conf;
            end

            responses.rt(section,trial) = tCatResp - t0;
            if two_response
                responses.rtConf(section,trial) = tConfResp - t1;
            end
            
            if (strcmp(type,'Testing') && test_feedback) || (~strcmp(type, 'Testing') && ~strcmp(type, 'Attention Training')) % give trial by trial feedback unless testing.
                if Chat == C
                    status = 'Correct!';
                    stat_col = color.grn;
                else
                    status = 'Incorrect!';
                    stat_col = color.red;
                end
                
                if choice_only
                    [~,ny]=center_print(sprintf('You said: Category %i',Chat),scr.cy-60); % scr.cy-50
                    [~,ny]=center_print(sprintf('\n%s', status),ny+10,stat_col);
                    
                    Screen('Flip',scr.win);
                    WaitSecs(t.feedback/1000);
                elseif ~choice_only
                    [~,ny]=center_print(sprintf('You said: Category %i with %s confidence.',Chat,confstr),scr.cy - 120); % -50
                    if test_feedback && strcmp(type, 'Category Training')
                        [~,ny]=center_print(sprintf('\nYour category choice is %s', lower(status)),ny+40,stat_col);
                        flip_key_flip(scr,'continue',ny,color, false, 'initial_wait', 0);
                        
                    elseif test_feedback && ~strcmp(type, 'Category Training')
                        [~,ny]=center_print(sprintf('\n%s', status),ny+40,stat_col);
                        Screen('Flip',scr.win); %, tCatResp+t.pause/1000);
                        
                        WaitSecs(t.feedback/1000);
                    end
                end
                
                
            end
            if P.eye_tracking
                Eyelink('Message', 'TRIAL_END %d', i_trial);
            end
        end
        
        responses.trial_order{section} = trial_order;
        
        %if another section in the same block immediately follows
        if section ~= n.sections
            [~,scorereport]=calcscore(responses,n.trials);
            if strcmp('Category Training', type) && blok == 1 % partway through training block 1. when experimenter should leave room
                midtxt = sprintf('You got %s\n\nYou have completed\n\n%s of %s%s.', scorereport, fractionizer(section, n.sections), task_str, type);
                str = 'continue';
            elseif strcmp('Testing', type)
                midtxt = sprintf('You have completed\n\n%s of %s%s Block %i of %i.', fractionizer(section, n.sections), task_str, type, blok, n.blocks);
                str = 'continue';
            else
                midtxt = sprintf('You have completed\n\n%s of %s%s.', fractionizer(section, n.sections), task_str, type);
                str = 'continue';
            end
            
            [~,ny]=center_print(midtxt,'center');
            flip_key_flip(scr,str,ny,color,false);
        end
    end
    
    
    % FEEDBACK/INSTRUCTIONS AFTER END OF BLOCK
    [blockscore,scorereport]= calcscore(responses,n.sections*n.trials);
    if blockscore >= 66
        motivational_str = 'Great job!';
    elseif blockscore < 66 && blockscore >=54
        motivational_str = 'Good.';
    elseif blockscore < 54
        motivational_str = 'Please try a bit harder!\n\nIf you are confused about the task,\n\nplease talk to the experimenter.';
    end
    
    experimenter_needed = false;
    switch type
        case 'Category Training'
            % strcmp(task_str,'') is a standin for nExperiments == 1.
            if blok == 1 && new_subject && ~test_feedback &&  (strcmp(task_str,'') | (~strcmp(task_str,'') && ~final_task))
                hitxt = sprintf('You just got %s\n\n\nPlease go get the experimenter from the other room!',scorereport);
                experimenter_needed = true;
            else
                hitxt = sprintf('%s\n\nYou just got %s\n',motivational_str,scorereport);
            end
        case 'Attention Training'
            hitxt = sprintf('%s\n\nYou just got %s\n', motivational_str, scorereport);
        case 'Testing'
            hitxt = sprintf('%s\n\nYou''ve just finished %sTesting Block %i of %i with\n\n%s\n',motivational_str,task_str,blok,n.blocks,scorereport);
        otherwise
            hitxt = sprintf('Great job! You have just finished %s.\n', type);
    end
    [~,ny]=center_print(hitxt,'center');
    flip_key_flip(scr,'continue',ny,color,experimenter_needed);
    
    
    if strcmp(type, 'Testing')
        %load top scores
        load top_ten
        ranking = 11 - sum(blockscore>=top_ten.(R.category_type).scores); % calculate current ranking
        
        if ranking < 11
            top_ten.(R.category_type).scores = [top_ten.(R.category_type).scores(1:(ranking-1));  blockscore;  top_ten.(R.category_type).scores(ranking:9)];
            for m = 10:-1:ranking+1
                top_ten.(R.category_type).initial{m} = top_ten.(R.category_type).initial{m-1};
            end
            top_ten.(R.category_type).initial{ranking} = subject_name;
            hitxt=sprintf('\n\nCongratulations! You made the %sTop Ten!\n\n',task_str);
        else
            hitxt='\n\n\n\n';
        end
        
        if ~any(strfind(subject_name,'test'))
            save top_ten top_ten;
        end
        
        [nx,ny] = center_print(sprintf('%sYour score for Testing Block %i of %i: %.1f%%\n\n%sTop Ten:\n\n',hitxt,blok,n.blocks,blockscore,task_str),-80);
        
        for j = 1:10
            [nx,ny] = center_print(sprintf('%i) %.1f%%    %s\n',j,top_ten.(R.category_type).scores(j),top_ten.(R.category_type).initial{j}),ny,[],scr.cx*.8 - (j==10)*20);
        end
        
        % instructions below top ten
        experimenter_needed = false;
        if blok ~= n.blocks
            if ~test_feedback
                hitxt = '\nPlease take a short break.\n\nYou may begin the next Category Training\n\n';
            else
                hitxt = '\nPlease take a short break.\n\nYou may begin the next Testing Block\n\n';
            end
        else
            if ~final_task
                if new_subject
                    hitxt = sprintf('\nYou''re done with %s\n\n\nPlease go get the experimenter from the other room!',task_str);
                    experimenter_needed = true;
                else
                    switch task_str
                        case 'Task A '
                            hitxt = '\nYou''re done with Task A.\n\n\nYou may begin Task B\n\n';
                        case 'Task B '
                            hitxt = '\nYou''re done with Task B.\n\n\nYou may begin Task A\n\n';
                    end
                end
            else
                hitxt='\n\n\n\nYou''re done with the experiment.\n\nThank you for participating!';
                experimenter_needed = true;
            end
        end
        
        [nx,ny] = center_print(hitxt,ny);
        if blok ~= n.blocks || (blok == n.blocks && ~final_task && ~new_subject)
            [nx,ny] = center_print('in ',ny,[],scr.cx-9.2*P.pxPerDeg); %570: 240
            countx=nx; county=ny;
            [nx,ny] = center_print('seconds, but you may take a\n\n',county,[],countx+1.7*P.pxPerDeg); % 3 spaces: 5 spaces
            [nx,ny] = center_print(['longer break and leave the room\n\n'...
                'or walk around.'],ny,[],[],50);
            countdown(scr,color,countx,county);
        end
        
        flip_key_flip(scr,'continue',ny,color,experimenter_needed);
        
    end
    
    
catch
    % I think these lines just indicate where an error occurred when you look back at the data.
    responses.tf(section, trial) = -1;
    responses.c(section, trial) = -1;
    responses.conf(section, trial) = -1;
    responses.rt(section, trial) = -1;
    if two_response
        responses.rtConf(section,trial) = -1;
    end
    
    
    psychrethrow(psychlasterror)
    save responses
    flag = 1;
    
end

    function [nx,ny]=center_print(str,y,text_color,x, wrapat)
        if ~exist('text_color','var') | isempty(text_color)
            text_color = color.wt;
        end
        if ~exist('x','var') | isempty(x)
            x = 'center';
        end
        if ~exist('wrapat') | isempty(wrapat)
            wrapat=[];
        end
        [nx,ny] = DrawFormattedText(scr.win, str, x, y, text_color, wrapat);
    end

end

function str = fractionizer(numerator,denominator)
n = {'one','two','three','four','five'};
d = {'','half','third','quarter','fifth'};

if numerator > denominator
    warning('numerator greater than denominator!')
    str = '';
else
    if numerator==denominator
        str = 'all';
    else
        if numerator==1
            str = sprintf('%s %s',n{1}, d{denominator});
        else
            str = sprintf('%s %ss',n{numerator}, d{denominator});
        end
    end
end
end