function [responses, flag] = run_exp(n, R, t, scr, color, P, type, blok, new_subject, task_str, final_task, subject_name, varargin)    

if length(varargin) == 1
    R2 = varargin{1};
    attention_manipulation = true;
else
    attention_manipulation = false;
end

    flag = 0;
    responses.c = zeros(n.sections, n.trials); % cat response
    responses.conf = zeros(n.sections, n.trials); % conf response
    responses.rt = zeros(n.sections, n.trials); % rt
    %bool matrix of correct/incorrect responses
    responses.tf = zeros(n.sections, n.trials);
    
    %%% Text before trials %%%
    switch type
        case 'Training'
            str='Coming up: Category Training';
        case 'Confidence Training'
            str=['Let''s get some quick practice with confidence ratings.\n\n'...
                'Coming up: ' task_str 'Confidence Training'];
        case 'Attention Training'
            str=['Let''s get some practice with the structure of this task.\n\n'...
                'In the center of the screen there will be a cross, which you should focus on.\n'...
                'The targets, which are striped circles will appear to both the left and right of the cross.\n'...
                'The targets will be tilted either clockwise or counter-clockwise from horizontal.\n\n'...
                'Before the targets appear, the left or right arm of the cross will turn white,\n'...
                'cuing you to focus your attention on either the left or right target.\n'...
                'After the targets are presented, the left or right arm will turn gray,\n'...
                'telling you which target to report.\n\n'...
                'Your task is to report whether that target was tilted clockwise or counter-clockwise,\n'...
                'as well as your confidence in that judgment.\n'...
                '70% of the time, the initial cue will correctly indicate the target you are asked to report,\n'...
                'and 30% of the time, you''ll be asked to report the uncued target.\n\n'...
                'Overall, payting attention to the target indicated by the initial cue will help you\n'...
                'on this task, so you should do your best to focus your attention on the cued target.\n\n'...
                'Remember to always keep your eyes on the cross and just shift your mental focus of attention\n'...
                'to the cued target.']
            Screen('TextSize', scr.win, round(scr.fontsize*.7));
                
        case 'Test'
            str=['Coming up: ' task_str 'Testing Block ' num2str(blok) ' of ' num2str(n.blocks)];
    end
    [~,ny]=center_print(str,'center');
    Screen('TextSize', scr.win, scr.fontsize); % reset fontsize if made small for attention training.
    flip_key_flip(scr,'begin',ny,color, false);
    
    
    %%%Run trials %%%
try
    for section = 1:n.sections
        for trial = 1:n.trials
            
            stim = struct;
            stim(1).ort = R.draws{blok}(section, trial);        %orientation
            stim(1).cur_sigma = R.sigma{blok}(section, trial);  %contrast
            stim(1).phase = R.phase{blok}(section, trial);      %phase (not needed by ellipse)

            Screen('DrawTexture', scr.win, scr.cross);
            t0 = Screen('Flip', scr.win);

            if ~attention_manipulation
                WaitSecs(t.betwtrials/1000);
            elseif attention_manipulation 
                stim(2).ort = R2.draws{blok}(section, trial);
                stim(2).cur_sigma = R2.sigma{blok}(section, trial);
                stim(2).phase = R2.phase{blok}(section, trial);
                
                % DISPLAY SPATIAL ATTENTION CUE
                if R2.cue{blok}(section, trial) == 1
                    Screen('DrawTexture', scr.win, scr.cueL);
                elseif R2.cue{blok}(section, trial) == 2
                    Screen('DrawTexture', scr.win, scr.cueR);
                end
                t_cue = Screen('Flip', scr.win, t0 + t.betwtrials/1000);
                Screen('DrawTexture', scr.win, scr.cross);
                t_cue_off = Screen('Flip', scr.win, t_cue + t.cue_dur/1000);
                
                %%% should make this timing exact by interfacing with grate
                WaitSecs(t.cue_target_isi/1000);
            end
            
            
            if strcmp(P.stim_type, 'gabor')
                r_gabor(P, scr, t, stim); % haven't yet added phase info to this function
            elseif strcmp(P.stim_type, 'grate')
                grate(P, scr, t, stim);
            elseif strcmp(P.stim_type, 'ellipse')
                ellipse(P, scr, t, stim); % ellipse doesn't need phase info
            end
            
            if attention_manipulation
                % DISPLAY RESPONSE CUE (i.e. probe)
                %%% should make this timing exact by interfacing with grate
                Screen('DrawTexture', scr.win, scr.cross);
                t_target_off = Screen('Flip', scr.win);
                if R2.probe{blok}(section, trial) == 1
                    Screen('DrawTexture', scr.win, scr.resp_cueL);
                    cval = R.trial_order{blok}(section, trial);
                elseif R2.probe{blok}(section, trial) == 2
                    Screen('DrawTexture', scr.win, scr.resp_cueR);
                    cval = R2.trial_order{blok}(section, trial);
                end
                t_resp_cue = Screen('Flip', scr.win, t_target_off + t.cue_target_isi/1000);
            else
                cval = R.trial_order{blok}(section, trial); %class
            end
            
            
            clc;
            fprintf('blok %g, section %g, trial %g\n\n',blok,section,trial)
            %subject input
            t0 = GetSecs;
            resp = 0;
            while resp == 0;
                [~, tResp, keyCode] = KbCheck;
                
                %To quit script, press insert and enter ONLY
                %simultaneously
                if keyCode(scr.keyinsert) && keyCode(scr.keyenter) && sum(keyCode)==2
                    error('You cancelled the script by pressing the insert and enter keys simultaneously.')
                end
                
                if strcmp(type, 'Training')
                    if keyCode(scr.key5) % cat 1
                        resp = 1;
                    elseif keyCode(scr.key6) % cat 2
                        resp = 2;
                    end
                else % if not in non-conf training
                    if keyCode(scr.key1) || keyCode(scr.key2) || keyCode(scr.key3) || keyCode(scr.key4) %cat 1 keys
                        resp = 1;
                    elseif keyCode(scr.key7) || keyCode(scr.key8) || keyCode(scr.key9) || keyCode(scr.key10) %cat 2 keys
                        resp = 2;
                    end
                    
                    if keyCode(scr.key1) || keyCode(scr.key10)
                        conf = 4;
                        confstr = 'VERY HIGH';
                    elseif keyCode(scr.key2) || keyCode(scr.key9)
                        conf = 3;
                        confstr = 'SOMEWHAT HIGH';
                    elseif keyCode(scr.key3) || keyCode(scr.key8)
                        conf = 2;
                        confstr = 'SOMEWHAT LOW';
                    elseif keyCode(scr.key4) || keyCode(scr.key7)
                        conf = 1;
                        confstr = 'VERY LOW';
                    end
                end
            end
            
            %record 1 if correct, 0 if incorrect
%             fprintf('cat %d - ACC %d\n', resp, resp==cval) % for debugging
            responses.tf(section, trial) = (resp == cval);
            responses.c(section, trial) = resp;
            if ~strcmp(type, 'Training') % if not in non-conf training
                responses.conf(section, trial) = conf;
            end
            responses.rt(section,trial) = tResp - t0;
            
            if strcmp(type, 'Training') || strcmp(type,'Confidence Training') || strcmp(type,'Attention Training') %to add random feedback during test: || rand > .9 %mod(sum(sum(tfresponses)) ,10)==9
                %feedback
                if resp == cval
                    status = 'Correct!';
                    stat_col = color.grn;
                else
                    status = 'Incorrect!';
                    stat_col = color.red;
                end
                
                switch type
                    case 'Training'
                        [~,ny]=center_print(sprintf('You said: Category %i',resp),scr.cy-50);
                        [~,ny]=center_print(sprintf('\n%s', status),ny+10,stat_col);
                    case 'Confidence Training'
                        [~,ny]=center_print(sprintf('You said: Category %i with %s confidence.',resp,confstr),scr.cy-50);
                    case 'Attention Training'
                        if resp == 1; str = 'LEFT'; else str = 'RIGHT'; end
                        [~,ny]=center_print(sprintf('You said: %s with %s confidence.',str,confstr),scr.cy-50);
                        [~,ny]=center_print(sprintf('\n%s',status),ny+10,stat_col);
                end
                
                Screen('Flip',scr.win, tResp+t.pause/1000);
                
                WaitSecs(t.feedback/1000);
                
            end
            
        end
        
        %if another section in the same block immediately follows
        if section ~= n.sections
            [~,scorereport]=calcscore(responses,n.trials);
            if strcmp('Training', type) && blok == 1 % partway through training block 1. when experimenter should leave room
                midtxt = sprintf('You got %s\n\nYou have completed\n\n%s of %sCategory Training.',scorereport,fractionizer(section,n.sections), task_str);
                str = 'continue';
            else
                midtxt = sprintf('You have completed\n\n%s of %sTesting Block %i of %i.',fractionizer(section,n.sections),task_str,blok,n.blocks);
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
        motivational_str = 'Please try a bit harder!\n\nIf you are confused about the task,\n\nplease talk to the experimenter.'
    end
    
    experimenter_needed = false;
    switch type
        case 'Training'
            % strcmp(task_str,'') is a standin for nExperiments == 1.
            if blok == 1 && new_subject && (strcmp(task_str,'') | (~strcmp(task_str,'') && ~final_task))
                hitxt = sprintf('You just got %s\n\n\nPlease go get the experimenter from the other room!',scorereport);
                experimenter_needed = true;
            else
                hitxt = sprintf('%s\n\nYou just got %s\n',motivational_str,scorereport);
            end
        case 'Confidence Training'
            hitxt = 'Great job! You have just finished Confidence Training.\n';
        case 'Attention Training'
            hitxt = sprintf('%s\n\nYou have just finished Attention Training with\n\n%s.\n',motivational_str,scorereport);
        case 'Test'
            hitxt = sprintf('%s\n\nYou''ve just finished %sTesting Block %i of %i with\n\n%s.\n',motivational_str,task_str,blok,n.blocks,scorereport);
    end
    [~,ny]=center_print(hitxt,'center');
    flip_key_flip(scr,'continue',ny,color,experimenter_needed);

    
    if strcmp(type, 'Test')
        %load top scores
        load top_ten
        ranking = 11 - sum(blockscore>=top_ten.(R.category_type).scores); % calculate current ranking
 
        if ranking < 11
            top_ten.(R.category_type).scores = [top_ten.(R.category_type).scores(1:(ranking-1));  blockscore;  top_ten.(R.category_type).scores(ranking:9)];
            for m = 10:-1:ranking+1
                top_ten.(R.category_type).initial{m} = top_ten.(R.category_type).initial{m-1};
            end
            top_ten.(R.category_type).initial{ranking} = subject_name;
            hitxt=sprintf('\n\nCongratulations! You made the %sTop Ten!\n\n',task_str);;
        else
            hitxt='\n\n\n\n';
        end

        if ~any(strfind(subject_name,'test'))
            save top_ten top_ten;
        end

        [nx,ny] = center_print(sprintf('%sYour score for Testing Block %i of %i: %.1f\n\n%sTop Ten:\n\n',hitxt,blok,n.blocks,blockscore,task_str),-90);
        
        for j = 1:10
            [nx,ny] = center_print(sprintf('%i) %.1f%%    %s\n',j,top_ten.(R.category_type).scores(j),top_ten.(R.category_type).initial{j}),ny,[],scr.cx*.8 - (j==10)*20);
        end

        % instructions below top ten
        experimenter_needed = false;
        if blok ~= n.blocks
            hitxt = '\nPlease take a short break.\n\n\nYou may begin the next Category Training\n\n';
        else
            if ~final_task
                if new_subject
                    hitxt = sprintf('\nYou''re done with %s\n\n\nPlease go get the experimenter from the other room!',task_str);
                    experimenter_needed = true;
                else
                    hitxt = sprintf('You''re done with %s\n\n\nYou may begin Task %s\n\n',task_str,other_task_letter);
                end
            else
                hitxt='\n\n\n\nYou''re done with the experiment.\n\nThank you for participating!';
                experimenter_needed = true;
            end
        end
        
        [nx,ny] = center_print(hitxt,ny);
        if blok ~= n.blocks || (blok == n.blocks && ~final_task && ~new_subject)
            [nx,ny] = center_print('in ',ny,[],scr.cx-570);
            countx=nx; county=ny;
            [nx,ny] = center_print('   seconds, but you may take a\n\n',county,[],countx);
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