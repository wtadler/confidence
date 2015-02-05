function [responses, flag, blockscore] = run_exp(n, R, t, scr, color, P, type, blok, new_subject_flag, varargin)

attention_manipulation = false;
if length(varargin) == 1
    R2 = varargin{1};
    attention_manipulation = true;
end

%%%Run trials (Training or Test)%%%
try
    flag = 0;
    %binary matrix of correct/incorrect responses
    responses.tf = zeros(n.sections, n.trials);
    responses.c = zeros(n.sections, n.trials);
    responses.conf = zeros(n.sections, n.trials);
    responses.rt = zeros(n.sections, n.trials);
    
    %abc = {'a','b','c'};
    
    for section = 1:n.sections
        for trial = 1:n.trials
            Screen('DrawTexture', scr.win, scr.cross);
            Screen('Flip', scr.win);
            WaitSecs(t.betwtrials/1000);
            
            cval = R.trial_order{blok}(section, trial); %class
            
            stim = struct;
            stim(1).ort = R.draws{blok}(section, trial);        %orientation
            stim(1).cur_sigma = R.sigma{blok}(section, trial);  %contrast
            stim(1).phase = R.phase{blok}(section, trial);      %phase (not needed by ellipse)
            
            if attention_manipulation
                stim(2).ort = R2.draws{blok}(section, trial);
                stim(2).cur_sigma = R2.sigma{blok}(section, trial);
                stim(2).phase = R2.phase{blok}(section, trial);
                
                % DISPLAY CUE ARROW
                if R2.cue{blok}(section, trial) == 1
                    Screen('DrawTexture', scr.win, scr.arrowL);
                elseif R2.cue{blok}(section, trial) == 2
                    Screen('DrawTexture', scr.win, scr.arrowR);
                end
                Screen('Flip', scr.win);
                WaitSecs(t.attention_cue/1000);
            end
            
            
            if strcmp(P.stim_type, 'gabor')
                r_gabor(P, scr, t, stim); % haven't yet added phase info to this function
            elseif strcmp(P.stim_type, 'grate')
                grate(P, scr, t, stim);
            elseif strcmp(P.stim_type, 'ellipse')
                ellipse(P, scr, t, stim); % ellipse doesn't need phase info
            end
            
            if attention_manipulation
                % DISPLAY PROBE
                if R2.probe{blok}(section, trial) == 1
                    Screen('DrawTexture', scr.win, scr.gray_arrowL);
                elseif R2.probe{blok}(section, trial) == 2
                    Screen('DrawTexture', scr.win, scr.gray_arrowR);
                end
                Screen('Flip', scr.win);
            end
            
            
            %Screen('Flip', scr.win); % unnecessary flip here?
            clc;
            fprintf('blok %g, section %g, trial %g\n\n',blok,section,trial)
            %subject input
            startsecs = GetSecs;
            resp = 0;
            while resp == 0;
                [~, secs, keyCode] = KbCheck;
                
                %To quit script, press x,z ONLY simultaneously
                %if keyCode(scr.keyx) && keyCode(scr.keyz) && sum(keyCode)==2
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
            responses.tf(section, trial) = (resp == cval);
            responses.c(section, trial) = resp;
            if ~strcmp(type, 'Training') % if not in non-conf training
                responses.conf(section, trial) = conf;
            end
            responses.rt(section,trial) = secs - startsecs;
            
            if strcmp(type, 'Training') || strcmp(type,'Confidence Training') %to add random feedback during test: || rand > .9 %mod(sum(sum(tfresponses)) ,10)==9
                %feedback
                WaitSecs(t.pause/1000);
                if resp == cval
                    status = 'Correct!';
                    stat_col = color.grn;
                else
                    status = 'Incorrect!';
                    stat_col = color.red;
                end
                
                if strcmp(type,'Training')
                    [~,ny]=DrawFormattedText(scr.win,['You said: Category ' num2str(resp)],'center',scr.cy-50,color.wt);
                    [~,ny]=DrawFormattedText(scr.win,['\n' status],'center',ny+10,stat_col);
                else % as in confidence training
                    [~,ny]=DrawFormattedText(scr.win,['You said: Category ' num2str(resp) ' with ' confstr ' confidence.'],'center',scr.cy-50,color.wt);
                end
                
                Screen('Flip',scr.win);
                
                WaitSecs(t.feedback/1000);
                
            end
            
        end
        
        %if another section in the same block immediately follows
        if section ~= n.sections
            [~,scorereport]=calcscore(responses,n.trials);
            if strcmp('Training', type) && blok == 1 % partway through training block 1. when experimenter should leave room
                midtxt = sprintf('Very good! You got %s\n\nYou have completed %s of Category Training.',scorereport,fractionizer(section,n.sections));
                str = 'continue';
            elseif strcmp('Training', type) % this isn't happening right now;
                midtxt = ['Coming up: Testing Block ' num2str(section+1) '\n\n'...
                    'Training Block ' num2str(blok) '\n\n\n\n'];
                str = 'begin';
            else
                midtxt = sprintf('You have completed %s of Testing Block %i of %i.',fractionizer(section,n.sections),blok,n.blocks)
                str = 'continue';
            end
            
            [~,ny]=DrawFormattedText(scr.win,midtxt,'center','center',color.wt);
            flip_pak_flip(scr,ny,color,str);
            
        end
        
        
    end
    [blockscore,scorereport]= calcscore(responses,n.sections*n.trials);
    if strcmp(type, 'Training') && blok == 1 && strcmp(new_subject_flag,'y')
        hitxt = ['Great job! You just got ' scorereport '\n\n\n'...
            'Please go get the experimenter from the other room!'];
    elseif strcmp(type,'Confidence Training')
        hitxt = ['Great job! You have just finished Confidence Training.\n\n'...
            'Coming up: Testing Block 1 of 3.'];
        str = 'begin'
    elseif strcmp(type,'Training')
        hitxt = ['Nice work! You just got ' scorereport ...
            '\n\nComing up: Testing Block ' num2str(blok) ' of ' num2str(n.blocks)];
        str = 'begin';
    elseif strcmp(type, 'Test')
        hitxt = ['Great! You''ve just finished Testing Block ' num2str(blok) ' with\n\n' scorereport];
        str = 'continue';
    end
    
    [~,ny]=DrawFormattedText(scr.win,hitxt,'center','center',color.wt);
    
    if strcmp(type,'Training') && blok == 1 && strcmp(new_subject_flag,'y') % this is the situation when the experimenter comes in.
        Screen('Flip', scr.win);
        WaitSecs(2);
        KbWait;
        Screen('Flip', scr.win);
    else % print the directions if the experimenter isn't supposed to come in.
        flip_pak_flip(scr,ny,color,str)
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