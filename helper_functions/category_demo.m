for category = 1 : 2
    DrawFormattedText(scr.win, sprintf('Examples of Category %i stimuli:', category), 'center', scr.cy-60, color.wt);
    Screen('Flip', scr.win);
    WaitSecs(2.5);
    
    for i = 1:nDemoTrials
        
        Screen('DrawTexture', scr.win, scr.cross);
        Screen('Flip', scr.win);
        WaitSecs(Demo.t.betwtrials/1000);
        
        stim.ort = stimulus_orientations(Test.category_params, category, 1, category_type);
        
        if strcmp(P.stim_type, 'gabor')
            r_gabor(P, scr, Demo.t, stim); % haven't yet added phase info to this function
        elseif strcmp(P.stim_type, 'grate')
            stim.phase = 360*rand;
            grate(P, scr, Demo.t, stim);
        elseif strcmp(P.stim_type, 'ellipse')
            ellipse(P, scr, Demo.t, stim);
        end
    end
    
    flip_key_flip(scr,'continue','center',color,new_subject);
    
end