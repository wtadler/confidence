function  grate(P, scr, t, stim)
%%% Create and and display grating (moving gabor) %%%

nStim = length(stim);

for i = 1:nStim
    % create texture
    texture{i} = CreateProceduralGabor(scr.win, P.grateAlphaMaskSize, P.grateAlphaMaskSize, [], [0.5 0.5 0.5 0.0],1,0.5);
    
    % define center point.
    % one stimulus: center_point{1} = screen center
    % two stimuli: center_point{1,2} = screen center ± stim_dist
    center_point{i} = [scr.cx scr.cy scr.cx scr.cy] + (nStim-1) * (2*i-3) * [P.stim_dist 0 P.stim_dist 0];
    
    w = P.grateAlphaMaskSize;
    destRect{i} = center_point{i}+[-w/2 -w/2 w/2 w/2];
end

%% show it

frame = 0;
startTime = GetSecs;
lastTime = 0;

while (lastTime - startTime)*1000 < t.pres;
    WaitSecs('UntilTime', startTime + frame*P.grateDt);
    frame = frame+1;

    for i = 1:nStim
        Screen('DrawTexture',scr.win,texture{i},[],destRect{i},90-stim(i).ort,[],[],[],[],kPsychDontDoRotation, [stim(i).phase+P.grateDt*P.grateSpeed*360*frame, P.grateSpatialFreq, P.grateSigma, stim(i).cur_sigma, P.grateAspectRatio, 0, 0, 0]);
        if nStim ~= 1
            Screen('DrawTexture', scr.win, scr.cross); % display fixation cross when there are multiple stimuli.
        end
    end
    lastTime = Screen('Flip',scr.win);
end

for i = 1:nStim
    Screen('Close', texture{i});
end

if numel(stim)==1 % don't show a blank screen in the attention task
    Screen('Flip', scr.win);
end