function  grate(P, scr, t, stim, demo)
%%% Create and and display drifting grating %%%

if ~exist('demo', 'var')
    demo = false;
end

nStimuli = length(stim);
if nStimuli == 4
    cross_rot = 45;
else
    cross_rot = 0;
end

for i = 1:nStimuli
    % create texture
    texture{i} = CreateProceduralGabor(scr.win, P.grateAlphaMaskSize, P.grateAlphaMaskSize, [], [0.5 0.5 0.5 0.0],1,0.5);
    
    % define center point.
    % one stimulus: center_point{1} = screen center
    % two stimuli: center_point{1,2} = screen center ± stim_dist
    if nStimuli == 1
        center_point{i} = [scr.cx scr.cy];
    elseif nStimuli == 2
        center_point{i} = [scr.cx scr.cy] + (2*i-3) * [P.stim_dist 0];
    elseif nStimuli == 4
        center_point{i} = [scr.cx scr.cy] + rotateCoords([P.stim_dist/sqrt(2); -P.stim_dist/sqrt(2)], -90*i)';
    end
   
    w = P.grateAlphaMaskSize;
    destRect{i} = [center_point{i} center_point{i}] +[-w/2 -w/2 w/2 w/2];
end

%% show it

frame = 0;
startTime = GetSecs;
lastTime = 0;

while (lastTime - startTime)*1000 < t.pres;
    frame = frame+1;

    for i = 1:nStimuli
        Screen('DrawTexture',scr.win,texture{i},[],destRect{i},90-stim(i).ort,[],[],[],[],kPsychDontDoRotation, [stim(i).phase+P.grateDt*P.grateSpeed*360*frame, P.grateSpatialFreq, P.grateSigma, stim(i).cur_sigma, P.grateAspectRatio, 0, 0, 0]);
    end
    
    if nStimuli ~= 1
        Screen('DrawTexture', scr.win, scr.cross, [], [], cross_rot); % display fixation cross when there are multiple stimuli.
    end

    
    if ~demo
        lastTime = Screen('Flip',scr.win, startTime + frame*P.grateDt);
    else
        return
    end
end

for i = 1:nStimuli
    Screen('Close', texture{i});
end

if nStimuli==1 % don't show a blank screen in the attention task
    Screen('Flip', scr.win);
end