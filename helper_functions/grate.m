function  grate(P, scr, t, stim)
%%% Create and and display grating (moving gabor) %%%

nStim = length(stim);

% 
% for i = 1:nStim
%     ort{i} = stim{i}.ort;
%     cur_sigma{i} = cur_sigma{i}.ort;
%     phase{i} = phase{i}.ort;
% end
    
%% Setup Screen and alpha blending
Screen('BlendFunction',scr.win,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
%% make the texture

% texture needs to be a bit larger because we're going to move it under the
% aperture to generate the motion
textureSize = ceil(P.diskSize + P.period);

% rotate co-ordinate system
phi = 2*pi*P.spatialFreq * (1:textureSize);

for i = 1:nStim
    grat = 127.5 + stim(i).cur_sigma * 126 * sin(phi + P.initialPhase); %'120' (col. 28) changed to '126' on 10/20 by RG
    
    % color grating
    color = permute(P.color,[2 3 1]);
    grat = bsxfun(@times,color,grat);
    
    % create texture
    texture{i} = Screen('MakeTexture',scr.win,grat);
    
    % define center point.
    % one stimulus: center_point{1} = screen center
    % two stimuli: center_point{1,2} = screen center ± stim_dist
    center_point{i} = [scr.cx scr.cy scr.cx scr.cy] + (nStim-1) * (2*i-3) * [P.stim_dist 0 P.stim_dist 0];

end

% center_point{1} = [scr.cx scr.cy scr.cx scr.cy];
% if nStim==2
%     stim_dist = round(P.attention_stim_spacing * P.pxPerDeg); % distance from center in pixels
%     center_point{1} = center_point - [stim_dist 0 stim_dist 0];
%     center_point{2} = center_point + [stim_dist 0 stim_dist 0];
% end


%% show it

%phi2 = P.initialPhase; % constant phase
%phi2 = 360*rand; % random phase % now obsolete since phase is generated in
%setup_exp_order

frame = 0;
startTime = GetSecs;
lastTime = 0;

while (lastTime - startTime)*1000 < t.pres;
    WaitSecs('UntilTime', startTime + frame*P.dt);
    frame = frame+1;

    Screen('FillRect', scr.win, P.bgColor, scr.rect);

    for i = 1:nStim

        ort = stim(i).ort;
        if frame == 1
            phase{i} = stim(i).phase;
        end
        
        % move grating
        u = mod(phase{i},P.period) - P.period/2;
        xInc = -u * sin(ort/180*pi);
        yInc = u * cos(ort/180*pi);
        ts = textureSize;
        destRect = center_point{i} + [-ts -ts ts ts] / 2 ...
            + [xInc yInc xInc yInc];
        phase{i} = phase{i} + P.speed;
        
        % draw grating
        Screen('DrawTexture',scr.win,texture{i},[],destRect,ort+90);
        
        % draw circular aperture
        alphaRect = center_point{i} + [-1 -1 1 1] * P.alphaMaskSize;
        % fix alphamask alpharect being too big. 
        Screen('DrawTexture',scr.win,scr.alphaMask,[],alphaRect,ort+90);
        
        if nStim ~= 1
            Screen('DrawTexture', scr.win, scr.cross); % display fixation cross when there are multiple stimuli.
        end

    end
    
    % fixation
    lastTime = Screen('Flip',scr.win);

end
for i = 1:nStim
    Screen('Close', texture{i});
end

Screen('FillRect', scr.win, P.bgColor, scr.rect);
Screen('Flip', scr.win);