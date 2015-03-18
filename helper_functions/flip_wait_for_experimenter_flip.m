function flip_wait_for_experimenter_flip(key, scr)
Screen('Flip', scr.win);
while true
    [~, ~, keyCode] = KbCheck;
    if keyCode(key)
        return
    end
end
Screen('Flip', scr.win);
end