function flip_key_flip(scr, string, y, color, experimenter_needed, varargin)
% flips what's drawn on screen.
% if experimenter_needed is true, waits for a specific keypress (enter by
% default), and then clears screen
% if experimenter_needed is false, tells subject to press any key. clears
% screen when subject presses a key.

initial_wait=2;
key='keyenter';
assignopts(who,varargin);

Screen('Flip',scr.win,[],1);

if experimenter_needed
    WaitSecs(initial_wait);
    while true
        [~,keyCode] = KbWait(-1);
        if keyCode(scr.(key))
            break
        end
    end
    Screen('Flip',scr.win);
else
    WaitSecs(initial_wait);
    DrawFormattedText(scr.win,['\n\n\nPress any key to ' string '.\n\n'],'center',y,color.wt);
    Screen('Flip',scr.win);
    WaitSecs(.15);
    KbWait(-1);
    Screen('Flip',scr.win);
    WaitSecs(.5);
end
end