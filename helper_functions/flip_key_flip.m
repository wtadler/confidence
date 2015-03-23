function flip_key_flip(scr, string, y, color, experimenter_needed, varargin)

initial_wait=2;
key='enter';
assignopts(who,varargin);

Screen('Flip',scr.win,[],1);


if experimenter_needed
    while true
        [~,~,keyCode] = KbWait;
        if keyCode(scr.(key))
            return
        end
    end
else
    WaitSecs(initial_wait);
    DrawFormattedText(scr.win,['\n\n\nPress any key to ' str '.\n\n'],'center',ny,color.wt);
    Screen('Flip',scr.win);
    WaitSecs(.2);
    KbWait;
    Screen('Flip',scr.win);
end

WaitSecs(.5);

end
