function flip_pak_flip(scr,ny,color,str,varargin)
% pak: Press Any Key...
% flips any drawn text, waits, flips PAK text, waits for input, flips and
% waits.

% define defaults
initial_wait=2;
assignopts(who,varargin);

Screen('Flip',scr.win,[],1);
WaitSecs(initial_wait);

DrawFormattedText(scr.win,['\n\n\nPress any key to ' str '.\n\n'],'center',ny,color.wt);

Screen('Flip',scr.win);
WaitSecs(.2);
KbWait;
Screen('Flip',scr.win);
WaitSecs(.5);