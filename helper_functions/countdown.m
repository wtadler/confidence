function countdown(scr,color,x,y)
for i=1:scr.countdown_time+1;
    Screen('FillRect',scr.win,color.bg,[x y x+1.6*scr.fontsize y+1.2*scr.fontsize]) %timer background
    DrawFormattedText(scr.win,[num2str(scr.countdown_time+1-i) '  '],x,y,color.wt);
    Screen('Flip',scr.win,[],1); % flip to screen without clearing
    WaitSecs(1);
end
end
