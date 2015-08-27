function  ellipse(P, scr, t, stim, varargin)
%%% Create and and display ellipse with given area and eccentricity.
% major axis is on the vertical with rot = 0. rotates clockwise by rot.
% fgcol is the color of the ellipse.
% bgcol is the color of the background.
% adapted from drawEllipse.m

ort = stim.ort+90;
cur_sigma = stim.cur_sigma;

im = drawEllipse(P.ellipseAreaPx, cur_sigma, ort, P.ellipseColor, scr.bg);

%% show it
% some shortcuts
centerX = scr.cx;
centerY = scr.cy;

textureW = size(im,2);
textureH = size(im,1);

texture = Screen('MakeTexture',scr.win,im);

destRect = ceil([centerX centerY centerX centerY] + [-textureW -textureH textureW textureH] / 2);

if length(varargin) == 1
    destRect = varargin{1};
end


if length(varargin) == 1 || ~isstruct(t) % either of these would indicate demo
    Screen('DrawTexture', scr.win, texture, [], destRect);
    Screen('Close', texture);
else %regular trials
Screen('DrawTexture', scr.win, texture, [], destRect);
Screen('Close', texture);
Screen('Flip', scr.win);
WaitSecs(t.pres/1000);
Screen('Flip', scr.win);
end


%Screen('Close', texture);
%Screen('FillRect', scr.win, P.bgColor, scr.rect);
%Screen('Flip', scr.win);