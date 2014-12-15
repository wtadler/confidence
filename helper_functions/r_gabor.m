function r_gabor(P, scr, t, stim, varargin)
if length(varargin) == 1
    dest = varargin{1};
end

%%%Create and and display gabor patch%%%
% NOTE THAT THIS DOESN'T RANDOMIZE PHASE. WTA
ort = stim.ort + 90; % this is to try to make things more similar after adding ellipse
cur_sigma = stim.cur_sigma;
%phase = stim.phase;

% set image size (cut off after center +/- 3*sigma)
imsize = round(6*[P.gabor_wavelength P.gabor_wavelength]);

% generate cosine pattern
X = ones(imsize(1),1)*[-(imsize(2)-1)/2:1:(imsize(2)-1)/2];
Y =[-(imsize(1)-1)/2:1:(imsize(1)-1)/2]' * ones(1,imsize(2));

cospattern = cos(2.*pi.*(P.gabor_period).* (cos(ort*pi/180).*X ...
   + sin(ort*pi/180).*Y)  ...
   - 0*P.gabor_phase*ones(imsize) );

% convolve with gaussian
rad   = (imsize-1)/2;
[x,y] = meshgrid(-rad(2):rad(2),-rad(1):rad(1));
arg   = -(x.*x + y.*y)/(2*(P.gabor_wavelength^2));
filt     = exp(arg);    %set up Bivariate Distribution
norm = sum(filt(:));
if norm ~= 0,
    filt  = filt/norm; %normalize it
end;

%filt = fspecial('gaussian', imsize, sigma);
filt = filt/max(max(filt));
im = cospattern .* filt;

%adjust brightness, contrast
gabor_max = P.gabor_mean * (1+cur_sigma);
gabor_min = P.gabor_mean - (gabor_max - P.gabor_mean);
im = (im+1)/2;
im = im * (gabor_max-gabor_min) + gabor_min;

gabortex = Screen('MakeTexture', scr.win, im);

if exist('dest') %in the demo, destination specified
    Screen('DrawTexture', scr.win, gabortex, [], dest);
    Screen('Close', gabortex);
else %regular trials
    Screen('DrawTexture', scr.win, gabortex);
    Screen('Close', gabortex);
    Screen('Flip', scr.win);
    WaitSecs(t.pres/1000);
    Screen('Flip', scr.win);
end