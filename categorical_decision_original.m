%Ryan George
%Theoretical Neuroscience Lab, Baylor College of Medicine
%Categorial decisions pilot program

function categorical_decision

% persistent have_done
% if numel(have_done) == 0 %if no previous session has been run today
%     %reset random seed to clock value (see help RandStream)
%     s = RandStream.create('mt19937ar','seed',sum(100*clock));
%     RandStream.setDefaultStream(s);
%     have_done = 1;
% end

fprintf('To abort the program, press x and z buttons simultaneously \n \n')
%initial = input('Please enter your initials \n\n', 's'); % 's' returns entered text as a string
%room_letter = input('Please enter the room letter (lowercase) \n\n', 's');
initial = 'wta';
room_letter = 'test';

if strcmp(room_letter,'b')||strcmp(room_letter,'r')
    screen_width = 16.25 * 2.54;         %physical screen size (in*(2.54cm/in) = cm)
    scr.key1 = 97; scr.key2 = 98;
elseif strcmp(room_letter,'lab')
    screen_width = 18.7 * 2.54;         %physical screen size (in*(2.54cm/in) = cm)
    scr.key1 = 35; scr.key2 = 40;
else 
    screen_width = 33.2509 %WTA's MBP, used to be 15 * 2.54; 
    %scr.key1 = 97; scr.key2 = 98;
    scr.key1 = 30; scr.key2 = 31; % These are the keycodes for WTA's laptop
end

% Close previous figure plots:
close all;
tic

%Paradigm Parameters stored (mainly) in the two structs 'Training' and 'Test'
P.stim_type = 'grate';  %options: 'gabor' or 'dots'

color.bg = [0.4902    0.5647    0.5137];
color.wt = [1 1 1];
color.bk = [0 0 0];
color.red = [0.3882    0.1843    0.1843];
color.grn = [0.2667    0.6588    0.3294];

Test.sigma.one = 3;
Test.sigma.two = 12;
Test.sigma.int = exp(-4:.5:-1.5); %to test final two:exp(-2:.5:-1.5)

Training.sigma = Test.sigma;
Training.sigma.int = 1;

Test.n.blocks = 3;
Test.n.sections = 3;
Test.n.trials = 6  *(numel(Test.sigma.int)*2);

Training.initial.n.blocks = 1; %Do Not Change
Training.initial.n.sections = 2;
Training.initial.n.trials = 72; %WTA from 72
Training.n.blocks = 0;%Test.n.blocks;
Training.n.sections = 1; %changed from '2' on 10/14
Training.n.trials = 48; %WTA from 48

Test.t.pres = 50;
Test.t.pause = 200;
Test.t.feedback = 1200;
Test.t.betwtrials = 1000;

Training.t.pres = 300; %how long to show first stimulus (ms)
Training.t.pause = 100; %time between response and feedback
Training.t.feedback = 800;   %time that score is on screen
Training.t.betwtrials = 1000;


fontsize = 28;
fontstyle = 0;


try
    % Choose screen with maximum id - the secondary display:
    screenid = max(Screen('Screens'));
    [scr.win scr.rect] = Screen('OpenWindow', screenid, 128);
    scr.w = scr.rect(3);
    scr.h = scr.rect(4);
    scr.cx = mean(scr.rect([1 3]));
    scr.cy = mean(scr.rect([2 4]));
    
    HideCursor
    
    Screen('TextSize', scr.win,fontsize);
    Screen('TextStyle', scr.win, fontstyle)
    
    % screen info
    
    screen_resolution = [scr.w scr.h];                 % screen resolution
    screen_distance = 50;                      % distance between observer and screen (in cm)
    screen_angle = 2*(180/pi)*(atan((screen_width/2) / screen_distance)) ; % total visual angle of screen
    screen_ppd = screen_resolution(1) / screen_angle;  % pixels per degree
    
    %set up fixation cross
    f_c_size = 18; %pixels
    white = 255;
    black =1;
    soft_black = 10;
    gray = (white+black)/2;
    f_c = ones(f_c_size);
    background_color = gray;
    f_c = background_color*f_c;
    f_c(f_c_size/2: f_c_size/2 + 1,:) = soft_black;
    f_c(:,f_c_size/2: f_c_size/2 + 1) = soft_black;
    scr.cross = Screen('MakeTexture', scr.win , f_c);
    
    %Stimulus Parameters stored in struct 'P'
    %if strcmp(P.stim_type, 'gabor')

        P.sc = 50.0;
        P.freq = .1;
        P.contrast = 100.0;
        P.gabor_spatialFreq = 70;  % spatial frequency of grating (cycles / deg)
        P.gabor_period = 1 / P.gabor_spatialFreq;
        P.gabor_wavelength    = 30; %gaussian envelope of gabor (pixels). Also called 'gabor_sigma'
        P.gabor_phase    = 0;    % gabor phase
        P.gabor_contrast = 50;   % gabor contrast (in %)
        P.gabor_mean     = 128;  % background luminance (in range [0..255])
        
    %elseif strcmp(P.stim_type, 'grate')
        
        
        %% Grating parameters
        %P.textureSize = 500;
        %P.alphaMaskSize = 500;
        P.diskSize = 150;
        % some member variables
        P.refresh = 60;
        
        % read parameters
        P.stimTime = 1500;
        P.location = [0 0]';
        P.postStimTime = 200;
        
        % new grating parameters
        %old:  P.spatialFreq = 1; % spatial frequency of grating (cycles / deg)
        %P.pxPerDeg = 68; % pixels per degree
        
        %new:
        P.spatialFreq = .526; % spatial frequency of grating (cycles / deg)
        dist = 24*2.54; %cm
        P.pxPerDeg     = scr.rect(3)*(1/screen_width)*dist*tan(pi/180); %
        %pixels/degree = (px/screen)*   (screen/cm)  *   (cm/deg)
        
        P.spatialFreq = P.spatialFreq / P.pxPerDeg; % cycles / pixel
        P.gabor_wavelength = 30;
        P.orientation = 0; % orientation of grating
        P.initialPhase = 0;
        P.phi = P.initialPhase; % initial phase
        P.period = 1 / P.spatialFreq;
        P.speed = 4.5; % cycles per second
        P.dt = .02; %seconds per frame
        P.speed = P.speed * P.period * P.dt;  % convert to px/frame
        
        % determine size
        %P.maxPeriod = max(P.textureSize) - max(P.diskSize);
        %P.alphaMaskSize = ceil(max(P.diskSize) / 2 + P.maxPeriod) + 5;
        alphaMaskSize = 5; %degrees   
            %EFFECTIVE RADIUS OF ALPHA MASK
            %for 97% filtered: 2.25 degrees (diam 4.5)
            %for 97.5% filtered: 2.30 degrees (diam(4.6)
            %for 99% filtered:  2.55 degrees  (diam(4.7)
        P.alphaMaskSize = round(alphaMaskSize*P.pxPerDeg);
        
        
        % set contrast and luminance
        P.contrast = 100;
        P.luminance = .5;
        P.color = [1 1 1];
        P.bgColor = [127 127 127]';
        
    %end
    
    if strcmp(P.stim_type, 'grate')
        %% make the alpha map
        x = -P.alphaMaskSize:P.alphaMaskSize-1;
        [X,Y] = meshgrid(x,x);
        alphaLum = repmat(permute(P.bgColor,[2 3 1]),2*P.alphaMaskSize,2*P.alphaMaskSize);
        %alphaBlend = 255 * (sqrt(X.^2 + Y.^2) > P.diskSize/2);
        [x2,y2] = meshgrid(x,x);
        arg   = -(x2.*x2 + y2.*y2)/(2*(P.gabor_wavelength^2));
        filt     = exp(arg);    %set up Bivariate Distribution
        normm = sum(filt(:));
        if normm ~= 0,
            filt  = filt/normm; %normalize it
        end;
        alphaBlend = 255*(1-(filt/max(max(filt))));
        scr.alphaMask = Screen('MakeTexture',scr.win,cat(3,alphaLum,alphaBlend));
    end
    
    %%%Setup routine   
    InitialTrainingpreR = setup_exp_order(Training.initial.n, Training.sigma);
    Training.n.blocks = Training.n.blocks - 1;
    TrainingpreR = setup_exp_order(Training.n, Training.sigma);
    Training.n.blocks = Training.n.blocks + 1;
    
    Training.R.trial_order{1} = InitialTrainingpreR.trial_order{1};
    Training.R.sigma{1} = InitialTrainingpreR.sigma{1};
    Training.R.draws{1} = InitialTrainingpreR.draws{1};
    for spec = 2:Training.n.blocks
        Training.R.trial_order{spec} = TrainingpreR.trial_order{spec-1};
        Training.R.sigma{spec} = TrainingpreR.sigma{spec-1};
        Training.R.draws{spec} = TrainingpreR.draws{spec-1};
    end
    
    Test.R = setup_exp_order(Test.n, Test.sigma);
    
    %subjects who have already run at least one session
    have_run = {'MR', 'RG', 'TS', 'ELA', 'wjma', 'RB'};
    
%only show demo for subjects who have never run a session
if sum(strcmp(have_run, initial)) == 0
    %%% DEMO  %%%
    %class_1_orts = Test.sigma.one*randn(5,1) + 90;%or can set arbitrarily: [90 97 80];
    class_1_orts = 90 + [2 -2 5 -2 10];
    class_2_orts = Test.sigma.two*randn(5,1) + 90;%or can set arbitrarily: [112 55 95];
    upper_left_corner = [scr.cx*.35 scr.cy*.4];
    gabor_size = 6*P.gabor_wavelength; %6*[P.gabor_wavelength P.gabor_wavelength];
    dest_one = [upper_left_corner   upper_left_corner+gabor_size];
    x_int = scr.cx*.05;
    y_int = scr.cy*.6;%go back to:  scr.cy*.8;
    dest_two = dest_one +  y_int*[0 1 0 1];

    Screen('DrawText', scr.win, 'Examples of Class 1 Stimuli:', upper_left_corner(1)-20, upper_left_corner(2)-60)
    Screen('DrawText', scr.win, 'Examples of Class 2 Stimuli:', upper_left_corner(1)-20, upper_left_corner(2)-60+y_int)

    %demo_contrasts = ceil(rand(5,2)*6);
    for n = 1:5
      r_gabor(P, 90, scr, Test.sigma.int(n), [], dest_one);
      r_gabor(P, class_2_orts(n), scr, 1, [], dest_two);
      dest_one(1:2:3) = dest_one(1:2:3) + x_int + gabor_size;
      dest_two(1:2:3) = dest_two(1:2:3) + x_int + gabor_size;
    end
    
    Screen('Flip', scr.win)
    WaitSecs(1)
    %KbWait
    WaitSecs(2)
    Screen('Flip', scr.win); %Screen('CloseAll', scr.win)

end
    %%% END DEMO %%%
    
    Screen('DrawText', scr.win, 'Press Any Key to Begin Initial Training', scr.cx*.3, scr.cy*.9)
    Screen('Flip', scr.win)
    WaitSecs(1)
    KbWait;
    Screen('Flip', scr.win)
    WaitSecs(.5);
    
    for k = 1:Test.n.blocks
        if k ==1
            Training.initial.n.blocks = Training.n.blocks;
            numbers = Training.initial.n;
        else
            numbers = Training.n;
        end
        
        [Training.responses{k} flag] = run_exp(numbers, Training.R, Training.t, scr, color, P, 'Training',k);
        if flag ==1,  break;  end
        [Test.responses{k} flag] = run_exp(Test.n, Test.R, Test.t, scr, color, P, 'Test',k);
        if flag ==1,  break;  end
        
        %load top scores
        load top_ten

        score = sum(sum(Test.responses{k}.tf));
        
        
        ranking = 11 - sum(score>=top_ten.scores);
        
        if ranking < 11
            top_ten.scores = [top_ten.scores(1:(ranking-1));  score;  top_ten.scores(ranking:9)];
            for m = 10:-1:ranking+1
                top_ten.initial{m} = top_ten.initial{m-1};
            end
            top_ten.initial{ranking} = initial;
        end
        
        save top_ten top_ten
        
        for j = 1:10
            xval = scr.cx*.8 - (j==10)*scr.cx*.025;
            yval = scr.cy*(.4+j/10);
            Screen('DrawText', scr.win, [num2str(j) ') ' num2str(top_ten.scores(j)) '   ' top_ten.initial{j}],xval, yval)
        end
        
        Screen('DrawText', scr.win, ['Your score for this block:  ' num2str(score) ], scr.cx*.35, scr.cy * .25)
        Screen('DrawText', scr.win, 'Top Ten:', scr.cx*.8, scr.cy*.4)
        if k~= Test.n.blocks
            Screen('DrawText', scr.win, ['Take a short break, then press'], scr.cx*.5, scr.cy * 1.6)
            Screen('DrawText', scr.win, ['any key to begin Training'], scr.cx*.6, scr.cy * 1.7)
            wait_time = 1; %waiting after top 10
        else
            Screen('DrawText', scr.win, ['You''re done with the experiment.'], scr.cx*.47, scr.cy * 1.6)
            Screen('DrawText', scr.win, ['Thank you for participating! '], scr.cx*.54, scr.cy * 1.7)
            wait_time = 0;
        end
        Screen('Flip', scr.win)
        WaitSecs(1)
        KbWait;
        Screen('Flip', scr.win)
        WaitSecs(wait_time);
    end

    Screen('CloseAll')
    if flag == 1
       initial = [initial '_rec']; 
    end
    
    save top_ten top_ten;
    savedatafcn(initial, Training, Test, P);
    
catch %if error or script is cancelled
    try
    Screen('CloseAll')
    catch; end
    file_name = [initial '_recovered'];
    savedatafcn(file_name, Training, Test, P); %save what we have
    fprintf('\n\nThe experiment can''t run, most likely because Psychtoolbox isn''t set up. To set it up, type: \n\n   cd C:\\toolbox\\Psychtoolbox \n   setupPsychtoolbox\n\n');
    fprintf('Then type ''yes'' when asked.\n')
    fprintf('(Note: this experiment is in directory N:\\malab\\Ryan_orientation_memory)\n')
    psychrethrow(psychlasterror);
end

function R = setup_exp_order(n, sigma)
R = [];
%%%Set up order of class, sigma, and orientation for entire scheme%%%
%Indexing:   R.Property{block}(trial, section)
% R.trial_order = zeros(n.trials, n.sections, n.blocks);
% R.sigma = zeros(n.trials, n.sections, n.blocks);
% R.draws = zeros(n.trials, n.sections, n.blocks);
intermediate = zeros(n.trials,1);

sample_section = repmat([1 2],1,n.trials);
sample_sigma = repmat(sigma.int,1,n.trials/numel(sigma.int)/2);

for k = 1:n.blocks
    for m = 1:n.sections
        %block_order{block}(section, trial)
        
        %permute classes to get order of classes
        R.trial_order{k}(m,:) = sample_section(randperm(n.trials));
        
        %permute sigma to get order of sigma CORRESPONDING TO order of classes
        intermediate(R.trial_order{k}(m,:)==1) = sample_sigma(randperm(n.trials/2));
        intermediate(R.trial_order{k}(m,:)==2) = sample_sigma(randperm(n.trials/2));
        R.sigma{k}(m,:) = intermediate;
    end
    %get random draws from normal distributions (~sigmal or ~sigma2) CORRESPONDING TO order of classes
    classtwos = (R.trial_order{k} == 2);
    R.draws{k} = sigma.one*randn(n.sections,n.trials);
    R.draws{k}(classtwos) = sigma.two*randn(nnz(classtwos),1);
end



function [responses flag] = run_exp(n, R, t, scr, color, P, type, blok)
%%%Run trials (Training or Test)%%%
try
    flag = 0;
    %binary matrix of correct/incorrect responses
    responses.tf = zeros(n.sections, n.trials);
    responses.c = zeros(n.sections, n.trials);
    abc = {'a','b','c'};

    for section = 1:n.sections
        for trial = 1:n.trials
            fprintf('blok %g, section %g, trial %g\n',blok,section,trial) % WTA

            Screen('DrawTexture', scr.win, scr.cross);
            Screen('Flip', scr.win);
            WaitSecs(t.betwtrials/1000)
            cval = R.trial_order{blok}(section, trial); %class
            ort = R.draws{blok}(section, trial);        %orientation
            cur_sigma = R.sigma{blok}(section, trial);  %contrast
            if strcmp(P.stim_type, 'gabor')
                r_gabor(P, ort, scr, cur_sigma, t);
            elseif strcmp(P.stim_type, 'grate')
                grate(P, ort, scr, cur_sigma, t);
            end
            Screen('Flip', scr.win)
            
            %subject input
            resp = 0;
            while resp == 0;
                [keyIsDown, secs, keyCode] = KbCheck; %changed this to KbCheck from KbCheck(1)
                
                %To quit script, press z,r,h ONLY simultaneously
                if keyCode(90) && keyCode(82) && keyCode(72) && sum(keyCode)==3
                    You cancelled the script by pressing the z, r, and h keys simultaneously.
                end
                
                if keyCode(scr.key1) || keyCode(49) %keys with '1'
                    resp = 1;
                elseif keyCode(scr.key2) || keyCode(50) %keys with '2'
                    resp = 2;
                end
            end
            %record 1 if correct, 0 if incorrect
            responses.tf(section, trial) = (resp == cval);
            responses.c(section, trial) = resp;
            
            if strcmp(type, 'Training') %to add random feedback during test: || rand > .9 %mod(sum(sum(tfresponses)) ,10)==9
                %feedback
                WaitSecs(t.pause/1000)
                if resp == cval
                    status = 'Correct!';
                    stat_col = color.red;
                    lj = .86;
                else
                    status = 'Incorrect!';
                    stat_col = color.grn;
                    lj = .83;
                end
                xpos = scr.cx*[lj  .9  .97];
                ypos = scr.cy*[.9  .99  1.08];
                Screen('DrawText', scr.win, status, xpos(1), ypos(1), stat_col);
                Screen('DrawText', scr.win, 'Class:', xpos(2), ypos(2) , color.wt);
                Screen('DrawText', scr.win, num2str(cval), xpos(3), ypos(3));
                Screen('Flip', scr.win)
                WaitSecs(t.feedback/1000);
                
            end
            %Screen('DrawText', scr.win, num2str(cur_sigma), scr.cx*.9, scr.cy*.81);

        end
        %show text at the end of the section
        %pcnt = num2str(sum(responses.tf(section, :)))/num2str(n.trials)
        %possible_texts = {'Try Harder' ,'Good Try.', 'Good job!', 'Great!', 'Awesome!', };
        %floor(pcnt*5) +1
        %encourager = possible_texts{floor(pcnt*5) +1};
        encourager = 'Good job!';
        toptxt = [encourager ' You got ' num2str(sum(responses.tf(section, :)))...
            ' of ' num2str(n.trials) ' trials correct in that section.'];
        Screen('DrawText', scr.win, toptxt,scr.cx*.02, scr.cy*.75, color.wt, color.bg);
        
        %if another section in the same block immediately follows
        if section ~= n.sections
            if strcmp('Training', type) && blok == 1
                midtxt = ['Coming up: Part 2 of Initial Training'];
                midtxtx = scr.cx*.4;
                lowtxt = ' ';
                lowtxtx = scr.cx*.59;
            elseif strcmp('Training', type)
                midtxt = ['Coming up: Test Block ' num2str(section+1)  'a'];
                midtxtx = scr.cx*.6;
                lowtxt = ['Training before Block ' num2str(blok)];
                lowtxtx = scr.cx*.59;
            else
                midtxt = ['Coming up: Test Block ' num2str(blok)  abc{section+1}];
                midtxtx = scr.cx*.57;
                lowtxt = ['(Block ' num2str(blok) ' of ' num2str(n.blocks) ')'];
                lowtxtx = scr.cx*.75;
            end
            Screen('DrawText', scr.win, midtxt, midtxtx, scr.cy*.95 , color.wt, color.bg);
            Screen('DrawText', scr.win, lowtxt, lowtxtx, scr.cy*1.05 , color.wt, color.bg);
            Screen('DrawText', scr.win, 'Press Any Key to Continue',scr.cx*.57, scr.cy*1.15, color.wt, color.bg);
            Screen('Flip', scr.win)
            WaitSecs(.5)
            KbWait;
            Screen('Flip', scr.win)
            WaitSecs(1.5)
        end
        
        
    end
    %end of block text
    if strcmp(type, 'Training')
        hitxt = ['You''ve also just finished training before block ' num2str(blok)];
        hitxtx = scr.cx*.1;
        midtxt = ['with a score of ' num2str(sum(sum(responses.tf))) '/' num2str(n.sections*n.trials) '.'];
    elseif strcmp(type, 'Test')
        hitxt = ['You''ve also just finished testing for block ' num2str(blok)];
        hitxtx = scr.cx*.2;
        midtxt = ['with a score of ' num2str(sum(sum(responses.tf))) '/' num2str(n.sections*n.trials) '.'];
    end
    Screen('DrawText', scr.win, hitxt, hitxtx, scr.cy*.85, color.wt, color.bg);
    Screen('DrawText', scr.win, midtxt,scr.cx*.65, scr.cy*.95, color.wt, color.bg);
    Screen('DrawText', scr.win, 'Press any key to continue',scr.cx*.6, scr.cy*1.25, color.wt, color.bg);
    
    if  strcmp(type, 'Training')
        lowtxt =['Coming up: Test, Block ' num2str(blok) 'a'];% ' of ' num2str(n.blocks)];
        Screen('DrawText', scr.win, lowtxt, scr.cx*.55, scr.cy*1.15, color.wt, color.bg);
    elseif blok~=n.blocks
        lowtxt =['Coming up: Training before block ' num2str(blok+1)]; %, Block ' num2str(blok+1) ' of ' num2str(n.blocks)];
        Screen('DrawText', scr.win, lowtxt, scr.cx*.45, scr.cy*1.15, color.wt, color.bg);
    end
    
    Screen('Flip', scr.win)
    WaitSecs(.5)
    KbWait;
    Screen('Flip', scr.win)
    
catch
    responses.tf(section, trial) = -1;
    responses.c(section, trial) = -1;
    psychrethrow(psychlasterror)
    save responses
    flag = 1;
end



function r_gabor(P, ort, scr, cur_sigma, t , dest)
%%%Create and and display gabor patch%%%

% set image size (cut off after center +/- 3*sigma)
imsize = round(6*[P.gabor_wavelength P.gabor_wavelength]);

% generate cosine pattern
X = ones(imsize(1),1)*[-(imsize(2)-1)/2:1:(imsize(2)-1)/2];
Y =[-(imsize(1)-1)/2:1:(imsize(1)-1)/2]' * ones(1,imsize(2));

cospattern = cos(2.*pi.*P.gabor_period.* (cos(ort*pi/180).*X ...
    + sin(ort*pi/180).*Y)  ...
    - P.gabor_phase*ones(imsize) );

%NEW
lambda = 1/20;
phase = 20;

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

if nargin > 5 %in the demo, destination specified
    Screen('DrawTexture', scr.win, gabortex, [], dest)
    Screen('Close', gabortex)
else %regular trials
    Screen('DrawTexture', scr.win, gabortex)
    Screen('Close', gabortex)
    Screen('Flip', scr.win)
    WaitSecs(t.pres/1000)
    Screen('Flip', scr.win)
end




function  grate(P, ort, scr, cur_sigma, t)
%%% Create and and display grating (moving gabor) %%%

%% Setup Screen and alpha blending
Screen('BlendFunction',scr.win,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
%% make the texture

% texture needs to be a bit larger because we're going to move it under the
% aperture to generate the motion
textureSize = ceil(P.diskSize + P.period);

% rotate co-ordinate system
phi = 2*pi*P.spatialFreq * (1:textureSize);
grat = 127.5 + cur_sigma * 126 * sin(phi + P.initialPhase); %'120' (col. 28) changed to '126' on 10/20 by RG

% color grating
color = permute(P.color,[2 3 1]);
grat = bsxfun(@times,color,grat);

% create texture
texture = Screen('MakeTexture',scr.win,grat);

%% show it
% some shortcuts
centerX = scr.cx;
centerY = scr.cy;


startTime = inf;
lastTime = 0;
phi2 = P.initialPhase;

k = 0;
tic
while (lastTime - startTime)*1000 < t.pres
    while toc < k*P.dt
    end
    k = k+1;
    Screen('FillRect', scr.win, P.bgColor, scr.rect);
    
    % move grating
    u = mod(phi2,P.period) - P.period/2;
    xInc = -u * sin(ort/180*pi);
    yInc = u * cos(ort/180*pi);
    ts = textureSize;
    destRect = [centerX centerY centerX centerY] + [-ts -ts ts ts] / 2 ...
        + [xInc yInc xInc yInc];
    phi2 = phi2 + P.speed;
    
    % draw grating
    Screen('DrawTexture',scr.win,texture,[],destRect,ort+90);

    % draw circular aperture
    alphaRect = [centerX centerY centerX centerY] + [-1 -1 1 1] * P.alphaMaskSize;
    Screen('DrawTexture',scr.win,scr.alphaMask,[],alphaRect,ort+90);
    
    % fixation
    lastTime = Screen('Flip',scr.win);

    % compute startTime
    if startTime == inf
        startTime = lastTime;
    end
end

Screen('Close', texture)
Screen('FillRect', scr.win, P.bgColor, scr.rect);
Screen('Flip', scr.win)



function savedatafcn(initial, Training, Test, P)
%%% Save data to file %%%

%Convert current date to string
Date = datevec(date);
year = num2str(Date(1));
month = num2str(Date(2));
day = num2str(Date(3));

if ((Date(2) >= 10) && (Date(3) >= 10))
    file_date = [year, month, day];
elseif ((Date(2) < 10) && (Date(3) >= 10))
    file_date = [year, '0', month, day];
elseif ((Date(2) >= 10) && (Date(3) < 10))
    file_date = [year, month, '0', day];
elseif ((Date(2) < 10) && (Date(3) < 10))
    file_date = [year, '0', month, '0', day];
end

letter_sub = [97:122]; %no observer is expected to run more than 26 blocks/day!
letter1 = 1;
dot_mat = '.mat';

file_name = [initial file_date eval('letter_sub(letter1)') dot_mat];

for (letter1 = 1:26)
    if (isempty(dir(file_name)) == 1)
        break;
    end
    file_name = [initial file_date eval('letter_sub(letter1)') dot_mat];
end

savedest = [cd '\' file_name]; %if you're logged into a nemo account

save(savedest, 'Training', 'Test', 'P')