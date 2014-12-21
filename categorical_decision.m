%Ryan George
%Theoretical Neuroscience Lab, Baylor College of Medicine
%Categorial decisions pilot program

rng('shuffle','twister')

Screen('Preference', 'SkipSyncTests', 1); % WTA.
if strcmp(computer,'MACI64') % Assuming this is running on my MacBook
    dir = '/Users/will/Google Drive/Ma lab/repos/qamar confidence';
elseif strcmp(computer,'PCWIN64') % Assuming the left Windows psychophysics machine
    dir = 'C:\Users\malab\Documents\GitHub\Confidence-Theory';
end

cd(dir)


% persistent have_done
% if numel(have_done) == 0 %if no previous session has been run today
%     %reset random seed to clock value (see help RandStream)
%     s = RandStream.create('mt19937ar','seed',sum(100*clock));
%     RandStream.setDefaultStream(s);
%     have_done = 1;
% end

%fprintf('To abort the program, press x and z buttons simultaneously \n\n')

initial = input('Please enter your initials.\n> ', 's'); % 's' returns entered text as a string

new_subject_flag = input('\nAre you new to this experiment? Please enter y or n.\n> ', 's');
while ~strcmp(new_subject_flag,'y') && ~strcmp(new_subject_flag,'n')
    new_subject_flag = input('You must enter y or n.\nAre you new to this experiment? Please enter y or n.\n> ', 's');
end

room_letter = input('\nPlease enter the room name [mbp] or [home] or [1139].\n> ', 's');

%initial = 'test';
%new = 'y'
%room_letter = 'mbp';

datetimestamp = datetimefcn; % establishes a timestamp for when the experiment was started

% map keys with 'WaitSecs(0.2); [~, keyCode] = KbWait;find(keyCode==1)' and
% then press a key.
switch room_letter
    case 'home'
        screen_width = 64;%64 %30" cinema display width in cm (.250 mm pixels)
    case 'mbp'
        screen_width = 33.2509;
    case '1139'
        screen_width = 37.632; % left LCD screen
        
        %scr.keyx=88; %x
        %scr.keyz=90; %z
        scr.keyinsert = 45; % insert
        scr.keyenter = 13; % enter
        
        %[scr.key1, scr.key2, scr.key3, scr.key4, scr.key5, scr.key6] = ...
        %deal(49, 50, 51, 48, 189, 187); % This is for keys 1,2,3,0,-,=
        
        %    [scr.key1, scr.key2, scr.key3, scr.key4, scr.key5, scr.key6,...
        %        scr.key7, scr.key8] = deal(65, 83, 68, 70, 72, 74, 75, 76)
        % This is for keys a,s,d,f,h,j,k,l
        
        [scr.key1, scr.key2, scr.key3, scr.key4, scr.key5, scr.key6,...
            scr.key7, scr.key8, scr.key9, scr.key10] ...
            = deal(112, 113, 114, 115, 116, 119, 120, 121, 122, 123);
        % This is for keys F1-5, F8-12.
end
if strcmp(room_letter,'home') || strcmp(room_letter,'mbp')
    [scr.key1, scr.key2, scr.key3, scr.key4, scr.key5, scr.key6,...
        scr.key7, scr.key8, scr.key9, scr.key10] ...
        = deal(30, 31, 32, 33, 34, 37, 38, 39, 45, 46); % This is for keys 1,2,3,4,5,8,9,0,-,=
    scr.keyinsert=53;% backspace
    scr.keyenter=42;%tilde
        %scr.keyx=27; %x
    %scr.keyz=29; %z
end

close all;
%%
demo_type='new'; % 'old' or 'new' ("movie")
    nDemoTrials = 50; % for 'new' style demo

%Paradigm Parameters stored (mainly) in the two structs 'Training' and 'Test'
P.stim_type = 'grate';  %options: 'grate', 'ellipse'
category_type = 'sym_uniform'; % 'same_mean_diff_std' or 'diff_mean_same_std' or 'sym_uniform' or 'half_gaussian. Further options for sym_uniform (ie bounds, and overlap) and half_gaussian (sig_s) are in setup_exp_order.m
attention_manipulation = true;
cue_validity = .7;
% colors in 0:1 space
% color.bg = [0.4902    0.5647    0.5137];
% color.wt = [1 1 1];
% color.bk = [0 0 0];
% color.red = [0.3882    0.1843    0.1843];
% color.grn = [0.2667    0.6588    0.3294];

% colors in 0:255 space
color.bg = [125.0010  143.9985  130.9935]; % this doesn't appear to be used?
color.wt = [255 255 255];
color.bk = [0 0 0];
color.red = [100 0  0];
color.grn = [0 100 0];

countdown = 30; % countdown between blocks

if strcmp(category_type, 'same_mean_diff_std')
    Test.category_params.sigma_1 = 3;
    Test.category_params.sigma_2 = 12;
elseif strcmp(category_type, 'diff_mean_same_std')
    Test.category_params.sigma_s = 5; % these params give a level of performance that is about on par withthe original task (above)
    Test.category_params.mu_1 = -4;
    Test.category_params.mu_2 = 4;
elseif strcmp(category_type, 'sym_uniform')
    Test.category_params.uniform_range = 15;
    Test.category_params.overlap = 0;
elseif strcmp(category_type, 'half_gaussian')
    Test.category_params.sigma_s = 5;
end


if strcmp(P.stim_type, 'ellipse')
    Test.category_params.test_sigmas = .4:.1:.9; % are these reasonable eccentricities?
else
    Test.category_params.test_sigmas= exp(-4:.5:-1.5); %4 of these 6 are in qamar 2013. WTA. to test final two:exp(-2:.5:-1.5)
end

Training.category_params = Test.category_params;
if strcmp(P.stim_type, 'ellipse')
    Training.category_params.test_sigmas = .95;
else
    Training.category_params.test_sigmas = 1;
end

Test.n.blocks = 3;% WTA from 3
Test.n.sections = 2; % WTA from 3
Test.n.trials = 18*numel(Test.category_params.test_sigmas); % 9*numel(Test.sigma.int)*2 = 108

Training.initial.n.blocks = 1; %Do Not Change
Training.initial.n.sections = 2; % WTA: 2
Training.initial.n.trials = 36;% WTA: 36
Training.confidence.n.blocks = 1;
Training.confidence.n.sections = 1;
Training.confidence.n.trials = 16; % WTA: 16
Training.n.blocks = Test.n.blocks; % was 0 before, but 0 is problematic.
Training.n.sections = 1; %changed from '2' on 10/14
Training.n.trials = 48; % WTA: 48

Demo.t.pres = 250;
Demo.t.betwtrials = 200;


Test.t.pres = 500;           %50. needs to be longer for attention experiment?
Test.t.pause = 200;         %200 isn't used
Test.t.feedback = 1200;     %1200 isn't used
Test.t.betwtrials = 1000;   %1000
Test.t.attention_cue = 1000;

Training.t.pres = 300; %300 %how long to show first stimulus (ms)
Training.t.pause = 100; %100 time between response and feedback
Training.t.feedback = 1200;  %1700 time of "correct" or "incorrect" onscreen
Training.t.betwtrials = 1000; %1000
Training.t.attention_cue = 1000;


%%8
if strfind(initial,'fast') > 0 % if 'fast' is in the initials, the exp will be super fast (for debugging)
    [Test.t.pres,Test.t.pause,Test.t.feedback,Test.t.betwtrials,Training.t.pres,Training.t.pause,Training.t.feedback,Training.t.betwtrials,countdown]...
        = deal(1);
end

if strfind(initial,'short') > 0 % if 'short' is in the initials, the exp will be short (for debugging)
    [Test.n.trials,Training.initial.n.trials,Training.confidence.n.trials,Training.n.trials]...
        = deal(numel(Test.category_params.test_sigmas)*2, 4, 4, 4);
    nDemoTrials = 5;
end

fontsize = 28;
fontstyle = 0;


try
    % Choose screen with maximum id - the secondary display:
    %primary display
    %screenid = min(Screen('Screens'));
    
    %secondary display
    screenid = max(Screen('Screens'));
    
    if strcmp(room_letter,'home')
        screenid = 0;
    end
    
    % OPEN FULL SCREEN
    [scr.win, scr.rect] = Screen('OpenWindow', screenid, 128); %scr.win is the window id (10?), and scr.rect is the coordinates in the form [ULx ULy LRx LRy]
    % OPEN IN WINDOW
    %[scr.win, scr.rect] = Screen('OpenWindow', screenid, 128, [100 100 1200 1000]);
 
    %LoadIdentityClut(scr.win) % default gamma table
    if strcmp(room_letter, '1139')
    load('calibration/1139_L_Dell_GammaTable') % gammatable calibrated on Meyer 1139 L Dell monitor, using CalibrateMonitorPhotometer (edits are saved in the calibration folder)
    Screen('LoadNormalizedGammaTable', scr.win, gammaTable*[1 1 1]);
    end
    
    scr.w = scr.rect(3); % screen width in pixels
    scr.h = scr.rect(4); % screen height in pixels
    scr.cx = mean(scr.rect([1 3])); % screen center x
    scr.cy = mean(scr.rect([2 4])); % screen center y
    
    HideCursor;
    SetMouse(0,0);
    
    Screen('TextSize', scr.win,fontsize); % Set default text size for this window, ie 28
    Screen('TextStyle', scr.win, fontstyle); % Set default text style for this window. 0 means normal, not bold/condensed/etc
    Screen('Preference', 'TextAlphaBlending', 0);
    
    % screen info
    
    screen_resolution = [scr.w scr.h];                 % screen resolution ie [1440 900]
    screen_distance = 50;                      % distance between observer and screen (in cm)
    screen_angle = 2*(180/pi)*(atan((screen_width/2) / screen_distance)) ; % total visual angle of screen in degrees
    screen_ppd = screen_resolution(1) / screen_angle;  % pixels per degree
    
    %set up fixation cross
    f_c_size = 18; %pixels
    black = 10;
    gray = 69;
    bg = 128;
    f_c = bg*ones(f_c_size);
    f_c(f_c_size/2: f_c_size/2 + 1,:) = black;
    f_c(:,f_c_size/2: f_c_size/2 + 1) = black;
    scr.cross = Screen('MakeTexture', scr.win , f_c);
    
    %set up attention arrow. this is kind of hacky, and a bit ugly.
    a_h = 87; % must be divisible by 3 and odd? this is annoying.
    a_w = ((a_h-1)/2)*3;
    arrow = bg*ones(a_h,a_w+1);
    %unfilled_arrow = arrow;
    rect_end = 2*a_w/3;
    arrow([a_h/3:2*a_h/3],[1:rect_end]) = black;
    %unfilled_arrow([a_h/3 2*a_h/3],1:rect_end) = black;
    for col = 1:(a_h-1)/2+1
        arrow(col:end+1-col,rect_end+col) = black;
    end
    gray_arrow = arrow;
    gray_arrow(gray_arrow==black) = gray;
    scr.arrowL = Screen('MakeTexture', scr.win, fliplr(arrow));
    scr.arrowR = Screen('MakeTexture', scr.win, arrow);
    scr.gray_arrowL = Screen('MakeTexture', scr.win, fliplr(gray_arrow));
    scr.gray_arrowR = Screen('MakeTexture', scr.win, gray_arrow);

    
    
    
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
    
    
    % Grating parameters
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
    dist = 24*2.54; %cm Why have this in addition to screen_width above? WTA
    P.pxPerDeg     = scr.rect(3)*(1/screen_width)*dist*tan(pi/180); %
    %pixels/degree = (px/screen)*   (screen/cm)  *   (cm/deg)
    
    
    % Ellipse parameters
    P.ellipseAreaDegSq = 1; % ellipse area in degrees squared
    P.ellipseAreaPx = P.pxPerDeg^2 * P.ellipseAreaDegSq; % ellipse area in number of pixels
    %P.ellipseArea = 40000; % in degrees squared
    P.ellipseColor = 0;

    
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
    alphaMaskSize = 5; %degrees % CHANGE ME?????
    %EFFECTIVE RADIUS OF ALPHA MASK
    %for 97% filtered: 2.25 degrees (diam 4.5)
    %for 97.5% filtered: 2.30 degrees (diam(4.6)
    %for 99% filtered:  2.55 degrees  (diam(4.7)
    P.alphaMaskSize = round(alphaMaskSize*P.pxPerDeg); % alpha mask square size in pixels
    
    
    % set contrast and luminance
    P.contrast = 100;
    P.luminance = .5;
    P.color = [1 1 1];
    P.bgColor = [127 127 127]';
    
    P.attention_stim_spacing = 3.5;% for two stimuli, distance from center, in degrees
    P.stim_dist = round(P.attention_stim_spacing * P.pxPerDeg); % distance from center in pixels
%save cdtest.mat    
    if strcmp(P.stim_type, 'grate')
        % make the alpha map
        x = -P.alphaMaskSize:P.alphaMaskSize-1;
        [X,Y] = meshgrid(x,x);
        alphaLum = repmat(permute(P.bgColor,[2 3 1]),2*P.alphaMaskSize,2*P.alphaMaskSize); % 204x204x3 matrix. 204px square, with equal RGB values (gray) at each pixel
        %alphaBlend = 255 * (sqrt(X.^2 + Y.^2) > P.diskSize/2);
        [x2,y2] = meshgrid(x,x);
        arg   = -(x2.*x2 + y2.*y2)/(2*(P.gabor_wavelength^2));
        filt     = exp(arg);    %set up Bivariate Distribution
        normm = sum(filt(:));
        if normm ~= 0,
            filt  = filt/normm; %normalize it
        end;
        alphaBlend = 255*(1-(filt/max(max(filt))));
        scr.alphaMask = Screen('MakeTexture',scr.win,cat(3,alphaLum,alphaBlend)); %not totally following here, but this is 204x204x4
    end
    
    %%%Setup routine. this is some complicated stuff to deal with the
    %%%two-part training thing
    
    InitialTrainingpreR = setup_exp_order(Training.initial.n, Training.category_params, category_type);
    
    Training.n.blocks = Training.n.blocks - 1;
    TrainingpreR = setup_exp_order(Training.n, Training.category_params, category_type);
    
    Training.n.blocks = Training.n.blocks + 1; % undo previous line
    
    Training.R.trial_order{1} = InitialTrainingpreR.trial_order{1};
    Training.R.sigma{1} = InitialTrainingpreR.sigma{1};
    Training.R.draws{1} = InitialTrainingpreR.draws{1};
    Training.R.phase{1} = InitialTrainingpreR.phase{1};
    for spec = 2:Training.n.blocks
        Training.R.trial_order{spec} = TrainingpreR.trial_order{spec-1};
        Training.R.sigma{spec} = TrainingpreR.sigma{spec-1};
        Training.R.draws{spec} = TrainingpreR.draws{spec-1};
        Training.R.phase{spec} = TrainingpreR.phase{spec-1};
    end
    
    Test.R = setup_exp_order(Test.n, Test.category_params, category_type);
    
    if attention_manipulation
        Test.R2 = setup_exp_order(Test.n, Test.category_params, category_type, 1, cue_validity); % second set of stimuli. This also contains the probe/cue info.
    end
    
    Training.confidence.R = setup_exp_order(Training.confidence.n, Test.category_params, category_type);
    

    %% DEMO for new subjects
    if strcmp(new_subject_flag,'y')
        [nx,ny]=DrawFormattedText(scr.win, 'Example Stimulus\n\n', 'center', scr.cy-60, color.wt);
        
        stim.ort = 0;
        stim.cur_sigma = Training.category_params.test_sigmas;

        if strcmp(P.stim_type, 'grate')
            example_w = 6*P.gabor_wavelength;
            %r_gabor(P,0,scr,1,[],[scr.cx-example_w/2 ny scr.cx+example_w/2 ny+example_w])
            r_gabor(P,scr,[],stim,[scr.cx-example_w/2 ny scr.cx+example_w/2 ny+example_w])
        elseif strcmp(P.stim_type, 'ellipse')
            im = drawEllipse(P.ellipseAreaPx, .95, 0, P.ellipseColor, mean(P.bgColor));
            max_ellipse_d = size(im,1); % in this case, this is the height of the longest (tallest) ellipse
            %ellipse(P, 0, scr, .95, 0)%, [scr.cx-size(im,2)/2 ny scr.cx+size(im,2)/2 ny+size(im,1)])
            ellipse(P, scr, [], stim)
        end
        Screen('Flip', scr.win)%,[],1);
        WaitSecs(1);
        %Screen('Flip', scr.win); % can't the above flip be 0 and this line be removed?
        KbWait;
        
        if strcmp(demo_type, 'old')
%             nExamples = 6;
%             
%             if strcmp(category_type, 'same_mean_diff_std')
%                 %class_1_orts = Test.category_params.sigma_1*randn(examples,1) + 90; % generate
%                 %orientations. runs risk of getting nonrepresentative sample.
%                 class_1_orts = [83.6018   90.6598   92.8989   91.0500   92.7054 88.3012]-90; % might be best to do arbitrarily.
%                 %class_2_orts = Test.category_params.sigma_2*randn(examples,1) + 90; % generate
%                 class_2_orts = [115.4543  104.9830   76.5218   80.0390   88.5387 109.2334]-90; % arbitrary
%             elseif strcmp(category_type, 'sym_uniform')
%                 class_1_orts = -15 * rand(1,6);
%                 class_2_orts = 15 * rand(1,6);
%             end
%             
%             upper_left_corner = [scr.cx*.05 scr.cy*.4]; %[scr.cx*.35 scr.cy*.4]
%             
%             x_int = scr.cx*.05; % space between each example
%             y_int = scr.cy*.6;%go back to:  scr.cy*.8;
% 
%             
%             for category=1:2;
%                 Screen('DrawText', scr.win, 'Examples of Category 1 stimuli:', upper_left_corner(1)-20, upper_left_corner(2)-60);
%                 if category==2;
%                     Screen('DrawText', scr.win, 'Examples of Category 2 stimuli:', upper_left_corner(1)-20, upper_left_corner(2)-60+y_int);
%                 end
%                 
%                 if strcmp(P.stim_type, 'grate')
%                     
%                     dest_one = [upper_left_corner   upper_left_corner+example_w]; % coordinates spanned by leftmost class 1 gabor
%                     dest_two = dest_one +  y_int*[0 1 0 1]; % coordinates spanned by leftmost class 2 gabor
%                     
%                     %demo_contrasts = rand(examples,2)/2; % random and relatively high contrast
%                     %demo_contrasts = [datasample(Test.category_params.test_sigmas,examples); %datasample(Test.category_params.test_sigmas,5)]' % real, from experiment. not true for training
%                     demo_contrasts = ones(nExamples,2); % 100% contrast. best for training
%                     
%                     for n = 1:nExamples
%                         % DEPRECATED r_gabor
%                         r_gabor(P, class_1_orts(n), scr, demo_contrasts(n,1), [], dest_one); % draw gabor on screen
%                         dest_one([1 3]) = dest_one([1 3]) + x_int + example_w; % next gabor's location
%                         if category==2;
%                             r_gabor(P, class_2_orts(n), scr, demo_contrasts(n,2), [], dest_two);
%                             dest_two([1 3]) = dest_two([1 3]) + x_int + example_w;
%                         end
%                     end
%                     
%                 elseif strcmp(P.stim_type, 'ellipse')
%                     demo_contrasts = .95*ones(nExamples, 2);
%                     for n = 1:nExamples
%                         if category==1
%                             im = drawEllipse(P.ellipseAreaPx, demo_contrasts(n,1), class_1_orts(n), P.ellipseColor, mean(P.bgColor));
%                             ew = size(im,2);
%                             eh = size(im,1);
%                             dest_one = [upper_left_corner(1)+(n-1)*(max_ellipse_d + x_int)+(max_ellipse_d-ew)/2, upper_left_corner(2)+(max_ellipse_d-eh)/2, upper_left_corner(1)+(n-1)*(max_ellipse_d + x_int)+(max_ellipse_d+ew)/2, upper_left_corner(2)+(max_ellipse_d+eh)/2];
%                             ellipse(P, class_1_orts(n), scr, demo_contrasts(n,1), [], dest_one);
%                         elseif category==2
%                             im = drawEllipse(P.ellipseAreaPx, demo_contrasts(n,1), class_2_orts(n), P.ellipseColor, mean(P.bgColor));
%                             ew = size(im,2);
%                             eh = size(im,1);
%                             dest_one = [upper_left_corner(1)+(n-1)*(max_ellipse_d + x_int)+(max_ellipse_d-ew)/2, upper_left_corner(2)+(max_ellipse_d-eh)/2, upper_left_corner(1)+(n-1)*(max_ellipse_d + x_int)+(max_ellipse_d+ew)/2, upper_left_corner(2)+(max_ellipse_d+eh)/2];
%                             dest_one = dest_one + y_int*[0 1 0 1];
%                             ellipse(P, class_2_orts(n), scr, demo_contrasts(n,2), [], dest_one);
%                             
%                         end
%                     end
%                 end
%                 
%                 
%                 Screen('Flip', scr.win,[],1);
%                 WaitSecs(1);
%                 %Screen('Flip', scr.win,[],1);
%                 KbWait;
%             end
%             Screen('Flip', scr.win);
%             
        elseif strcmp(demo_type,'new')
            for category = 1 : 2
                DrawFormattedText(scr.win, sprintf('Examples of Category %i stimuli:', category), 'center', scr.cy-60, color.wt);
                Screen('Flip', scr.win);
                WaitSecs(1);
                KbWait;
                                
                for i = 1:nDemoTrials
                
                    Screen('DrawTexture', scr.win, scr.cross);
                    Screen('Flip', scr.win);
                    WaitSecs(Demo.t.betwtrials/1000);
                    
                    stim.ort = stimulus_orientations(Test.category_params, category_type, category, 1, 1);

                    if strcmp(P.stim_type, 'gabor')
                        r_gabor(P, scr, Demo.t, stim); % haven't yet added phase info to this function
                    elseif strcmp(P.stim_type, 'grate')
                        stim.phase = 360*rand;
                        grate(P, scr, Demo.t, stim);
                    elseif strcmp(P.stim_type, 'ellipse')
                        ellipse(P, scr, Demo.t, stim);
                    end            
                end
                
                KbWait;
            end
            
        end
    end
    %END DEMO
    
    
    %% START TRIALS
    [~,ny]=DrawFormattedText(scr.win,'Coming up: Category Training','center','center',color.wt)
    
    flip_pak_flip(scr,ny,color,'begin');
    
    for k = 1:Test.n.blocks
        if k == 1
            Training.initial.n.blocks = Training.n.blocks;
            numbers = Training.initial.n;
        else
            numbers = Training.n;
        end
        
        [Training.responses{k}, flag] = run_exp(numbers, Training.R, Training.t, scr, color, P, 'Training',k, new_subject_flag);
        if flag ==1,  break;  end
        
        if k == 1 && strcmp(new_subject_flag,'y') % if we are on block 1, and subject is new
            [~,ny]=DrawFormattedText(scr.win,['Let''s get some quick practice with confidence ratings.\n\n'...
                'Coming up: Confidence Training'],'center',ny,color.wt);
            flip_pak_flip(scr,ny,color,'begin')
            
            [Training.confidence.responses, flag] = run_exp(Training.confidence.n,Training.confidence.R,Test.t,scr,color,P,'Confidence Training',k, new_subject_flag);
            if flag==1,break;end
            
        end
        
        if attention_manipulation
            [Test.responses{k}, flag, blockscore] = run_exp(Test.n, Test.R, Test.t, scr, color, P, 'Test',k, new_subject_flag, Test.R2);
        else
            [Test.responses{k}, flag, blockscore] = run_exp(Test.n, Test.R, Test.t, scr, color, P, 'Test',k, new_subject_flag);
        end
        if flag ==1,  break;  end
        
       
        %load top scores
        load top_ten
        
        ranking = 11 - sum(blockscore>=top_ten.scores); % calculate current ranking
                
        if ranking < 11
            top_ten.scores = [top_ten.scores(1:(ranking-1));  blockscore;  top_ten.scores(ranking:9)];
            for m = 10:-1:ranking+1
                top_ten.initial{m} = top_ten.initial{m-1};
            end
            top_ten.initial{ranking} = initial;
            hitxt='\n\nCongratulations! You made the top ten!\n\n';
        else
            hitxt='\n\n\n\n';
        end
        
        if ~any(strfind(initial,'test'))
        save top_ten top_ten;
        end
        
        save(strrep([dir '/data/backup/' initial '_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'P') % block by block backup. strrep makes the file separator system-dependent.
        
        [nx,ny] = DrawFormattedText(scr.win,[hitxt 'Your score for Testing Block ' num2str(k) ': ' num2str(blockscore,'%.1f') '%\n\n'...
            'Top Ten:\n\n'],'center',0,color.wt);
        for j = 1:10
            [nx,ny] = DrawFormattedText(scr.win,[num2str(j) ') ' num2str(top_ten.scores(j),'%.1f') '%    ' top_ten.initial{j} '\n'],scr.cx*.8 - (j==10)*20,ny,color.wt);
        end
        
        if k ~= Test.n.blocks % if didn't just finish final testing block
            [nx,ny] = DrawFormattedText(scr.win,'\nPlease take a short break.\n\n\n','center',ny,color.wt);
            [nx,ny] = DrawFormattedText(scr.win,'You may start the next Training Block in ',scr.cx-570,ny,color.wt);
            countx=nx; county=ny;
            [nx,ny] = DrawFormattedText(scr.win,'   seconds,\n\n',countx,county,color.wt);
            [nx,ny] = DrawFormattedText(scr.win,['but you may take a longer break\n\n'...
                'and leave the room or walk around.\n\n\n'...
                'Coming up: Category Training before\n\n'...
                'Testing Block ' num2str(k+1)],'center',ny,color.wt,50);
            
            for i=1:countdown+1;
                Screen('FillRect',scr.win,128,[countx county countx+1.5*fontsize county+1.1*fontsize]) %timer background
                DrawFormattedText(scr.win,[num2str(countdown+1-i) '  '],countx,county,color.wt);
                Screen('Flip',scr.win,[],1); % flip to screen without clearing
                WaitSecs(1);
            end
            
            flip_pak_flip(scr,ny,color,'begin','initial_wait',0);
        % end top ten scores
           
        else
            [nx,ny] = DrawFormattedText(scr.win,'\n\n\n\nYou''re done with the experiment.\n\nThank you for participating!','center',ny,color.wt);
            Screen('Flip',scr.win)
            WaitSecs(1);
            KbWait([], 0, GetSecs+180); % automatically quit after 3 minutes
        end

        WaitSecs(1);
        
    end
    
    if flag == 1 % when run_exp errors
        initial = [initial '_flaggedinrunexp'];
    end
    
    save top_ten top_ten;
    
    save(strrep([dir '/data/' initial '_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'P') % save complete session
    recycle('on'); % tell delete to just move to recycle bin rather than delete entirely.
    delete([dir '/data/backup/' initial '_' datetimestamp '.mat']) % delete the block by block backup
    
    Screen('CloseAll');
    
catch %if error or script is cancelled
    try
        Screen('CloseAll');
    catch; end
    
    %file_name = ['backup/' initial '_recovered'];
    %savedatafcn(file_name, datetimestamp, Training, Test, P); %save what we have
    save(strrep([dir '/data/backup/' initial '_recovered_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'P')
    
    psychrethrow(psychlasterror);
end