function categorical_decision(category_type, subject_name, new_subject, room_letter, attention_manipulation, exp_number, nExperiments)
% Ryan George
% Theoretical Neuroscience Lab, Baylor College of Medicine
% Will Adler
% Ma Lab, New York University


if ~exist('exp_number','var') || ~exist('nExperiments','var')
    exp_number = 1;
    nExperiments = 1;
end

try
    rng('shuffle','twister')
catch
    s = RandStream.create('mt19937ar','seed',sum(100*clock));
    RandStream.setDefaultStream(s);
end

Screen('Preference', 'SkipSyncTests', 1); % WTA.
Screen('Preference', 'VisualDebuglevel', 3);

% switch user
%     case 'rachel'
        dir = pwd;
        datadir = [dir '/data'];
        addpath(genpath(dir))
%     otherwise
%         if strcmp(computer,'MACI64') % Assuming this is running on my MacBook
%             dir = '/Users/will/Google Drive/Ma lab/repos/qamar confidence';
%             datadir = '/Users/will/Google Drive/Will - Confidence/Data';
%             
%         elseif strcmp(computer,'PCWIN64') % Assuming the left Windows psychophysics machine
%             dir = 'C:\Users\malab\Documents\GitHub\Confidence-Theory';
%             datadir = 'C:\Users\malab\Google Drive/Will - Confidence/Data';
%             
%         end
% end

cd(dir)

%subject_name = input('Please enter your initials.\n> ', 's'); % 's' returns entered text as a string

%new_subject_flag = input('\nAre you new to this experiment? Please enter y or n.\n> ', 's');
%while ~strcmp(new_subject_flag,'y') && ~strcmp(new_subject_flag,'n')
%    new_subject_flag = input('You must enter y or n.\nAre you new to this experiment? Please enter y or n.\n> ', 's');
%end

%room_letter = input('\nPlease enter the room name [mbp] or [home] or [1139].\n> ', 's');

%subject_name = 'test';
%new = 'y'
%room_letter = 'mbp';

datetimestamp = datetimefcn; % establishes a timestamp for when the experiment was started

% map keys with 'WaitSecs(0.2); [~, keyCode] = KbWait;find(keyCode==1)' and
% then press a key.
scr.fontsize = 28; % unless the hires rig below
scr.fontstyle = 0; % unless the hires rig below

switch room_letter
    case 'home'
        screen_width = 64;%64 %30" cinema display width in cm (.250 mm pixels)
    case 'mbp'
        screen_width = 33.2509;
    case '1139'
        screen_width = 19.7042; %iPad screen width in cm
        scr.keyinsert = 45; % insert
        scr.keyenter = 13; % enter
        [scr.key1, scr.key2, scr.key3, scr.key4, scr.key5, scr.key6,...
            scr.key7, scr.key8, scr.key9, scr.key10] ...
            = deal(112, 113, 114, 115, 116, 119, 120, 121, 122, 123);
        % This is for keys F1-5, F8-12.
        scr.fontsize = 42;
        scr.fontstyle = 1;
    case 'Carrasco_L1'
        screen_width = 40;
        screen_distance = 56;
        
end

if strcmp(room_letter,'home') || strcmp(room_letter,'mbp') || strcmp(room_letter,'Carrasco_L1')
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
nDemoTrials = 72; % for 'new' style demo
elapsed_mins = 0;

%Paradigm Parameters stored (mainly) in the two structs 'Training' and 'Test'
P.stim_type = 'grate';  %options: 'grate', 'ellipse'
%category_type = 'same_mean_diff_std'; % 'same_mean_diff_std' or 'diff_mean_same_std' or 'sym_uniform' or 'half_gaussian. Further options for sym_uniform (ie bounds, and overlap) and half_gaussian (sig_s) are in setup_exp_order.m
% attention_manipulation = true;
cue_validity = .7;

% colors in 0:255 space
white = 255;
color.wt = [white white white];
lightgray = 180;
bg = 127.5;
color.bg = bg;
darkgray = 10;
color.bk = [0 0 0];
color.red = [100 0  0];
color.grn = [0 100 0];

switch category_type
    case 'same_mean_diff_std'
        Test.category_params.sigma_1 = 3;
        Test.category_params.sigma_2 = 12;
    case 'diff_mean_same_std'
        Test.category_params.sigma_s = 5; % these params give a level of performance that is about on par withthe original task (above)
        Test.category_params.mu_1 = -4;
        Test.category_params.mu_2 = 4;
    case 'sym_uniform'
        if attention_manipulation
            Test.category_params.uniform_range = 15; % 5
        else
            Test.category_params.uniform_range = 15;
        end
        Test.category_params.overlap = 0;
    case 'half_gaussian'
        Test.category_params.sigma_s = 5;
end

if nExperiments == 1
    task_letter = '';
    task_str = '';
    final_task = true;
elseif nExperiments > 1
    if exp_number ~= nExperiments
        final_task = false;
    else
        final_task = true;
    end
    switch category_type
        case 'same_mean_diff_std'
            task_letter = 'B';
            other_task_letter = 'A';
        case 'diff_mean_same_std'
            task_letter = 'A';
            other_task_letter = 'B';
    end
    task_str = ['Task ' task_letter ' '];
end

Test.category_params.category_type = category_type;

if strcmp(P.stim_type, 'ellipse')
    Test.category_params.test_sigmas = .4:.1:.9; % are these reasonable eccentricities?
else
    if attention_manipulation
        Test.category_params.test_sigmas = exp(-3.5);
    else
        Test.category_params.test_sigmas= exp(linspace(-5.5,-2,6)); % on previous rig: exp(-4:.5:-1.5)
    end
end

Training.category_params = Test.category_params;
if strcmp(P.stim_type, 'ellipse')
    Training.category_params.test_sigmas = .95;
else
    Training.category_params.test_sigmas = 1;
end

if attention_manipulation
    Test.n.blocks = 4;
    Test.n.sections = 4;
    Test.n.trials = 40; % 9*numel(Test.sigma.int)*2 = 108
else
    Test.n.blocks = 3;% WTA from 3
    Test.n.sections = 3; % WTA from 3
    Test.n.trials = 8*numel(Test.category_params.test_sigmas); % 9*numel(Test.sigma.int)*2 = 108
end

Training.initial.n.blocks = 1; %Do Not Change
Training.initial.n.sections = 2; % WTA: 2
Training.initial.n.trials = 36;% WTA: 36
Training.confidence.n.blocks = 1;
Training.confidence.n.sections = 1;
Training.confidence.n.trials = 24; % WTA: 16
Training.n.blocks = Test.n.blocks; % was 0 before, but 0 is problematic.
Training.n.sections = 1; %changed from '2' on 10/14
Training.n.trials = 48; % WTA: 48

if attention_manipulation
    Training.attention.n.blocks = 1;
    Training.attention.n.sections = 1;
    Training.attention.n.trials = 36;
    Training.attention.category_params = Test.category_params;
    Training.attention.category_params.category_type = 'sym_uniform';
end

Demo.t.pres = 250;
Demo.t.betwtrials = 200;

Test.t.pres = 50;           %50
Test.t.pause = 200;         %200 isn't used
Test.t.feedback = 1200;     %1200 isn't used
Test.t.betwtrials = 1000;   %1000
Test.t.cue_dur = 150;
Test.t.cue_target_isi = 150;

Training.t.pres = 300; %300 %how long to show first stimulus (ms)
Training.t.pause = 100; %100 time between response and feedback
Training.t.feedback = 1100;  %1700 time of "correct" or "incorrect" onscreen
Training.t.betwtrials = 1000; %1000
Training.t.cue_dur = 150;
Training.t.cue_target_isi = 150;

scr.countdown_time = 30;

if strfind(subject_name,'fast') > 0 % if 'fast' is in the initials, the exp will be super fast (for debugging)
    [Test.t.pres,Test.t.pause,Test.t.feedback,Test.t.betwtrials,Training.t.pres,Training.t.pause,...
        Training.t.feedback,Training.t.betwtrials,scr.countdown_time, Demo.t.pres, Demo.t.betwtrials,...
        Training.t.cue_dur, Training.t.cue_target_isi, Test.t.cur_dur, Test.t.cue_target_isi]...
        = deal(1);
end

if strfind(subject_name,'short') > 0 % if 'short' is in the initials, the exp will be short (for debugging)
    [Test.n.trials,Training.initial.n.trials,Training.confidence.n.trials,Training.n.trials, Training.attention.n.trials]...
        = deal(6);
    nDemoTrials = 5;
    scr.countdown_time = 2;
end

if strfind(subject_name,'notrain') > 0 % if 'notrain' is in the initials, the exp will not include training (for debugging)
    notrain = 1;
else
    notrain = 0;
end


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
    [scr.win, scr.rect] = Screen('OpenWindow', screenid, color.bg); %scr.win is the window id (10?), and scr.rect is the coordinates in the form [ULx ULy LRx LRy]
    % OPEN IN WINDOW
    %[scr.win, scr.rect] = Screen('OpenWindow', screenid, color.bg, [100 100 1200 1000]);
    
    %LoadIdentityClut(scr.win) % default gamma table
    switch room_letter
        case '1139'
%             gt=load('calibration/iPadGammaTable'); % gammatable calibrated on Meyer 1139 L Dell monitor, using CalibrateMonitorPhotometer (edits are saved in the calibration folder)
%             Screen('LoadNormalizedGammaTable', scr.win, gt.gammaTable*[1 1 1]);
        case 'Carrasco_L1'
            calib = load('../../Displays/0001_james_TrinitonG520_1280x960_57cm_Input1_140129.mat');
            Screen('LoadNormalizedGammaTable', scr.win, repmat(calib.calib.table,1,3));
            % check gamma table
            gammatable = Screen('ReadNormalizedGammaTable', scr.win);
            if nnz(abs(gammatable-repmat(calib.calib.table,1,3))>0.0001)
                error('Gamma table not loaded correctly! Perhaps set screen res and retry.')
            end
    end
    
    scr.w = scr.rect(3); % screen width in pixels
    scr.h = scr.rect(4); % screen height in pixels
    scr.cx = mean(scr.rect([1 3])); % screen center x
    scr.cy = mean(scr.rect([2 4])); % screen center y
    
    if ~strfind(subject_name,'test')
        HideCursor;
        SetMouse(0,0);
    end
    
    Screen('TextSize', scr.win, scr.fontsize); % Set default text size for this window, ie 28
    Screen('TextStyle', scr.win, scr.fontstyle); % Set default text style for this window. 0 means normal, not bold/condensed/etc
    Screen('Preference', 'TextAlphaBlending', 0);
    
    % screen info
    screen_resolution = [scr.w scr.h];                 % screen resolution ie [1440 900]
    if ~exist('screen_distance','var')
        screen_distance = 50;                      % distance between observer and screen (in cm) NOTE THAT FOR EXP v3 THIS WAS SET TO 50, BUT TRUE DIST WAS ~32
    end
    screen_angle = 2*(180/pi)*(atan((screen_width/2) / screen_distance)) ; % total visual angle of screen in degrees
    P.pxPerDeg = screen_resolution(1) / screen_angle;  % pixels per degree
    
    %set up fixation cross
    f_c_size = 30; % length and width. must be even.
    fw = 1; % line thickness = 2+2*fw pixels
    f_c = bg*ones(f_c_size);
    f_c(f_c_size/2 - fw: f_c_size/2 + 1 + fw,:) = darkgray;
    f_c(:,f_c_size/2 - fw: f_c_size/2 + 1 + fw) = darkgray;
    scr.cross = Screen('MakeTexture', scr.win , f_c);

    if attention_manipulation
        cross_whiteL = f_c;
        cross_whiteL(f_c_size/2-fw:f_c_size/2 + 1 + fw, 1:f_c_size/2-1-fw) = white;
        cross_grayL = f_c;
        cross_grayL(f_c_size/2-fw:f_c_size/2 + 1 + fw, 1:f_c_size/2-1-fw) = lightgray;
        
        scr.cueL = Screen('MakeTexture', scr.win, cross_whiteL);
        scr.cueR = Screen('MakeTexture', scr.win, fliplr(cross_whiteL));
        scr.resp_cueL = Screen('MakeTexture', scr.win, cross_grayL);
        scr.resp_cueR = Screen('MakeTexture', scr.win, fliplr(cross_grayL));
    end
    
    % set up grating parameters
    P.grateSigma = .8; % Gabor Gaussian envelope standard deviation (degrees)
    P.grateSigma = P.grateSigma * P.pxPerDeg; %...converted to pixels
    P.grateAspectRatio = 1;
    P.grateSpatialFreq = .8; % cycles/degree
    P.grateSpatialFreq = P.grateSpatialFreq / P.pxPerDeg; % cycles / pixel
    P.grateSpeed = 10; % cycles per second
    P.grateDt = .01; %seconds per frame
    P.grateAlphaMaskSize = round(10*P.grateSigma);
    
    % Ellipse parameters
    P.ellipseAreaDegSq = 1; % ellipse area in degrees squared
    P.ellipseAreaPx = P.pxPerDeg^2 * P.ellipseAreaDegSq; % ellipse area in number of pixels
    P.ellipseColor = 0;
    
    P.attention_stim_spacing = 3.5;% for two stimuli, distance from center, in degrees
    P.stim_dist = round(P.attention_stim_spacing * P.pxPerDeg); % distance from center in pixels
    
    %%%Setup routine. this is some complicated stuff to deal with the
    %%%two-part training thing
    
    InitialTrainingpreR = setup_exp_order(Training.initial.n, Training.category_params);
    
    Training.n.blocks = Training.n.blocks - 1;
    TrainingpreR = setup_exp_order(Training.n, Training.category_params);
    
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
    
    Test.R = setup_exp_order(Test.n, Test.category_params);
    
    if attention_manipulation
        Test.R2 = setup_exp_order(Test.n, Test.category_params, cue_validity); % second set of stimuli. This also contains the probe/cue info.
    end
    
    Training.confidence.R = setup_exp_order(Training.confidence.n, Test.category_params);
    
    if attention_manipulation
        Training.attention.R = setup_exp_order(Training.attention.n, Training.attention.category_params);
        Training.attention.R2 = setup_exp_order(Training.attention.n, Training.attention.category_params, cue_validity);
    end
    
    start_t = tic;
    %% DEMO
    
    stim.ort = 0;
    stim.cur_sigma = Training.category_params.test_sigmas;
    
    if strcmp(P.stim_type, 'grate')
        gabortex = CreateProceduralGabor(scr.win, P.grateAlphaMaskSize, P.grateAlphaMaskSize, [], [0.5 0.5 0.5 0.0],1,0.5);
        Screen('DrawTexture', scr.win, gabortex, [], [], 90-stim.ort, [], [], [], [], kPsychDontDoRotation, [0, P.grateSpatialFreq, P.grateSigma, stim.cur_sigma, P.grateAspectRatio, 0, 0, 0]);
        [nx,ny]=DrawFormattedText(scr.win, 'Example stimulus\n\n', 'center', scr.cy-P.grateAlphaMaskSize/2, color.wt);
        ny = ny+P.grateAlphaMaskSize/2;
    elseif strcmp(P.stim_type, 'ellipse')
        im = drawEllipse(P.ellipseAreaPx, .95, 0, P.ellipseColor, mean(P.bgColor));
        max_ellipse_d = size(im,1); % in this case, this is the height of the longest (tallest) ellipse
        ellipse(P, scr, [], stim)
        ny = ny+max_ellipse_d;
    end
    
 
    flip_key_flip(scr, 'continue', ny, color, new_subject)
        
    if nExperiments > 1
        hitxt=sprintf('Important: You are now doing Task %s!\n\n',task_letter);
        
        if ~new_subject
            if strcmp(task_letter, 'A')
                midtxt = 'In Task A, stimuli from Category 1 tend to\n\nbe left-tilted, and stimuli from Category 2\n\ntend to be right-tilted.\n\nSee Task sheet for more info.';
            elseif strcmp(task_letter, 'B')
                midtxt = 'In Task B, a flat stimulus is more likely\n\nto be from Category 1, and a strongly tilted\n\nstimulus is more likely\n\nto befrom Category 2.\n\nSee Task sheet for more info.';
            else
                midtxt = '';
            end
        else
            midtxt = '';
        end
        [nx,ny]=DrawFormattedText(scr.win, [hitxt midtxt], 'center', 'center', color.wt);
        flip_key_flip(scr,'continue',ny,color,new_subject);
    end
    
    %% RUN TRIALS

    for k = 1:Test.n.blocks
        
        % Training
        if ~notrain
            if k == 1
                if attention_manipulation                
                    [Training.attention.responses, flag] = run_exp(Training.attention.n,Training.attention.R,Test.t,scr,color,P,'Attention Training',k, new_subject, task_str, final_task, subject_name, Training.attention.R2);
                    if flag==1,break;end
                end
                
                category_demo
                
                Training.initial.n.blocks = Training.n.blocks;
                [Training.responses{k}, flag] = run_exp(Training.initial.n, Training.R, Training.t, scr, color, P, 'Training',k, new_subject, task_str, final_task, subject_name);
                if flag ==1,  break;  end

%                 [Training.confidence.responses, flag] = run_exp(Training.confidence.n,Training.confidence.R,Test.t,scr,color,P,'Confidence Training',k, new_subject, task_str, final_task, subject_name);
%                 if flag ==1,  break;  end
                
            else
                [Training.responses{k}, flag] = run_exp(Training.n, Training.R, Training.t, scr, color, P, 'Training',k, new_subject, task_str, final_task, subject_name);
                if flag ==1,  break;  end

            end
        end
        
        % Testing
        if attention_manipulation
            [Test.responses{k}, flag] = run_exp(Test.n, Test.R, Test.t, scr, color, P, 'Test',k, new_subject, task_str, final_task, subject_name, Test.R2);
        else
            [Test.responses{k}, flag] = run_exp(Test.n, Test.R, Test.t, scr, color, P, 'Test',k, new_subject, task_str, final_task, subject_name);
        end
        if flag ==1,  break;  end
        
        elapsed_mins = toc(start_t)/60;
        save(strrep([datadir '/backup/' subject_name '_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'P','elapsed_mins') % block by block backup. strrep makes the file separator system-dependent.
            
        if flag == 1 % when run_exp errors
            subject_name = [subject_name '_flaggedinrunexp'];
        end
    end
    
    save(strrep([datadir '/' subject_name '_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'P', 'category_type', 'elapsed_mins') % save complete session
    recycle('on'); % tell delete to just move to recycle bin rather than delete entirely.
    delete([datadir '/backup/' subject_name '_' datetimestamp '.mat']) % delete the block by block backup
    
    Screen('CloseAll');
    
catch %if error or script is cancelled
    Screen('CloseAll');
    
    save(strrep([datadir '/backup/' subject_name '_recovered_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'P','category_type', 'elapsed_mins')
    
    psychrethrow(psychlasterror);
end

end