function categorical_decision(category_type, subject_name, new_subject, room_letter, attention_manipulation, eye_tracking, exp_number, nExperiments)
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

Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference', 'VisualDebuglevel', 3);

dir = pwd;
datadir = [dir '/data'];
addpath(genpath(dir))

% eye tracking params
eye_data_dir = 'eyedata';
eye_file = sprintf('%s%s', subject_name([1:2 end-1:end]), datestr(now, 'mmdd'));
P.eye_rad = 1; % allowable radius of eye motion, in degrees visual angle
P.eye_slack = 0.05; % (s) cushion between the fixation check and the next stim presentation
P.eye_tracking = eye_tracking; % are we eye tracking?

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
        scr.displayHz = 60;
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
        scr.displayHz = 60;
        
        f_c_size = 37; % length and width. must be even.
        fw = 1; % line thickness = 2+2*fw pixels

    case 'Carrasco_L1'
        screen_width = 40;
        screen_distance = 56;
        scr.displayHz = 100;
        scr.res = [1280 960]; % should be 1280 x 960 screen res
        
        f_c_size = 28; % length and width. must be even.
        fw = 0; % line thickness = 2+2*fw pixels
end

if strcmp(room_letter,'home') || strcmp(room_letter,'mbp') || strcmp(room_letter,'Carrasco_L1')
    [scr.key1, scr.key2, scr.key3, scr.key4, scr.key5, scr.key6,...
        scr.key7, scr.key8, scr.key9, scr.key10] ...
        = deal(30, 31, 32, 33, 34, 37, 38, 39, 45, 46); % This is for keys 1,2,3,4,5,8,9,0,-,=
    scr.keyenter=88;
    scr.keyinsert=76;% delete
%     scr.keyenter=42;%tilde
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
lightgray = 180; % 157
bg = 127.5;
color.bg = bg;
darkgray = 10;
color.bk = [0 0 0];
color.red = [100 0  0];
color.grn = [0 100 0];

scr.countdown_time = 30;

% CATEGORY PARAMS
tmp.Test.category_params.category_type = category_type;
tmp.Training.category_params.category_type = category_type;
tmp.ConfidenceTraining.category_params.category_type = category_type;
tmp.AttentionTraining.category_params.category_type = 'sym_uniform';
tmp.AttentionTrainingConf.category_params.category_type = 'sym_uniform';

fields = fieldnames(tmp);
for f = 1:length(fields)
    switch tmp.(fields{f}).category_params.category_type
        case 'same_mean_diff_std'
            tmp.(fields{f}).category_params.sigma_1 = 3;
            tmp.(fields{f}).category_params.sigma_2 = 12;
        case 'diff_mean_same_std'
            tmp.(fields{f}).category_params.sigma_s = 5; % these params give a level of performance that is about on par withthe original task (above)
            tmp.(fields{f}).category_params.mu_1 = -4;
            tmp.(fields{f}).category_params.mu_2 = 4;
        case 'sym_uniform'
            tmp.(fields{f}).category_params.uniform_range = 15;
            tmp.(fields{f}).category_params.overlap = 0;
        case 'half_gaussian'
            tmp.(fields{f}).category_params.sigma_s = 5;
    end
    eval([fields{f} '= tmp.(fields{f});'])
end
clear tmp

if strcmp(P.stim_type, 'ellipse')
    Training.category_params.test_sigmas = .95;
    Test.category_params.test_sigmas = .4:.1:.9; % are these reasonable eccentricities?
else
    if attention_manipulation
        % AttentionTraining.category_params.test_sigmas = 1;
        % AttentionTrainingConf.category_params.test_sigmas = Test.category_params.test_sigmas;        
        Training.category_params.test_sigmas = 0.08;
        Test.category_params.test_sigmas = Training.category_params.test_sigmas;
%         Test.category_params.test_sigmas = exp(-3.5);
    else
        Training.category_params.test_sigmas = 1;
        Test.category_params.test_sigmas= exp(linspace(-5.5,-2,6)); % on previous rig: exp(-4:.5:-1.5)
    end
end

ConfidenceTraining.category_params.test_sigmas = Test.category_params.test_sigmas;

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
        case 'diff_mean_same_std'
            task_letter = 'A';
    end
    task_str       = ['Task ' task_letter ' '];
end

% number of trials, timing

if attention_manipulation    
    Demo.t.pres = 300; % 250
    Demo.t.betwtrials = 550; % 200

    ConfidenceTraining.n.blocks = 1;
    ConfidenceTraining.n.sections = 2;
    ConfidenceTraining.n.trials = 36; % WTA: 16

    Training.t.betwtrials = 800;

    Test.n.blocks = 3;
    Test.n.sections = 2;
    Test.n.trials = 40; % 9*numel(Test.sigma.int)*2 = 108

    Test.t.betwtrials = 800;
    
    %     AttentionTraining.n.blocks = 1;
    %     AttentionTraining.n.sections = 1;
    %     AttentionTraining.n.trials = 36;
    %
    %     AttentionTrainingConf.n = AttentionTraining.n;
    
    %     PreTest.n.blocks = 1;
    %     PreTest.n.sections = 1;
    %     PreTest.n.trials = 40; % 9*numel(Test.sigma.int)*2 = 108    
else
    Demo.t.pres = 250;
    Demo.t.betwtrials = 200;
    
    ConfidenceTraining.n.blocks = 1;
    ConfidenceTraining.n.sections = 1;
    ConfidenceTraining.n.trials = 24;

    Training.t.betwtrials = 1000; %1000

    Test.n.blocks = 3;% WTA from 3
    Test.n.sections = 3; % WTA from 3
    Test.n.trials = 8*numel(Test.category_params.test_sigmas); % 9*numel(Test.sigma.int)*2 = 108
    
    Test.t.betwtrials = 1000;   %1000

end

Training.initial.n.blocks = 1; %Do Not Change
Training.initial.n.sections = 2; % WTA: 2
Training.initial.n.trials = 36;% WTA: 36

Training.n.blocks = Test.n.blocks; % was 0 before, but 0 is problematic.
Training.n.sections = 2; %changed from '2' on 10/14
Training.n.trials = 48; % WTA: 48

Training.t.pres = 300; %300 % time stimulus is on screen
Training.t.pause = 100; %100 time between response and feedback
Training.t.feedback = 1100;  %1700 time of "correct" or "incorrect" onscreen
Training.t.cue_dur = 150;
Training.t.cue_target_isi = 150;

Test.t.pres = 50;           % time stimulus is on screen
Test.t.pause = 100; % time between response and feedback
Test.t.feedback = 1200;
Test.t.cue_dur = 300; %150
Test.t.cue_target_isi = 300; %150



if strfind(subject_name,'fast') > 0 % if 'fast' is in the initials, the exp will be super fast (for debugging)
    [Test.t.pres,Test.t.pause,Test.t.feedback,Test.t.betwtrials,Training.t.pres,Training.t.pause,...
        Training.t.feedback,Training.t.betwtrials,scr.countdown_time, Demo.t.pres, Demo.t.betwtrials,...
        Training.t.cue_dur, Training.t.cue_target_isi, Test.t.cur_dur, Test.t.cue_target_isi]...
        = deal(1);
end

if strfind(subject_name,'short') > 0 % if 'short' is in the initials, the exp will be short (for debugging)
    [Test.n.trials,Training.initial.n.trials,ConfidenceTraining.n.trials,Training.n.trials,nDemoTrials]...%, AttentionTraining.n.trials, AttentionTrainingConf.n.trials, PreTest.n.trials]...
        = deal(4);
    scr.countdown_time = 5;
end

if strfind(subject_name,'notrain') > 0 % if 'notrain' is in the initials, the exp will not include training (for debugging)
    notrain = true;
else
    notrain = false;
end

if strfind(subject_name, 'nodemo') > 0
    nodemo = true;
else
    nodemo = false;
end

if strfind(subject_name, 'noconf') > 0
    noconftraining = true;
else
    noconftraining = false;
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
            calib=load('calibration/iPadGammaTable');
            Screen('LoadNormalizedGammaTable', scr.win, calib.gammaTable*[1 1 1]);
        case 'Carrasco_L1'
            % Check screen resolution and refresh rate - if it's not set correctly to
            % begin with, the color might be off
            res = Screen('Resolution', screenid);
            if ~all([res.width res.height res.hz] == [scr.res scr.displayHz])
                error('Screen resolution and/or refresh rate has not been set correctly by the experimenter!')
            end

            % GRAYSCALE
            % calib = load('../../Displays/0001_james_TrinitonG520_1280x960_57cm_Input1_140129.mat');
            % rgbtable = calib.calib.table*[1 1 1];
            calib = load('calibration/carrasco_l1_calibration_42015.mat')
%             calib = load('calibration/carrasco_l1_calibration_42215_grayscale.mat');
            rgbtable = calib.gammaTable1*[1 1 1];
            
            % RGB
            % calib = load('calibration/carrasco_l1_calibration_42215_rgb.mat');
            % rgbtable = calib.rgb_gammatable;
            
            Screen('LoadNormalizedGammaTable', scr.win, rgbtable);
            
            % check gamma table
            gammatable = Screen('ReadNormalizedGammaTable', scr.win);
            if nnz(abs(gammatable-rgbtable)>0.0001)
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
    
    scr.rad = P.eye_rad*P.pxPerDeg;
    
    %set up fixation cross
    f_c = bg*ones(f_c_size);
    f_c(f_c_size/2 - fw: f_c_size/2 + 1 + fw,:) = darkgray;
    f_c(:,f_c_size/2 - fw: f_c_size/2 + 1 + fw) = darkgray;
    scr.cross = Screen('MakeTexture', scr.win , f_c);

    if attention_manipulation
        arm_rows = f_c_size/2-fw:f_c_size/2 + 1 + fw;
        L_arm_cols = 1:f_c_size/2-1-fw;
        R_arm_cols = f_c_size/2+2+fw:f_c_size;
        
        cross_whiteL = f_c;
        cross_whiteL(arm_rows, L_arm_cols) = white;
        cross_whiteLR = cross_whiteL; % neutral cue
        cross_whiteLR(arm_rows, R_arm_cols) = white;
        
        cross_grayL = cross_whiteL;
        cross_grayL(cross_grayL==white) = lightgray;
        
        scr.cueL = Screen('MakeTexture', scr.win, cross_whiteL);
        scr.cueR = Screen('MakeTexture', scr.win, fliplr(cross_whiteL));
        scr.cueLR= Screen('MakeTexture', scr.win, cross_whiteLR);
        
        scr.resp_cueL = Screen('MakeTexture', scr.win, cross_grayL);
        scr.resp_cueR = Screen('MakeTexture', scr.win, fliplr(cross_grayL));
    end
    
    % set up grating parameters
    P.grateSigma = .8; % Gabor Gaussian envelope standard deviation (degrees)
    P.grateSigma = P.grateSigma * P.pxPerDeg; %...converted to pixels
    P.grateAspectRatio = 1;
    P.grateSpatialFreq = .8; % cycles/degree
    P.grateSpatialFreq = P.grateSpatialFreq / P.pxPerDeg; % cycles / pixel
    P.grateSpeed = 6; % cycles per second % formerly 10
    P.grateDt = 1/scr.displayHz; %seconds per frame % formerly .01
    P.grateAlphaMaskSize = round(10*P.grateSigma);
    
    % Ellipse parameters
    P.ellipseAreaDegSq = 1; % ellipse area in degrees squared
    P.ellipseAreaPx = P.pxPerDeg^2 * P.ellipseAreaDegSq; % ellipse area in number of pixels
    P.ellipseColor = 0;
    
    P.attention_stim_spacing = 7; % 5.5 % for two stimuli, distance from center, in degrees (5 to 8, 4/21/15)
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
    
    ConfidenceTraining.R = setup_exp_order(ConfidenceTraining.n, ConfidenceTraining.category_params);
    Test.R = setup_exp_order(Test.n, Test.category_params);
%     PreTest.R = setup_exp_order(PreTest.n, Test.category_params);
    
    if attention_manipulation
        ConfidenceTraining.R2 = setup_exp_order(ConfidenceTraining.n, ConfidenceTraining.category_params, cue_validity);
        Test.R2 = setup_exp_order(Test.n, Test.category_params, cue_validity); % second set of stimuli. This also contains the probe/cue info.
%         PreTest.R2 = setup_exp_order(PreTest.n, Test.category_params, cue_validity); % second set of stimuli. This also contains the probe/cue info.

%         AttentionTraining.R = setup_exp_order(AttentionTraining.n, AttentionTraining.category_params);
%         AttentionTraining.R2 = setup_exp_order(AttentionTraining.n, AttentionTraining.category_params, cue_validity);
%         
%         AttentionTrainingConf.R = setup_exp_order(AttentionTrainingConf.n, AttentionTrainingConf.category_params);
%         AttentionTrainingConf.R2 = setup_exp_order(AttentionTrainingConf.n, AttentionTrainingConf.category_params, cue_validity);
    end
    
    %% Start eyetracker
    if eye_tracking
        % Initialize eye tracker
        [el exit_flag] = rd_eyeLink('eyestart', scr.win, eye_file);
        if exit_flag
            return
        end
        
        % Write subject ID into the edf file
        Eyelink('message', 'BEGIN DESCRIPTIONS');
        Eyelink('message', 'Subject code: %s', subject_name);
        Eyelink('message', 'END DESCRIPTIONS');
        
        % No sounds for drift correction
        el.drift_correction_target_beep = [0 0 0];
        el.drift_correction_failed_beep = [0 0 0];
        el.drift_correction_success_beep = [0 0 0];
        
        % Accept input from all keyboards
        el.devicenumber = -1; %see KbCheck for details of this value
        
        % Update with custom settings
        EyelinkUpdateDefaults(el);
        
        % Calibrate eye tracker
        [cal exit_flag] = rd_eyeLink('calibrate', scr.win, el);
        if exit_flag
            return
        end
        
        scr.el = el; % store el in scr
    end
    
    start_t = tic;
    
    %% Show example stimulus    
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
            switch task_letter
                case 'A'
                    midtxt = 'In Task A, stimuli from Category 1 tend to\n\nbe left-tilted, and stimuli from Category 2\n\ntend to be right-tilted.\n\nSee Task sheet for more info.';
                case 'B'
                    midtxt = 'In Task B, a flat stimulus is more likely\n\nto be from Category 1, and a strongly tilted\n\nstimulus is more likely\n\nto be from Category 2.\n\nSee Task sheet for more info.';
                otherwise
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
                % UNCOMMENT THIS BLOCK!!!!
                [nx,ny]=DrawFormattedText(scr.win, 'Let''s get some practice with the\n\ncategories we''ll be using in this task.', 'center', 'center', color.wt);
                flip_key_flip(scr,'continue',ny,color,new_subject);
                if ~nodemo
                    category_demo
                end
                
                Training.initial.n.blocks = Training.n.blocks;
                [Training.responses{k}, flag] = run_exp(Training.initial.n, Training.R, Training.t, scr, color, P, 'Training',k, new_subject, task_str, final_task, subject_name);
                if flag ==1,  break;  end

                if attention_manipulation && new_subject
                    % attention and probe demo
                    
                    demostim(1).ort = -45;
                    demostim(2).ort = -4;
                    demostim(1).phase = 360*rand;
                    demostim(2).phase = 360*rand;
                    demostim(1).cur_sigma = Test.category_params.test_sigmas;
                    demostim(2).cur_sigma = Test.category_params.test_sigmas;
                    
                    % display static 2-stimulus screen
                    grate(P, scr, t, demostim, true)
                    
                    % display probe
                    Screen('DrawTexture', scr.win, scr.resp_cueL);
                    Screen('Flip', scr.win);
                    flip_key_flip(scr, 'continue', scr.cy, color, true);
                    
                
                if ~noconftraining
                    if attention_manipulation
                        [ConfidenceTraining.responses, flag] = run_exp(ConfidenceTraining.n,ConfidenceTraining.R,Test.t,scr,color,P,'Confidence Training',k, new_subject, task_str, final_task, subject_name, ConfidenceTraining.R2);
                    else
                        [ConfidenceTraining.responses, flag] = run_exp(ConfidenceTraining.n,ConfidenceTraining.R,Test.t,scr,color,P,'Confidence Training',k, new_subject, task_str, final_task, subject_name);
                    end
                end
                if flag ==1,  break;  end

%                 if attention_manipulation
%                     [AttentionTraining.responses, flag] = run_exp(AttentionTraining.n,AttentionTraining.R, Training.t,scr,color,P,'Attention Training',k, new_subject, task_str, final_task, subject_name, AttentionTraining.R2);
%                     if flag==1,break;end
%                     
%                     [AttentionTrainingConf.responses, flag] = run_exp(AttentionTrainingConf.n,AttentionTrainingConf.R, Test.t,scr,color,P,'Attention Training Conf',k, new_subject, task_str, final_task, subject_name, AttentionTrainingConf.R2);
%                     if flag==1,break;end
%                 end
%
%                 if attention_manipulation
%                     [Test.responses{k}, flag] = run_exp(PreTest.n, PreTest.R, Test.t, scr, color, P, 'PreTest',k, new_subject, task_str, final_task, subject_name, PreTest.R2);
%                 end
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
    
    save(strrep([datadir '/' subject_name '_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'ConfidenceTraining', 'AttentionTraining', 'P', 'category_type', 'elapsed_mins') % save complete session
    recycle('on'); % tell delete to just move to recycle bin rather than delete entirely.
    delete([datadir '/backup/' subject_name '_' datetimestamp '.mat']) % delete the block by block backup
    
    %% Save eye data and shut down the eye tracker
    if eye_tracking
        rd_eyeLink('eyestop', scr.win, {eye_file, eye_data_dir});
        
        % rename eye file
        eye_file_full = sprintf('%s/%s_CategoricalDecision_%s.edf', eye_data_dir, subject_name, datestr(now, 'yyyymmdd'));
        copyfile(sprintf('%s/%s.edf', eye_data_dir, eye_file), eye_file_full)
    end
    
    Screen('CloseAll');    
    
catch %if error or script is cancelled
    Screen('CloseAll');
    
    save(strrep([datadir '/backup/' subject_name '_recovered_' datetimestamp '.mat'],'/',filesep), 'Training', 'Test', 'P','category_type', 'elapsed_mins')
    
    psychrethrow(psychlasterror);
end

end