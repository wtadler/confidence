function HVIllusioncode_test

% Horizontal-Vertical Illusion Experiment

% This experiment measures how human eyes may perceive the length of 
% straight lines differently depending on the orientations
%%
%------- EXPERIMENT SETTINGS -------
displayTime       = 0.3;                                      % display time for both stimuli
delayTime         = 0.5;
comparisonLength  = [4 4.25 4.5 4.75 5 5.25 5.5 5.75 6]';     % total of 9 reference lengths (in cm)in a vertical array
referenceLength   = 5;                                        % standard length (in cm)
orientations      = [300 330 0 30 60 90]';                    % convert orientations from degrees to radians and store in a vertical array; [5.2360 5.7596 0 1.5708 1.0472 0.5236]                                                              
nTrialsPerSession = 576;                                      % 576 trials per session 
nBlocksPerSession = 6;
nTrialsPerBlock   = 96;
breakTime         = 30;                                       % length of break between blocks (in seconds)
lineWidth         = 3;                                        % line width for the stimuli (for 'DrawLine')
lineColor         = 250;                                      % white
bgcolor           = 128;                                      % set background color to be a certain grey
fixation_color    = [200 0 0];                                % bright red
fixation_size     = [0 0 7 7];                                % size of the red dot
count             = 0;                                        % number of trials subject gave the correct answer
textSize          = 20;
subjectDistance   = 57;                                       % distance between observer and screen (in cm)    !!!TBD!!!
screen_width      = 18;                                       % in cm; data collected by measuring !!!
%-----------------------------------

clc;  % clears the command window
subjectID = [];
session_num = [];

% check if the subject's info has been entered
while isempty(subjectID) && isempty(session_num) 
    subjectID = input('Subject initials: ','s');
    session_num = input('Number of session: ');
end

% define button keys
% map keys with 'WaitSecs(0.2); [~, keyCode] = KbWait;find(keyCode==1)' and
% then press a key.
if any(regexp(subjectID,'will'))
    key1=30;
    key2=31;
    keyesc=41;
    cd('/Users/will/Ma lab/repos/qamar confidence/emma')
else
    key1=49;
    key2=50;
    keyesc=27;
end

% Create a vector for 576 comparison lengths
i = repmat(comparisonLength,64,1); % create in total 576 comparison lengths (9 * 64)
pseudo_random = randperm(nTrialsPerSession); % generate a vector with integers from 1 to 576 ordered randomly
CLArray = i(pseudo_random); % shuffle all 576 comparison lengths randomly in the vertical array

% % Create a vector for 576 reference lengths
% first_line = ones(288,1); % when the first line is 5cm long
% second_line = repmat(2,288,1); % when the second line is 5cm long
% vector = [first_line',second_line']'; % append the second vector after the first vector
% RLArray = vector; % NOT shuffle 1s and 2s in the vertical vector

% Create a vector for 576 orientations
j = repmat(orientations,96,1); % create in total 576 orientations (6 * 96)
firstOrientation = j(pseudo_random); % shuffle 576 orientations randomly in the vertical vector, for the first line
pseudo_random1 = randperm(nTrialsPerSession);
secondOrientation = j(pseudo_random1); % shuffle 576 orientations randomly for the second line

% Create a vector for 576 reference lengths
referenceL = repmat(referenceLength, nTrialsPerSession,1);
% a 576 by 1 vector filled with zeros 
empty = zeros(nTrialsPerSession,1); 
correct_choice = empty;
key = empty;
correct = empty;


% open screen
%HideCursor;
Screen('Preference', 'SkipSyncTests',1)

%screenid = max(Screen('Screens'));
screenid = 1;

% OPEN FULL SCREEN
%[scr.win, scr.rect] = Screen('OpenWindow', screenid, 128); %scr.win is the window id (10?), and scr.rect is the coordinates in the form [ULx ULy LRx LRy]
%[windowPtr,rect] = Screen('OpenWindow',screenid,bgcolor,[],32,2); % why
%32,2?
[windowPtr,rect] = Screen('OpenWindow', screenid, bgcolor);

Screen('BlendFunction', windowPtr,'GL_SRC_ALPHA','GL_ONE_MINUS_SRC_ALPHA')
%%
% OPEN IN WINDOW
%[scr.win, scr.rect] = Screen('OpenWindow', screenid, 128, [100 100 1200 1000]);
%[windowPtr,rect] = Screen('OpenWindow',screenid,bgcolor,[100 100 1200 1000],32,2);
rect


[mx, my] = RectCenter(rect);

% show start screen
Screen('fillRect',windowPtr,bgcolor); % set background color to be a certain grey
Screen('TextSize',windowPtr,textSize);  % set text's size

% all the text will be shown in black (white is most preferred)
instructionText = [ 'This experiment session consists of ' num2str(nBlocksPerSession) ' blocks of trails.\n'...
    '\n'...
    'In each trial,two lines will be shown subsequently\n'...
    '\n'...
    'Your job is to judge which line is longer.\n'...
    '\n'...
    '\n'...
    'PRESS "1" if first line looks longer, \n'...
    '\n'...
    'PRESS "2" if second line looks longer.\n'...
    '\n'...
    '\n'...
    '\n'...
    'Note: the lines will be facing different directions, and\n'...
    '\n'...
    'the first lines will always be the SAME length.\n'...
    '\n'...
    '\n'...
    'You will have a break after each block.\n'...
    '\n'...
    '\n'...
    '\n'...
    '\n'...
    '\n'...
    'Press any key to start the experiment.\n'];

% draw instructionText, centered in the display window
DrawFormattedText(windowPtr,instructionText,'center','center', [0 0 0]);
Screen('flip',windowPtr);

KbWait([], 2);

%-%-%-%-%-
%- INIT %-
%-%-%-%-%-

AssertOpenGL;                % check whether Psychtoolbox is working properly with "Screen()" functions
KbName('UnifyKeyNames');
KbCheck;
% screen info
%[w, h]=Screen('WindowSize', screenid);             % screen resolution--[w h] = [1024 768]
w = rect(3);
h = rect(4);
screen_resolution = [w, h];                 % screen resolution--screen_resolution = [1024 768]
screen_angle = 2*(180/pi)*(atan((screen_width/2) / subjectDistance)) ; % total visual angle of screen--screen_angle = 
screen_ppd = screen_resolution(1) / screen_angle;  % pixels per degree--screen_ppd = 
screen_fixposxy = screen_resolution .* [.5 .5];    % fixation position--screen-fixposxy = [512 384]

settings.delayTime = delayTime;
settings.displayTime = displayTime;
settings.referenceLength = referenceLength;
settings.lineWidth = lineWidth;
settings.subjectDistance = subjectDistance;


for ii = 1:nTrialsPerSession

    %-%-%-%-%-%-%-%-%-%
    % Generate trials %
    %-%-%-%-%-%-%-%-%-%
    outputfile = ['output/' upper(subjectID) '_' num2str(session_num) '.mat'];
       
    firstAngle = firstOrientation(ii);
    secondAngle = secondOrientation(ii);
    % the reference line(5cm) will always be the first line
    % the comparison line(4-6cm) always be the second
    firstLine = referenceL(ii);
    secondLine = CLArray(ii);
    
%     choice = RLArray(ii);    
%     % decide which line is the reference line; if choice = 2,the second
%     % line is the reference line, if choice = 1, first line is the
%     % reference line
%     if choice == 2 % second column of 'matrix'
%         firstLine = referenceL(ii);
%         secondLine = CLArray(ii);
%         
%         
%     elseif choice == 1
%         firstLine = CLArray(ii);
%         secondLine = referenceL(ii);
%         temp = referenceL(ii);
%         referenceL(ii) = CLArray(ii);
%         CLArray(ii) = temp;
%         
%     end
    
    if firstLine > secondLine
        correct_choice(ii) = 1;
        
    elseif secondLine > firstLine
        correct_choice(ii) = 2;
    elseif  firstLine == secondLine
        correct_choice(ii) = 0;
        
    end
    
    %-%-%-%-%-%-%-%-%-%-%-%-%
    %- LOOP THROUGH TRIALS %-
    %-%-%-%-%-%-%-%-%-%-%-%-%
    nextFlipTime = 0; % just to initialize
    
    % SCREEN 1: FIXATION
    Screen('FillRect',windowPtr, bgcolor);
    rect = CenterRectOnPoint(fixation_size, screen_fixposxy(1), screen_fixposxy(2)); % decides the size of the circle and the place of the circle; all further 'FillOval' depends on this statement
    Screen('FillOval',windowPtr, fixation_color, rect);
    pause(delayTime);
    Screen('Flip',windowPtr,nextFlipTime);
    nextFlipTime = GetSecs + delayTime;

    
    % SCREEN 2: STIMULUS
    Screen('FillRect',windowPtr, bgcolor);
%     Screen('FillOval',windowPtr, fixation_color, rect);
    screenLength1 = (atan(firstLine/2*subjectDistance)/screen_width)* 180/pi *screen_ppd; % convert line length from cm to degrees of visual angle
    [x, y] = pol2cart(firstAngle,screenLength1); % convert from polar coordinate to Cartesian coordinate
    complementaryAngle = firstAngle + pi; % complementary angle (in radian) is angle + 180degrees
    [x1,y1] = pol2cart(complementaryAngle,screenLength1);

    
    grabSize = 18 * screen_ppd;
    grabrect = CenterRectOnPoint([0 0 grabSize grabSize],w/2,h/2);
    i = sqrt((x - x1)^2 + (y - y1)^2);

    center = screenLength1 / 2;

    %%%%%%%% your line drawing code should look something like this:
    x_center_px = w/2;
    y_center_px = h/2;
    line_length_px = firstLine * screen_ppd;
    add_x = cos(firstAngle) * line_length_px / 2;
    add_y = sin(firstAngle) * line_length_px / 2;
    Screen('DrawLines', windowPtr, [x_center_px + add_x, x_center_px - add_x; y_center_px + add_y, y_center_px - add_y], lineWidth, lineColor, [], 1);

    %im = drawBar(i,lineWidth,firstAngle,lineColor,bgcolor); %im is a matrix representation of the stimulus using the colors
    %tex = Screen('MakeTexture',windowPtr,im);
    %rect1 = CenterRectOnPoint([0 0 size(im')], w/2, h/2);
    %Screen('DrawTexture', windowPtr, tex ,[0 0 size(im')], rect1);
    
    pause(displayTime);
    Screen('Flip',windowPtr,nextFlipTime);
    nextFlipTime = GetSecs + displayTime;


    %in = Screen('getimage',windowPtr,grabrect);
    %imwrite(in,['screenshots/' datestr(now,30) '.png'],'png');
    
    
    % SCREEN 3: DELAY
    Screen('FillRect',windowPtr, bgcolor);
    Screen('FillOval',windowPtr, fixation_color, rect);
    pause(delayTime);
    Screen('Flip',windowPtr,nextFlipTime);
    nextFlipTime = GetSecs + delayTime;
    
    % SCREEN 4: STIMULUS
    Screen('FillRect',windowPtr,bgcolor);
%     Screen('FillOval',windowPtr, fixation_color, rect);
    screenLength2 = (atan(secondLine/2*subjectDistance)/screen_width)* 180/pi *screen_ppd; % convert line length from cm to degrees of visual angle
    [xx, yy] = pol2cart(secondAngle,screenLength2);
    complementaryAngle = secondAngle + pi; % complementary angle (in radian) is angle + 180degrees
    [xx1,yy1] = pol2cart(complementaryAngle,screenLength2);
    
%     display(screenLength1);
%     display(screenLength2);
    
    
%     grabSize = 18 * screen_ppd;
%     grabrect = CenterRectOnPoint([0 0 grabSize grabSize],w/2,h/2);
%     
    im = sqrt((xx - xx1)^2 + (yy - yy1)^2);
    im = drawBar(i,lineWidth, secondAngle,lineColor,bgcolor);
    save testout.mat
    tex = Screen('MakeTexture',windowPtr,im);
    rect1 = CenterRectOnPoint([0 0 size(im')], w/2, h/2);
    Screen('DrawTexture', windowPtr, tex ,[0 0 size(im')], rect1);    
    pause(displayTime);
    Screen('Flip',windowPtr,nextFlipTime);
    
%     in = Screen('getimage',windowPtr,grabrect);
%     imwrite(in,['screenshots/' datestr(now,30) '.png'],'png');
    
    nextFlipTime = GetSecs + displayTime;
    tic; % timer goes off
    
    % SCREEN 5: DELAY
    Screen('FillRect',windowPtr,bgcolor);
    Screen('FillOval',windowPtr, fixation_color, rect);
    pause(delayTime);
    Screen('Flip',windowPtr,nextFlipTime);
    
    % SCREEN 6: RESPONSE
    Screen('FillRect',windowPtr,bgcolor); % set background color to be a certain grey
    Screen('TextSize',windowPtr,textSize);
    responseQuestion = ('Which line is longer?(1 or 2)');
    DrawFormattedText(windowPtr,responseQuestion,'center','center');
    Screen('Flip',windowPtr);
    keyCode = zeros(1,256);
    while ~any([keyCode(key1), keyCode(key2), keyCode(keyesc)]) % keyCode(key1)="1", keyCode(key2)="2", keyCode(keyesc)="esc"
        [resptime, resps,keyCode] = KbCheck; % wait for subject to press "1" or "2" or "esc"
    end
    
    keyPress = KbName(keyCode); % find which key was pressed; translate code into string
    key(ii) = str2double(keyPress(1));
    data.reactionTime(ii) = toc;
        
    % save results in matrix
    
    % count the number of times the subject has chosen correctly
    if correct_choice(ii) == key(ii)
       count = count + 1;
       correct(ii) = count; 
    end
    
    % if both lines are of the same length, there is no correct answer(i.e.
    % NaN)
    if correct_choice(ii) == 0
        correct(ii) = NaN;
    end
    
    % records matrix 
    matrix = [referenceL,CLArray,firstOrientation,secondOrientation,correct_choice,key,correct];
    data.matrix = matrix;
    save(outputfile,'settings','data');
    
    
    % check if ESC is pressed !!!delete this when running subjects!!!
    %[keyIsDown,secs,keyCode] = KbCheck;
    if keyCode(keyesc) % check if ESC key is pressed
        Screen('closeall');
        error('Program aborted');
    end
    
    % BREAK TIME
    currentBlock = ii / nTrialsPerBlock;
    if (rem(ii,nTrialsPerBlock) == 0) && (currentBlock < 6) % if ii is a multiply of nTrialsPerBlock, the subject needs to take at last 30second break 
       Screen('TextSize',windowPtr,textSize);
       breakStart=GetSecs;
       
       while (GetSecs-breakStart)<breakTime
           Screen('fillRect',windowPtr,bgcolor);
           totalBreak = GetSecs-breakStart; % while loop, totalBreak changes every second as GetSecs gets the current time 
           breakText = ['Good Job! You have finished ' num2str(currentBlock) ' out of ' num2str(nBlocksPerSession) ' blocks.\n'...
               'Please take a short break now.'...
               'You can continue in ' num2str(ceil(breakTime-totalBreak)) ' seconds.\n'];
           DrawFormattedText(windowPtr,breakText,'center','center');
           Screen('flip',windowPtr);
       end
       % once 30s is up, the subject is prompted to press any key to resume
       % the experiment
       Screen('fillRect',windowPtr,bgcolor);
       endBreakText = ['You have finished ' num2str(currentBlock) ' out of ' num2str(nBlocksPerSession) ' blocks.\n'...
           'Press any key to continue.\n'];
       DrawFormattedText(windowPtr,endBreakText,'center','center');
       Screen('flip',windowPtr);
       [resptime, keyCode] = KbWait;
    end
    
     % check if all the trials of this session is finished
       if ii == nTrialsPerSession
           Screen('fillRect',windowPtr,128);
           endText = ['You have finished this session.'...
               'Please inform your instructor about your completion\n'...
               'Thank you for your participation!'];
           DrawFormattedText(windowPtr,endText,'center','center');
           Screen('flip',windowPtr)
       end
end


% combine everything into a matrix

%clean up before exit
ShowCursor;

% finalize
Screen('closeall');

%-%-%-%-%-%-%-%-%-%-%-%-%- HELPER FUNCTIONS %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-


