function savedatafcn(name, datetime, Training, Test, P)
% OBSOLETE!

%%% Save data to file %%%

%Convert current date to string
% Date = datevec(date);
% year = num2str(Date(1));
% month = num2str(Date(2));
% day = num2str(Date(3));
% 
% if ((Date(2) >= 10) && (Date(3) >= 10))
%     file_date = [year, month, day];
% elseif ((Date(2) < 10) && (Date(3) >= 10))
%     file_date = [year, '0', month, day];
% elseif ((Date(2) >= 10) && (Date(3) < 10))
%     file_date = [year, month, '0', day];
% elseif ((Date(2) < 10) && (Date(3) < 10))
%     file_date = [year, '0', month, '0', day];
% end


%letter_sub = [97:122]; %no observer is expected to run more than 26 sessions/day!
%letter1 = 1;

%dot_mat = '.mat';

%file_name = [name eval('letter_sub(letter1)') dot_mat];

% 
% for (letter1 = 1:26)
%     if (isempty(dir(['data/' file_name])) == 1) % if file with this name doesn't exist
%         break; %break for loop, save file.
%     end
%     file_name = [name eval('letter_sub(letter1)') dot_mat]; %if file does exist, try with the next letter1.
% end
% 
% savedest = [cd '/data/' file_name];

savedest = [cd '/data/' name '_' datetime '.mat'];

save(savedest, 'Training', 'Test', 'P')