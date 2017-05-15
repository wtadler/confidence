datadir = check_datadir('~/Google Drive/Will - Confidence/Data/v3_all');
st.A = compile_data('datadir', datadir.A);
st.B = compile_data('datadir', datadir.B);

% matrix should have subject, stimtype, task, s, contrast_id, Chat, g, tf, resp,rt;
%%
grate = [1 2 4 5 7];
ellipse = [3 6 8:11];

file = fopen('v3_data.csv','w');
fprintf(file, 'subject,stimtype,task,s,contrast_id,Chat,g,tf,resp,rt,C\n')
            
tasks = {'A', 'B'};
for s = 1:11
    for task = 1:2
        if any(s==grate)
            stim = 'grate';
        else
            stim = 'ellipse';
        end
        
        data = st.(tasks{task}).data(s).raw;
        
        for t = 1:2160
            fprintf(file, '%i,%s,%s,%.6g,%i,%i,%i,%i,%i,%.6g,%i\n',...
                s, stim, tasks{task}, data.s(t), data.contrast_id(t), data.Chat(t),...
                data.g(t), data.tf(t), data.resp(t), data.rt(t), data.C(t));
        end
    end
        
end


% m = zeros(nTrials, 10);

%%
load('~/Google Drive/Will - Confidence/Analysis/A_choice_noise_params.mat');
bp = best_params_free(4:9,:);
csvwrite('logsigmas.csv',bp)
%%

