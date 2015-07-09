function one_model_one_task_multiple_subjects_multiple_stats(model, datadir, nBins, marg_over_s)

datadir = '/Users/will/Google Drive/Will - Confidence/Data/v3/taskA';


real_st = compile_data('datadir', datadir);
nSubjects = length(model.extracted);
if nSubjects ~= length(real_st.data)
    error('something''s wrong')
end

% need to generate fake data from data_plots. complicated, esp with dual tasks. maybe start from scratch with the single task here?
show_data(datadir, 'nBins', nBins, 'marg_over_s', marg_over_s)