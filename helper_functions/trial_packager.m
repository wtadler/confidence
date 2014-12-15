function trial = trial_packager(raw, trial_number)

fields = fieldnames(raw);

for i = 1 : length(fields)
    trial.(fields{i}) = raw.(fields{i})(trial_number);
end
trial.Chat = 2 * trial.Chat - 3; % remap Chat 1 and 2 onto -1 and 1.
trial.C    = 2 * trial.C    - 3; % remap Chat 1 and 2 onto -1 and 1.