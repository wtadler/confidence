function st = compile_data(varargin)
% compiles all real data. formerly, generated stats. now thinking that that
% should be left out.
% define defaults
% sig_levels = 6; % defines by how much we group the sigma values. 6 is no grouping. can do 1,2,3,6
% flipsig = true;
% n_bins=19; % must be odd to plot a point at 0.
% binstyle = 'quantile';
% o_boundary=25;
shuffle = false;

conflevels = 4;
% decb_analysis = false; % set to true if you want to look at choice and confidence at the decision boundary.
%     window = 1; % binning window (in degrees) around decision boundary
%     decb   = 5.16; % decision boundary (in degrees)
datadir='/Users/will/Google Drive/Will - Confidence/Data/v3/taskA';

crossvalidate = false;
    k = 2; % for k-fold cross-validation

old_attention_manipulation = false;
attention_manipulation = false;
multi_prior = false;

training_data = false; % compiles training data instead of test data

assignopts(who,varargin);

if ~any(regexp(datadir, '/$'))
    datadir = [datadir '/'];
end

% load all complete sessions
session_files = what(datadir);
session_files = session_files.mat;
% session_files= dir([datadir '*.mat']) % this is probably the proper thing
% to do but having problems with regexp below

% find unique subject names
names = regexp(session_files,'^[a-zA-Z0-9]+(?=_)','match'); % find characters before _ in session_files. make this accept caps
names = unique(cat(1,names{:}));

% compile raw data for individual subjects, compute individual stats, and summary stats.
st = struct; % probably want to pre-allocate this in some way.

%[st.bins, st.axis] = bin_generator(n_bins, 'binstyle', binstyle);
% optionally, to examine what's happening at the decision boundary, make different bins:
% if decb_analysis
%     [st.bins, st.axis, n_bins] = bin_generator(n_bins, 'binstyle', 'defined','o_axis',...
%         [-decb-window -decb -decb+window decb-window decb decb+window]); % redefine bins in this case
% end

for subject = 1 : length(names)
    % load all files with name
    subject_sessions = dir([datadir names{subject} '_*.mat']);

    % initialize raw fields
    clear raw
    fields = {'C', 's', 'contrast', 'probe', 'cue', 'cue_validity', 'prior', 'Chat', 'g', 'tf', 'rt', 'resp'};
    for f = 1:length(fields)
        raw.(fields{f}) = [];
    end
    
    for session = 1 : length(subject_sessions); % for each session
        load([datadir subject_sessions(session).name])
        
        if training_data
            data = Training;
        else
            data = Test;
        end
        
        if isfield(data, 'R2')
            old_attention_manipulation = true;
        elseif ndims(data.R.draws{1}) == 3
            attention_manipulation = true;
        end
        
        if isfield(data.R, 'prior') && length(unique(data.R.prior{1})) > 1
            multi_prior = true;
        end
        
        
        
        %tmp.Training = Training; % maybe work on this later. it's going to
        %change how the data comes out and might mess with other scripts.
        %tmp.data = data;
        %TrTest = {'Training', 'data'}
        
        for block = 1:length(data.responses)
            for section = 1:size(data.responses{block}.c,1)
                %                 start_trial = (session - 1) * data.n.blocks * data.n.sections * data.n.trials + (block - 1) * data.n.sections * data.n.trials + (section - 1) * data.n.trials + 1;
                %                 end_trial   = (session - 1) * data.n.blocks * data.n.sections * data.n.trials + (block - 1) * data.n.sections * data.n.trials + (section - 1) * data.n.trials + data.n.trials;
                nTrials = size(data.R.draws{block}, 2);
                st.data(subject).name = names{subject};
                
                % trials
                raw.C           = [raw.C        data.R.trial_order{block}(section,:) * 2 - 3];
                
                if attention_manipulation
                    draws = data.R.draws{block}(section, :, :);
                    sigma = data.R.sigma{block}(section, :, :);
                    probe = data.R.probe{block}(section,:);
                    index = sub2ind(size(draws), ones(1, nTrials), 1:nTrials, probe);
                    
                    raw.s = [raw.s draws(index)];
%                     raw.contrast = [raw.contrast sigma(index)];
                    raw.cue_validity = [raw.cue_validity data.R.cue_validity{block}(section, :)];
                    [raw.contrast_values, raw.contrast_id] = unique_contrasts(raw.cue_validity);
                    raw.cue_validity_id = raw.contrast_id;
                elseif old_attention_manipulation
                    raw.probe = [raw.probe data.R2.probe{block}(section,:)];
                    raw.cue   = [raw.cue   data.R2.cue{block}(section,:)];
                    
                    raw.cue_validity(raw.probe == raw.cue)                  =  1;  % valid cues
                    raw.cue_validity(raw.cue == 0)                          =  0;  % neutral cues
                    raw.cue_validity(raw.probe ~= raw.cue & raw.cue ~= 0)   = -1; % invalid cues
                    
                    [raw.contrast_values, raw.contrast_id] = unique_contrasts(raw.cue_validity);%, 'flipsig', flipsig);

                    raw.cue_validity_id = 2 - raw.cue_validity; % maps [-1 0 1] onto [3 2 1]. i think this is the same as raw.contrast_id
                    
                    % vector of order of trials. eg, [1 2 3 2] indicates that trial 2 was repeated. nothing is recorded for the first attempt at trial 2
                    trial_order = data.responses{block}.trial_order{section};
                    
                    % there must be a better way to do this part.
                    % flip, go backwards, finding unique trial numbers. flip again. will result in, e.g., [1 3 2]
                    flipped_trial_order = fliplr(trial_order);
                    trials = [];
                    for trial = 1:length(flipped_trial_order)
                        if ~any(flipped_trial_order(trial)==trials)
                            trials = [trials flipped_trial_order(trial)];
                        end
                    end
                    trial_order = fliplr(trials);
                    
                    % re-order all trial info.
                    fields = setdiff(fieldnames(raw), 'contrast_values'); % because contrast_values is not a nTrials list
                    for f = 1:length(fields)
                        cur_trials = raw.(fields{f})(end-nTrials+1:end);
                        raw.(fields{f})(end-nTrials+1:end) = cur_trials(trial_order);
                    end
                else
                    raw.s           = [raw.s        data.R.draws{block}(section,:)];
                    raw.contrast    = [raw.contrast data.R.sigma{block}(section,:)];
                    [raw.contrast_values, raw.contrast_id] = unique_contrasts(raw.contrast);%,'flipsig',flipsig);
                end
                
                if multi_prior
                    raw.prior = [raw.prior data.R.prior{block}(section, :)];
                    [raw.prior_values, raw.prior_id] = unique_contrasts(raw.prior, 'flipsig', true);
                end
                    
                    
                
%                 if numel(contrast_values) ~= sig_levels  % group and average contrast values if sig_levels specifies something different than the normal 6 raw.contrast_values.
%                     newcontrast       = nan(1,sig_levels);
%                     contrast_valuespergroup = length(contrast_values) / sig_levels;
%                     for i=1:sig_levels;
%                         contrast_range = (i-1)*contrast_valuespergroup+1 : (i-1)*contrast_valuespergroup + contrast_valuespergroup;
%                         raw.contrast_id(ismember(raw.contrast_id,contrast_range)) = i;
%                         newcontrast(i) = mean(contrast_values(contrast_range));
%                     end
%                     raw.contrast_values = newcontrast;
%                     clear newcontrast;
%                 end
                
                % responses
                
                raw.Chat    = [raw.Chat data.responses{block}.c(section,:) * 2 - 3];
                raw.tf      = [raw.tf   data.responses{block}.tf(section,:)];
                raw.g       = [raw.g    data.responses{block}.conf(section,:)];
                raw.rt      = [raw.rt   data.responses{block}.rt(section,:)];
                raw.resp    = [raw.resp data.responses{block}.conf(section,:) + conflevels + ...
                    (data.responses{block}.c(section,:)-2) .* ...
                    (2 * data.responses{block}.conf(section,:) - 1)];
                                
            end
        end
    end
%     fields = fieldnames(raw);
%     fields = fields([1 2 3 5 6 7 8 9 10]); % drop 'contrast_values'
    
    if crossvalidate % can remove the second clause to get lots of samples for each subject (see below)
        nTrials = length(raw.C);
        if rem(nTrials,k)~=0
            error(sprintf('error: the number of subject %i trials (%i trials) cannot be divided into %i sets without remainder.',subject,nTrials,k))
        else
            nTestTrials = nTrials/k;
        end
        
        for sample = 1 : k
            start_trial = (sample-1)*nTestTrials + 1;
            end_trial   = (sample)*nTestTrials;
            test_idx    = start_trial:end_trial;
            train_idx   = setdiff(1:nTrials,test_idx);

            for field = 1 : length(fields)
                st.data(subject).train(sample).(fields{field}) = raw.(fields{field})(train_idx);
                st.data(subject).test(sample).(fields{field}) = raw.(fields{field})(test_idx);
            end
        end
    else % if not cross-validating
        
        if shuffle
            idx=randperm(length(raw.C));
            for field = 1 : length(fields)
                raw.(fields{field}) = raw.(fields{field})(idx);
            end
        end
        
        st.data(subject).raw = raw;
    end
    
    
    
end