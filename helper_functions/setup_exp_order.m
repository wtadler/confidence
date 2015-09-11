function R = setup_exp_order(n, category_params, varargin)
%%%Set up order of class, sigma, and orientation for entire scheme%%%

R = [];

if isempty(varargin)
    nStimuli = 1;
else
    nStimuli = varargin{1};
    cue_validity = varargin{2};
    prop_neutral_trials = 1/3;
end


for k = 1:n.blocks
    R.category_type = category_params.category_type;
    R.trial_order{k} = reshape(randsample(2,                           n.trials * n.sections * nStimuli, true),...
        n.sections, n.trials, nStimuli);
    R.sigma{k} =       reshape(randsample(category_params.test_sigmas, n.trials * n.sections * nStimuli, true),...
        n.sections, n.trials, nStimuli);
    
    %get random orientation draws from normal distributions (~sigmal or ~sigma2)
    %CORRESPONDING TO order of classes.
    classtwos = (R.trial_order{k} == 2); % index of class 2 trials
    R.phase{k} = 360*rand(n.sections, n.trials, nStimuli); % random phase draws
    
    R.draws{k} =    reshape(stimulus_orientations(category_params, 1, n.sections * n.trials * nStimuli, category_params.category_type), n.sections, n.trials, nStimuli);
    R.draws{k}(classtwos) = stimulus_orientations(category_params, 2, nnz(classtwos),      category_params.category_type);
    
    if nStimuli > 1
        
        R.probe{k} = reshape(randsample(nStimuli, n.trials*n.sections, true), n.sections, n.trials);
        R.categories{k} = R.trial_order{k};
        R.trial_order{k} = zeros(n.sections, n.trials);
        for i = 1:nStimuli
            [rows, cols] = find(R.probe{k} == i);
            R.trial_order{k}(R.probe{k} == i) = R.categories{k}(sub2ind(size(R.categories{k}), rows, cols, i*ones(size(rows))));
        end
        
        randmat = rand(n.sections, n.trials);
        R.cue_validity{k} = ones(n.sections, n.trials); % start all trials valid
        R.cue_validity{k}(randmat <= prop_neutral_trials) = 0; % neutral trials
        R.cue_validity{k}(randmat > prop_neutral_trials + (1 - prop_neutral_trials) * cue_validity) = -1; %invalid trials
        
        R.cue{k} = R.probe{k};
        
        R.cue{k}(R.cue_validity{k} == 0) = 0;
        
        invalid_idx = find(R.cue_validity{k} == -1)';

        if nStimuli == 2
            R.cue{k}(invalid_idx) = 3 - R.cue{k}(invalid_idx); % flip 2 to 1, and 1 to 2
        elseif nStimuli > 2
            for i = 1:numel(invalid_idx)
                idx = invalid_idx(i);
                % for each flipped trial, pick the cue randomly from the other stimulus numbers
                R.cue{k}(idx) = randsample(setdiff(1:nStimuli, R.cue{k}(idx)), 1);
            end
        end        
    end
end