function R = setup_exp_order(n, category_params, varargin)
%%%Set up order of class, sigma, and orientation for entire scheme%%%

nStimuli = 1;
cue_validity = .8;
priors = [.5; 1];
assignopts(who, varargin);

nPriors = size(priors, 2);
R = [];
prop_neutral_trials = 1/6;


for k = 1:n.blocks
    R.category_type = category_params.category_type;
    
    [~, R.prior{k}] = histc(rand(1, n.trials*n.sections), [0, cumsum(priors(2,:))]); % sample prior according to specified distribution
    R.prior{k} = priors(1, R.prior{k}); % assign prior value
    R.trial_order{k} = zeros(1, n.trials*n.sections);
    for p = 1:nPriors
        prior = priors(1, p);
        prior_trials = R.prior{k} == prior;
        n_prior_trials = sum(prior_trials);
        [~, R.trial_order{k}(prior_trials)] = histc(rand(1, n_prior_trials), [0, prior, 1]); % sample category according to prior
    end
    
    R.prior{k} = reshape(R.prior{k}, n.sections, n.trials);
    R.trial_order{k} = reshape(R.trial_order{k}, n.sections, n.trials);
    
    stim_per_block = n.trials * n.sections * nStimuli;
    
    R.trial_order{k} = reshape(randsample(2, stim_per_block, true),...
        n.sections, n.trials, nStimuli);
    R.sigma{k} =       reshape(randsample(category_params.test_sigmas, stim_per_block, true),...
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