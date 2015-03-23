function R = setup_exp_order(n, category_params, varargin)
%%%Set up order of class, sigma, and orientation for entire scheme%%%

R = [];

if isempty(varargin)
    attention_manipulation = false;
elseif length(varargin)==1
    attention_manipulation = true;
    cue_validity = varargin{1};
end


for k = 1:n.blocks
    R.category_type = category_params.category_type;
    R.trial_order{k} = reshape(randsample(2,n.trials*n.sections,true),n.sections,n.trials);
    R.sigma{k} = reshape(randsample(category_params.test_sigmas, n.trials*n.sections,true),n.sections,n.trials);
    
    %get random orientation draws from normal distributions (~sigmal or ~sigma2)
    %CORRESPONDING TO order of classes.
    classtwos = (R.trial_order{k} == 2); % index of class 2 trials
    R.phase{k} = 360*rand(n.sections,n.trials); % random phase draws
    
    R.draws{k}= reshape(stimulus_orientations(category_params, 1, n.sections*n.trials), n.sections, n.trials);
    R.draws{k}(classtwos) = stimulus_orientations(category_params, 2, nnz(classtwos));
    
    if attention_manipulation
        R.probe{k} = reshape(randsample(2,n.trials*n.sections,true), n.sections, n.trials);
        
        R.cue{k} = R.probe{k};
        flip_idx = rand(size(R.probe{k}))>cue_validity; % find trials where cue is invalid. need to flip those cues.
        R.cue{k}(flip_idx) = -R.cue{k}(flip_idx) + 3; % y = -x + 3 flips 2 trials to 1, and 1 trials to 2.       
    end
end