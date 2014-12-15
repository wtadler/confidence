function R = setup_exp_order(n, sigma, category_type, varargin)
%%%Set up order of class, sigma, and orientation for entire scheme%%%

R = [];
sigma.uniform_range = 15; % for sym_uniform
sigma.overlap = 0; % for sym_uniform
sigma.s = 5; % for half_gaussian
if length(varargin)==0
    attention_manipulation = 0;
elseif length(varargin)==1
    attention_manipulation = varargin{1};
elseif length(varargin)==2
    attention_manipulation = varargin{1};
    cue_validity = varargin{2};
end


for k = 1:n.blocks
%     for m = 1:n.sections
%         %Indexing: R.property{block}(section, trial)
%         
%         R.trial_order{k}(m,:) = randsample(2,n.trials,true); % full random classes
%         R.sigma{k}(m,:) = randsample(sigma.int,n.trials,true); % full random sigmas
%         
%     end
    

    R.trial_order{k} = reshape(randsample(2,n.trials*n.sections,true),n.sections,n.trials);
    R.sigma{k} = reshape(randsample(sigma.int,n.trials*n.sections,true),n.sections,n.trials);
    
    %get random orientation draws from normal distributions (~sigmal or ~sigma2)
    %CORRESPONDING TO order of classes.
    classtwos = (R.trial_order{k} == 2); % index of class 2 trials
    R.phase{k} = 360*rand(n.sections,n.trials); % random phase draws
    
    R.draws{k}= stimulus_orientations(sigma, category_type, n.sections, n.trials);
    R.draws{k}(classtwos) = stimulus_orientations(sigma, category_type, nnz(classtwos), 1);
    
    function s = stimulus_orientations(sigma, category_type
    switch category_type
        case 'same_mean_diff_std'
            R.draws{k} = sigma.one*randn(n.sections,n.trials); % get all draws from sigma.one
            R.draws{k}(classtwos) = sigma.two*randn(nnz(classtwos),1); % replace the class 2 draws with draws from sigma.two
    
        case 'diff_mean_same_std'
            R.draws{k} = sigma.mu1 + sigma.one*randn(n.sections, n.trials);
            R.draws{k}(classtwos) = sigma.mu2 + sigma.one*randn(nnz(classtwos),1);

        case 'sym_uniform'
            R.draws{k} = - uniform_range * rand(n.sections, n.trials) + overlap;
            R.draws{k}(classtwos) = uniform_range * rand(nnz(classtwos),1) - overlap;
    
        case 'half_gaussian'
            R.draws{k} = - abs(sig_s * randn(n.sections, n.trials));
            R.draws{k}(classtwos) = -R.draws{k}(classtwos);
    end
    
    if attention_manipulation
        R.probe{k} = reshape(randsample(2,n.trials*n.sections,true), n.sections, n.trials);
        
        R.cue{k} = R.probe{k};
        flip_idx = rand(size(R.probe{k}))>cue_validity; % find trials where cue is invalid. need to flip those cues.
        R.cue{k}(flip_idx) = -R.cue{k}(flip_idx) + 3; % y = -x + 3 flips 2 trials to 1, and 1 trials to 2.       
    end
end