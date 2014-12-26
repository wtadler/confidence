function raw = trial_generator(p_in, model, varargin)

% define defaults
n_samples = 3240;
dist_type = 'same_mean_diff_std'; % 'same_mean_diff_std' (Qamar) or 'diff_mean_same_std' or 'sym_uniform' or 'half_gaussian' (Kepecs)
sig_s = 1; % for 'diff_mean_same_std' and 'half_gaussian'
a = 0; % overlap for sym_uniform
mu_1 = -5; % mean for 'diff_mean_same_std'
mu_2 = 5;
uniform_range = 1;
contrasts  = exp(-4:.5:-1.5);%[.125 .25 .5 1 2 4];
model_fitting_data = [];
conf_levels = 4;
assignopts(who,varargin);

nContrasts = length(contrasts);

p = parameter_variable_namer(p_in, model.parameter_names, model);

if model.free_cats
    sig1 = p.sig1;
    sig2 = p.sig2;
else
    sig1 = 3; % defaults for qamar distributions
    sig2 = 12;
end

category_params.sigma_1 = sig1;
category_params.sigma_2 = sig2;
category_params.overlap = a;
category_params.uniform_range = uniform_range;
category_params.sigma_s = sig_s;
category_params.mu_1 = mu_1;
category_params.mu_2 = mu_2;

[raw.C, raw.s, raw.sig, raw.Chat] = deal(zeros(1, n_samples));
if ~model.choice_only
    raw.g = zeros(1, n_samples);
end

if isempty(model_fitting_data)
    raw.C         = randsample([-1 1], n_samples, 'true');
    raw.contrast  = randsample(contrasts, n_samples, 'true'); % if no p, contrasts == sig
else % take real data
    raw.C           = model_fitting_data.C; % 2C-3 converts [1,2] to [-1,1]
    raw.contrast    = model_fitting_data.contrast;
    raw.s           = model_fitting_data.s;
end

[raw.contrast_values, raw.contrast_id] = unique_contrasts(raw.contrast);
sigs = sqrt(p.sigma_0^2 + p.alpha .* raw.contrast_values .^ - p.beta);
raw.sig = sqrt(p.sigma_0^2 + p.alpha .* raw.contrast .^ - p.beta);
raw.sig = reshape(raw.sig,1,length(raw.sig)); % think this should be okay.

if isempty(model_fitting_data)
    raw.s(raw.C == -1) = stimulus_orientations(category_params, dist_type, 1, 1, sum(raw.C ==-1));
    raw.s(raw.C ==  1) = stimulus_orientations(category_params, dist_type, 2, 1, sum(raw.C == 1));
end

raw.x = raw.s + randn(size(raw.sig)) .* raw.sig; % add noise to s. this line is the same in both tasks

switch dist_type
    case 'same_mean_diff_std'
        if strcmp(model.family,'opt')
            if model.non_overlap
                raw.d = zeros(1, n_samples);
                for c = 1 : nContrasts; % for each sigma level, generate d from the separate function
                    cursig = sqrt(p.sigma_0^2 + p.alpha .* contrasts(c) .^ - p.beta);
                    s=trun_sigstruct(cursig,sig1,sig2);
                    raw.d(raw.sig==cursig) = trun_da(raw.x(raw.sig==cursig), s);
                end
            else
                raw.k1 = .5 * log( (raw.sig.^2 + sig2^2) ./ (raw.sig.^2 + sig1^2));% + p.b_i(5);
                raw.k2 = (sig2^2 - sig1^2) ./ (2 .* (raw.sig.^2 + sig1^2) .* (raw.sig.^2 + sig2^2));
                raw.d = raw.k1 - raw.k2 .* raw.x.^2;
            end
            raw.posterior = 1 ./ (1 + exp(-raw.d));
        end
        
    case 'half_gaussian'        
        mu = (raw.x.* sig_s^2)./(raw.sig.^2 + sig_s^2);
        k = raw.sig .* sig_s ./ sqrt(raw.sig.^2 + sig_s^2);
        raw.d = log(normcdf(0,mu,k)./normcdf(0,-mu,k));
        
    case 'sym_uniform'
        denom = raw.sig * sqrt(2);
        raw.d = log( (erf((raw.x-a)./denom) - erf((raw.x+1-a)./denom)) ./ (erf((raw.x-1+a)./denom) - erf((raw.x+a)./denom)));
        
    case 'diff_mean_same_std'
        % work out decision variable.
        
    otherwise
        error('DIST_TYPE is not valid.')
        
end

confidences = [linspace(conf_levels,1,conf_levels) linspace(1,conf_levels,conf_levels)];

if strcmp(model.family,'opt') % for all opt family models
    if model.d_noise% add D noise
        raw.d = raw.d + p.sigma_d * randn(size(raw.d));
    end
    raw.d(raw.d==Inf)  =  1e6;
    raw.d(raw.d==-Inf) = -1e6;

    raw.Chat(raw.d >= p.b_i(5)) = -1;
    raw.Chat(raw.d < p.b_i(5)) = 1;

        for g = 1 : conf_levels * 2;
            raw.g( p.b_i(g) <= raw.d ...
                 & raw.d    <= p.b_i(g+1)) = confidences(g);
        end
        
elseif strcmp(model.family, 'MAP')        
    raw.shat = zeros(1,3240);
    for i = 1:nContrasts
        sig = sigs(i);
        idx = find(raw.contrast_id==i);
        
        k1 = sqrt(1/(sig^-2 + sig1^-2));
        mu1 = raw.x(idx)'*sig^-2 * k1^2;
        k2 = sqrt(1/(sig^-2 + sig2^-2));
        mu2 = raw.x(idx)'*sig^-2 * k2^2;
        
        raw.shat(idx) = gmm1max_n2_fast([normpdf(raw.x(idx),0,sqrt(sig1^2 + sig^2))' normpdf(raw.x(idx),0,sqrt(sig2^2 + sig^2))'],...
            [mu1 mu2], repmat([k1 k2],length(idx),1));
    end
    
    b = p.b_i(5);
    raw.Chat(abs(raw.shat) <= b) = -1;
    raw.Chat(abs(raw.shat) >  b) =  1;
    
    if ~model.choice_only
        for g = 1 : conf_levels * 2
            raw.g( p.b_i(g)   <  abs(raw.shat) ...
                &  p.b_i(g+1) >= abs(raw.shat)) = confidences(g);
        end
    end
    
else % all measurement models    
    if strcmp(model.family, 'lin')
        b = p.b_i(5) + p.m_i(5) * raw.sig;
    elseif strcmp(model.family, 'quad')
        b = p.b_i(5) + p.m_i(5) * raw.sig.^2;
    else
        b = p.b_i(5);
    end
    raw.Chat(abs(raw.x) <= b)   = -1;
    raw.Chat(abs(raw.x) >  b)   =  1;
    
    if ~model.choice_only % all non-optimal confidence models
        for g = 1 : conf_levels * 2
            if strcmp(model.family, 'lin')
                raw.g( p.b_i(g) + p.m_i(g) * raw.sig < abs(raw.x) ...
                    &  p.b_i(g+1) + p.m_i(g+1) * raw.sig >= abs(raw.x)) = confidences(g);
            elseif strcmp(model.family, 'quad')
                raw.g( p.b_i(g) + p.m_i(g) * raw.sig.^2 < abs(raw.x) ...
                    &  p.b_i(g+1) + p.m_i(g+1) * raw.sig.^2 >= abs(raw.x)) = confidences(g);
            else
                raw.g( p.b_i(g)   <  abs(raw.x) ...
                    &  p.b_i(g+1) >= abs(raw.x)) = confidences(g);
            end
        end
    end
end


% LAPSE TRIALS %%%%%%%%%%%%%%
randvals = rand(1, n_samples);

if model.multi_lapse
    cuml=[0 cumsum(p.lambda_i)]; % cumulative confidence lapse rate
    Chat_lapse_rate = cuml(end); 
    
    for l = 1 : conf_levels
        lapse_trials = randvals > cuml(l)...
                     & randvals < cuml(l+1);
        raw.g(lapse_trials) = l;
    end
    
else % models with full lapse
    Chat_lapse_rate = p.lambda;
end

Chat_lapse_trials = randvals < Chat_lapse_rate; % lapse Chat at each conf level
n_Chat_lapse_trials = sum(Chat_lapse_trials);
raw.Chat(Chat_lapse_trials) = randsample([-1 1], n_Chat_lapse_trials, 'true');
if ~model.choice_only && ~model.multi_lapse
    raw.g(Chat_lapse_trials) = randsample(conf_levels, n_lapse_trials, 'true');
end

if model.partial_lapse
    partial_lapse_rate = p.lambda_g;
    partial_lapse_trials = randvals > Chat_lapse_rate...
                         & randvals < Chat_lapse_rate + p.lambda_g;
    n_partial_lapse_trials = sum(partial_lapse_trials);
    raw.g(partial_lapse_trials) = randsample(conf_levels, n_partial_lapse_trials, 'true');
else
    partial_lapse_rate = 0;
end

if model.repeat_lapse
    repeat_lapse_rate = p.lambda_r;
    repeat_lapse_trials = find(randvals > Chat_lapse_rate + partial_lapse_rate...
                             & randvals < Chat_lapse_rate + partial_lapse_rate + repeat_lapse_rate);
    raw.g(repeat_lapse_trials) = raw.g(max(1,repeat_lapse_trials-1)); % max(1,etc) is to avoid issues when the first trial is chosen to be a repeat lapse (impossible)
    raw.Chat(repeat_lapse_trials) = raw.Chat(max(1,repeat_lapse_trials-1));
else
    repeat_lapse_rate = 0; % this is in case we come up with another kind of lapsing.
end


if ~model.choice_only
    raw.resp  = raw.g + conf_levels + ... % combine conf and class to give resp on 8 point scale
        (raw.Chat * .5 -.5) .* (2 * raw.g - 1);
end

raw.tf = raw.Chat == raw.C;