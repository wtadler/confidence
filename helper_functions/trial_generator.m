function raw = trial_generator(p_in, model, varargin)

% define defaults
n_samples = 1e4;
dist_type = 'qamar'; % or 'kepecs' (half-gaussian) or 'sym_uniform'
sig_s = 1; % sig_s is the width of the half-gaussian.
a = 0; % overlap for sym_uniform
contrasts  = [.125 .25 .5 1 2 4];
model_fitting_data = [];
conf_levels = 4;
assignopts(who,varargin);

p = parameter_variable_namer(p_in, model.parameter_names, model);

if model.free_cats
    sig1 = p.sig1;
    sig2 = p.sig2;
else
    sig1 = 3; % defaults for qamar distributions
    sig2 = 12;
end

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
raw.sig = sqrt(p.sigma_0^2 + p.alpha .* raw.contrast .^ - p.beta);
%raw.sig = sig_c(raw.contrast_id);
raw.sig = reshape(raw.sig,1,length(raw.sig)); % think this should be okay.

switch dist_type
    case 'qamar'
        if isempty(model_fitting_data)
            raw.s(raw.C == -1) = randn(1,sum(raw.C ==-1))*sig1; % Generate s trials for Cat1 and Cat2
            raw.s(raw.C ==  1) = randn(1,sum(raw.C == 1))*sig2;
        end
        raw.x = raw.s + randn(size(raw.sig)) .* raw.sig; % add noise to s. this line is the same in both tasks
        if strcmp(model.family,'opt')
            
            if model.non_overlap
                raw.d = zeros(1, n_samples);
                for c = 1 : length(contrasts); % for each sigma level, generate d from the separate function
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
    case 'kepecs'
        if isempty(model_fitting_data)
            raw.s(raw.C == -1)  = -abs(normrnd(0, sig_s, 1, sum(raw.C == -1)));
            raw.s(raw.C == 1)   =  abs(normrnd(0, sig_s, 1, sum(raw.C ==  1)));
        end
        
        raw.x               = raw.s + randn(size(raw.s)) .* raw.sig;
        
        mu = (raw.x.* sig_s^2)./(raw.sig.^2 + sig_s^2);
        k = raw.sig .* sig_s ./ sqrt(raw.sig.^2 + sig_s^2);
        raw.d = log(normcdf(0,mu,k)./normcdf(0,-mu,k));
        
    case 'sym_uniform'
        % symmetric with overlap a
        if isempty(model_fitting_data)
            raw.s(raw.C == -1)    = rand(1, sum(raw.C == -1)) - 1 + a; % Generate s trials from shifted uniforms
            raw.s(raw.C ==  1)    = rand(1, sum(raw.C ==  1)) - a;
        end
        
        raw.x                 = raw.s + randn(size(raw.s)) .* raw.sig;
        
        denom = raw.sig * sqrt(2);
        raw.d = log( (erf((raw.x-a)./denom) - erf((raw.x+1-a)./denom)) ./ (erf((raw.x-1+a)./denom) - erf((raw.x+a)./denom)));
        
    otherwise
        error('DIST_TYPE is not valid.')
        
end

if strcmp(model.family,'opt') % for all opt family models
    if model.d_noise% add D noise
        raw.d = raw.d + p.sigma_d * randn(size(raw.d));
    end
    raw.d(raw.d==Inf)  =  1e6;
    raw.d(raw.d==-Inf) = -1e6;

    raw.Chat(raw.d >= p.b_i(5)) = -1;
    raw.Chat(raw.d < p.b_i(5)) = 1;

    %if ~model.symmetric % merge this with fixed. it's the same.
%         raw.Chat(raw.d >= p.b_i(5)) = -1;
%         raw.Chat(raw.d < p.b_i(5)) = 1;

        confidences = [linspace(conf_levels,1,conf_levels) linspace(1,conf_levels,conf_levels)];
        for g = 1 : conf_levels * 2;
            raw.g( p.b_i(g) <= raw.d ...
                 & raw.d    <= p.b_i(g+1)) = confidences(g);
        end
        
    %else
        
%         raw.Chat(raw.d >= 0) = -1;
%         raw.Chat(raw.d <  0) =  1;
        
%         %put limit on d. this is an issue with the non-overlapping categories, and the below for loop.
%         if ~model.choice_only % symmetric opt confidence models
%             for g = 1 : conf_levels
%                 raw.g( p.b_i(g)   <=  -raw.Chat .* raw.d ...
%                     &  p.b_i(g+1) > -raw.Chat .* raw.d) = g; % see conf data likelihood.pages to see proof
%             end
%         end
%     end
elseif strcmp(model.family, 'map')
    prior=(1/(2*sqrt(2*pi)))*((1/sig1)*exp((-x.^2)/(2*sig1^2)) + (1/sig2)*exp((-x.^2)/(2*sig2^2))); % re-normalized sum of the two gaussians
    
    
else % all measurement models
%     if any(regexp(model, '^lin(?!2)')) % lin but not lin2
%         k_multiplier = 1+raw.sig/p.sigma_p;
%         
%     elseif any(regexp(model, '^quad(?!2)'))
%         k_multiplier = 1+raw.sig.^2/p.sigma_p^2;
%         
%     elseif any(regexp(model, '^fixed'))
%         k_multiplier = 1;
%         
%     end
    
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
        
        confidences = [linspace(conf_levels,1,conf_levels) linspace(1,conf_levels,conf_levels)];
        
        for b = 1 : conf_levels * 2
            if strcmp(model.family, 'lin')
                raw.g( p.b_i(b) + p.m_i(b) * raw.sig < abs(raw.x) ...
                    &  p.b_i(b+1) + p.m_i(b+1) * raw.sig >= abs(raw.x)) = confidences(b);
            elseif strcmp(model.family, 'quad')
                raw.g( p.b_i(b) + p.m_i(b) * raw.sig.^2 < abs(raw.x) ...
                    &  p.b_i(b+1) + p.m_i(b+1) * raw.sig.^2 >= abs(raw.x)) = confidences(b);
            else
                raw.g( p.b_i(b)   <  abs(raw.x) ...
                    &  p.b_i(b+1) >= abs(raw.x)) = confidences(b);
            end
        end
    end
end

if isfield(p,'lambda_i')
    randvals = rand(1, n_samples);
    cuml=[0 cumsum(p.lambda_i)];
    Chat_lapse_trials = randvals < cuml(conf_levels + 1);
    n_Chat_lapse_trials = sum(Chat_lapse_trials);
    raw.Chat(Chat_lapse_trials) = randsample([-1 1], n_Chat_lapse_trials, 'true');
    for l = 1 : conf_levels
        lapse_trials = randvals > cuml(l) & randvals < cuml(l+1);
        raw.g(lapse_trials) = l;
    end
    if isfield(p,'lambda_g')
        partial_lapse_trials = randvals > cuml(conf_levels+1) & randvals < cuml(conf_levels+1) + p.lambda_g;
        n_partial_lapse_trials = sum(partial_lapse_trials);
        raw.g(partial_lapse_trials) = randsample(4, n_partial_lapse_trials, 'true');
        if isfield(p, 'lambda_r')
            repeat_lapse_trials = find(randvals > cuml(conf_levels+1) + p.lambda_g & randvals < cuml(conf_levels+1) + p.lambda_g + p.lambda_r);
            %n_repeat_lapse_trials = length(repeat_lapse_trials);
            raw.g(repeat_lapse_trials) = raw.g(max(1,repeat_lapse_trials-1)); % max(1,etc) is to avoid issues when the first trial is chosen to be a repeat lapse (impossible)
            raw.Chat(repeat_lapse_trials) = raw.Chat(max(1,repeat_lapse_trials-1));
        end
    end
% scramble some Chat and g trials, according to lapse rates p.lambda and p.lambda_g.
elseif isfield(p, 'lambda') % models with lapse
    randvals = rand(1, n_samples);
    lapse_trials            = randvals < p.lambda;
    n_lapse_trials = sum(lapse_trials);
    raw.Chat(lapse_trials)         = randsample([-1 1], n_lapse_trials, 'true');
    if isfield(p, 'lambda_g')
        partial_lapse_trials    = randvals > p.lambda & randvals < p.lambda + p.lambda_g;
        n_partial_lapse_trials  = sum(partial_lapse_trials);
        raw.g   (partial_lapse_trials) = randsample(1:conf_levels, n_partial_lapse_trials, 'true');
    end
end


if ~model.choice_only
    
    %raw.g   (lapse_trials)         = 1;%randsample(1:conf_levels, n_lapse_trials, 'true'); % lapse confidence, if confidence exists
    
    
    
    raw.resp  = raw.g + conf_levels + ... % combine conf and class to give resp on 8 point scale
        (raw.Chat * .5 -.5) .* (2 * raw.g - 1);
end

raw.tf = raw.Chat == raw.C;