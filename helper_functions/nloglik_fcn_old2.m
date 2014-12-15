function nloglik = nloglik_fcn(p_in, raw, model, varargin)

% % this is now going to break for non-fmincon becaues i took out the constraints
% if length(varargin) == 2 | length(varargin) == 3
%     alg = varargin{2};
%     if strcmp(alg,'snobfit') | strcmp(alg,'mcs') % opt algorithms that don't have linear constraints built in
%         c = varargin{1};
%         p_in = reshape(p_in,length(p_in),1);
%         if any(c.A * p_in > c.b) || any(p_in < c.lb') || any(p_in > c.ub')
%             nloglik = 1e8;
%             %disp('violation!')
%             return
%         end
%     end
% end

% I don't like strfind as much as regexp, but it's much faster.

global p conf_levels
    

p = parameter_variable_namer(p_in, model.parameter_names, model);

if length(varargin) == 1 %| length(varargin) == 3
    randn_samples = varargin{end};
        nDNoiseSets = size(randn_samples, 1);
    d_noise = p.sigma_d * randn_samples;
else
    nDNoiseSets = 1;
    d_noise = 0;
end


if isfield(p,'b_i')
    conf_levels = (length(p.b_i) - 1)/2;
else
    conf_levels = 0;
end

contrasts=exp(-4:.5:-1.5);

sig1    = 3;
sig2    = 12;
sig = sqrt(max(0,p.sigma_0^2 + p.alpha .* raw.contrast .^ -p.beta)); %  the max is there because tomlab fmincon appears to be a little naughty and likes to go below the lb.

nSamples = length(raw.s);

if strcmp(model.family, 'opt') % for optimal family of models
    k1      = .5 * log( (sig.^2 + sig2^2) ./ (sig.^2 + sig1^2)) + p.b_i(5); %log(prior / (1 - prior));
    k2      = (sig2^2 - sig1^2) ./ (2 .* (sig.^2 + sig1^2) .* (sig.^2 + sig2^2));
    %p_choice = -raw.Chat .* f(sqrt(k1 ./ k2), raw.s, sig) + (raw.Chat/2) + .5; % this was incorrectly not repmatted. previous estimates with d noise might have been off.
    f = @(y,s,sigma) f(y,s,sigma);
    save nloldtest.mat
p_choice = -repmat(raw.Chat, nDNoiseSets, 1) .* f(repmat(real(sqrt(k1 ./ k2)),nDNoiseSets,1), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)) + repmat(raw.Chat, nDNoiseSets, 1)/2 + .5;
            p_choice = mean(p_choice, 1);

    if model.choice_only
        a   = (repmat(k1, nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
        p_choice = -repmat(raw.Chat, nDNoiseSets, 1) .* f(sqrt(a), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)) + repmat(raw.Chat, nDNoiseSets, 1)/2 + .5; % this is a misnomer in this model
        p_choice = mean(p_choice, 1);
        
    else % conf models
        if ~model.symmetric % for opt asymmetric conf
            a = (repmat(k1 - bf((raw.Chat + 1)./2 - raw.Chat .* raw.g), nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
            b = (repmat(k1 - bf((raw.Chat - 1)./2 - raw.Chat .* raw.g), nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
            p_conf_choice = f(sqrt(b), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)) - f(sqrt(a), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1));
            p_conf_choice = mean(p_conf_choice, 1);
            
        else % opt sym conf
            a = (repmat(k1 + raw.Chat .* p.b_i(raw.g + 1), nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
            b = (repmat(k1 + raw.Chat .* p.b_i(raw.g    ), nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
            
            p_conf_choice = repmat(raw.Chat, nDNoiseSets, 1) .* (f(sqrt(a), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)) - f(sqrt(b), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)));
            p_conf_choice = mean(p_conf_choice, 1);
        end
    end
    
else % for non-bayesian models
    
    if strcmp(model.family, 'lin')
        k = max(bf(0) + mf(0) * sig, 0); % this is right.
    elseif strcmp(model.family, 'quad')
        k = max(bf(0) + mf(0) * sig.^2, 0);
%     elseif strcmp(model.family, 'quad2') % ????? this isn't programmed
%         k = max(bf(0) + mf(0) * sig.^2 + af(0) * sig, 0);
%     elseif strfind(model, 'lin')
%         k_multiplier = 1+sig/p.sigma_p;
%     elseif strfind(model, 'quad')
%         k_multiplier = 1+sig.^2/p.sigma_p^2;
    elseif strcmp(model.family, 'fixed')
        k_multiplier = 1;
    end
    
    %if ~any(regexp(model, '^(lin|quad)[0-9]')) % if not lin2 or quad2 or quad3
    if ~strcmp(model.family, 'lin') && ~strcmp(model.family, 'quad')
        k = k_multiplier * bf(0);
    end

    p_choice = -raw.Chat .* .5 .* (erf((raw.s + k)./(sig*sqrt(2))) - erf((raw.s - k)./(sig*sqrt(2)))) + (raw.Chat/2) + .5;
    
    if ~model.choice_only
        if ~isfield(raw, 'g')
            error('You are trying to fit confidence responses in a dataset that has no confidence trials.')
        end
        if strcmp(model.family, 'lin')
            term1 = raw.Chat .* max(bf(raw.Chat .* (raw.g    )) + sig .* mf(raw.Chat .* (raw.g    )), 0);
            term2 = raw.Chat .* max(bf(raw.Chat .* (raw.g - 1)) + sig .* mf(raw.Chat .* (raw.g - 1)), 0);
        elseif strcmp(model.family, 'quad')
            term1 = raw.Chat .* max(bf(raw.Chat .* (raw.g    )) + sig.^2 .* mf(raw.Chat .* (raw.g    )), 0);
            term2 = raw.Chat .* max(bf(raw.Chat .* (raw.g - 1)) + sig.^2 .* mf(raw.Chat .* (raw.g - 1)), 0);
%         elseif strfind(model.family, 'quad3')
%             term1 = raw.Chat .* max(bf(raw.Chat .* (raw.g    )) + sig.^2 .* mf(raw.Chat .* (raw.g    )) + sig .* af(raw.Chat .* (raw.g    )), 0);
%             term2 = raw.Chat .* max(bf(raw.Chat .* (raw.g - 1)) + sig.^2 .* mf(raw.Chat .* (raw.g - 1)) + sig .* af(raw.Chat .* (raw.g - 1)), 0);
        else
            term1 = raw.Chat .* bf(raw.Chat .* (raw.g)    ) .* k_multiplier;
            term2 = raw.Chat .* bf(raw.Chat .* (raw.g - 1)) .* k_multiplier;
        end
        %p_conf_choice      = normcdf(term1, raw.s, sig) - normcdf(term2, raw.s, sig) + normcdf(-term2, raw.s, sig) - normcdf(-term1, raw.s, sig);
        p_conf_choice      = .5 * (erf((raw.s + term1)./(sig*sqrt(2))) - erf((raw.s - term1)./(sig*sqrt(2))) + erf((raw.s - term2)./(sig*sqrt(2))) - erf((raw.s + term2)./(sig*sqrt(2))));
        p_conf_choice = max(p_conf_choice, 0);
    end
end


if isfield(p, 'lambda_i')
    p_full_lapse = p.lambda_i(raw.g)/2;
    p.lambda = sum(p.lambda_i);
else
    if ~isfield(p, 'lambda') % this is only for a few d_noise models that are probably deprecated
        p.lambda=0;
    end
    p_full_lapse = p.lambda/8;
end

if ~isfield(p, 'lambda_g')
    p.lambda_g = 0;
end

if ~model.choice_only
    p_repeat = [0 diff(raw.resp)==0];
else % choice models with lambda_r are not made yet.
    p_repeat = [0 diff(raw.Chat)==0];
end

if ~isfield(p, 'lambda_r')
    p.lambda_r = 0;
end
    

if ~model.choice_only
    save nloldtest2.mat
    loglik_vec = log (p_full_lapse + ...
        (p.lambda_g / 4) * p_choice + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_g - p.lambda_r) * p_conf_choice);
    
else % choice models
    loglik_vec = log(p.lambda / 2 + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_r) * p_choice);
    
end

% Set all -Inf logliks to an arbitrarily small number. It looks like these
% trials are all ones in which abs(s) was very large, and the subject
% didn't respond with full confidence. This occurs about .3% of trials.
% Shouldn't happen if we have lapse rate.
loglik_vec(loglik_vec < -1e5) = -1e5;
nloglik = - sum(loglik_vec);
end

function retval = f(y,s,sigma)
retval              = zeros(size(y)); % length of all trials
pos_y_idx           = find(y>0);      % find all trials where y is greater than 0. y is either positive or imaginary. so a non-positive y would indicate negative a or b
s                   = s(pos_y_idx);
sigma               = sigma(pos_y_idx);
y                   = y(pos_y_idx);
%retval(pos_y_idx)   = normcdf(y,s,sigma)-normcdf(-y,s,sigma);
retval(pos_y_idx)   = .5 * (erf((s+y)./(sigma*sqrt(2))) - erf((s-y)./(sigma*sqrt(2)))); % erf is faster than normcdf.
end

function bval = bf(name)
global p conf_levels
bval = p.b_i(name + repmat(conf_levels + 1, 1, length(name)));
end

function mval = mf(name)
global p conf_levels
mval = p.m_i(name + repmat(conf_levels + 1, 1, length(name)));
end

function aval = af(name)
global p conf_levels
aval = p.a_i(name + repmat(conf_levels + 1, 1, length(name)));
end