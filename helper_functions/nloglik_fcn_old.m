function nloglik = nloglik_fcn(p, raw, model, varargin)

if length(varargin) == 1
    randn_samples = varargin{1};
    nDNoiseSets = size(randn_samples, 1);
end


[alpha, beta, sigma_0, prior, b_i, lambda, lambda_g, sigma_d, sigma_p] = parameter_variable_namer(p,model);

contrasts=exp(-4:.5:-1.5);

sig1    = 3;
sig2    = 12;
sig = sqrt(sigma_0^2 + alpha .* raw.contrast .^ -beta);

nSamples = length(raw.s);

if regexp(model, '^opt') % for optimal family of models
    k1      = .5 * log( (sig.^2 + sig2^2) ./ (sig.^2 + sig1^2)) + log(prior / (1 - prior));
    k2      = (sig2^2 - sig1^2) ./ (2 .* (sig.^2 + sig1^2) .* (sig.^2 + sig2^2));
    
    if regexp(model, 'd_noise') % for optimal models with d noise
        
        d_noise = sigma_d * randn_samples;
        
        if regexp(model, 'conf') % do noise conf
            a       = (repmat(k1 + raw.Chat .* b_i(raw.g + 1), nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
            b       = (repmat(k1 + raw.Chat .* b_i(raw.g    ), nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
            
            p_no_lapse = repmat(raw.Chat, nDNoiseSets, 1) .* (f(sqrt(a), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)) - f(sqrt(b), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)));
            p_no_lapse = mean(p_no_lapse, 1);
            
        else % d noise choice
            a   = (repmat(k1, nDNoiseSets, 1) + d_noise) ./ repmat(k2, nDNoiseSets, 1);
            p_partial_lapse = -repmat(raw.Chat, nDNoiseSets, 1) .* f(sqrt(a), repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)) + repmat(raw.Chat, nDNoiseSets, 1)/2 + .5; % this is a misnomer in this model
            p_partial_lapse = mean(p_partial_lapse, 1);
            
        end
        
    else % for opt models without d noise
        
        p_partial_lapse = -raw.Chat .* f(sqrt(k1 ./ k2), raw.s, sig) + (raw.Chat/2) + .5;
        if regexp(model, 'asym')
            a       = (k1 - b_i(5 + (raw.Chat + 1)./2 - raw.Chat .* raw.g)) ./ k2;
            b       = (k1 - b_i(5 + (raw.Chat - 1)./2 - raw.Chat .* raw.g)) ./ k2;
            p_no_lapse = -(f(sqrt(a), raw.s, sig) - f(sqrt(b), raw.s, sig));
        elseif regexp(model, 'conf')
            a       = (k1 + raw.Chat .* b_i(raw.g + 1))   ./ k2;
            b       = (k1 + raw.Chat .* b_i(raw.g    ))   ./ k2;
            p_no_lapse = raw.Chat .* (f(sqrt(a), raw.s, sig) - f(sqrt(b), raw.s, sig));
            
        end
    end
    
else % for non-optimal models
    
    if any(regexp(model, '^lin'))
        k_multiplier = 1+sig/sigma_p;
        
    elseif any(regexp(model, '^quad'))
        k_multiplier = 1+sig.^2/sigma_p^2;
        
    elseif any(regexp(model, '^fixed'))
        k_multiplier = 1;
        
    end
    k = k_multiplier * b_i(5);
    p_partial_lapse = -raw.Chat .* .5 .* (erf((raw.s + k)./(sig*sqrt(2))) - erf((raw.s - k)./(sig*sqrt(2)))) + (raw.Chat/2) + .5;
    if regexp(model, 'conf')
        term1 = raw.Chat .* b_i(raw.Chat .* (raw.g)     + 5) .* k_multiplier;
        term2 = raw.Chat .* b_i(raw.Chat .* (raw.g - 1) + 5) .* k_multiplier;
        
        %p_no_lapse      = normcdf(term1, raw.s, sig) - normcdf(term2, raw.s, sig) + normcdf(-term2, raw.s, sig) - normcdf(-term1, raw.s, sig);
        p_no_lapse      = .5 * (erf((raw.s + term1)./(sig*sqrt(2))) - erf((raw.s - term1)./(sig*sqrt(2))) + erf((raw.s - term2)./(sig*sqrt(2))) - erf((raw.s + term2)./(sig*sqrt(2))));
        
    end
end



if regexp(model, '(?<!no_)partial_lapse') % for all models with partial_lapse. fancy regexp negative lookbehind looks for models with partial_lapse in the model, that isn't preceded by 'no_'
    loglik_vec = log ((lambda   / 8) + ...
        (lambda_g / 4) * p_partial_lapse + ...
        (1 - lambda - lambda_g) * p_no_lapse);
    
elseif regexp(model, '^opt.*d_noise$')
    loglik_vec = log(p_partial_lapse);
    
elseif ~any(strfind(model, 'conf')) % for all choice models (they don't have conf in the name)
    loglik_vec = log((lambda / 2) + ...
        (1 - lambda) * p_partial_lapse);
    
else % for all remaining confidence models. (that don't have partial lapse)
    loglik_vec = log ((lambda   / 8) + ...
        (1 - lambda) * p_no_lapse);
    
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