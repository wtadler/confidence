function unique_sigs = params2sig(p,varargin);

% this functionality should be in parameter_variable_namer now? 6/2/15
contrasts = exp(linspace(-5.5,-2,6));
assignopts(who,varargin);
p

if isfield(p,'sigma_c_low') % new style
    c_low = min(contrasts);
    c_hi = max(contrasts);
    alpha = (p.sigma_c_low^2-p.sigma_c_hi^2)/(c_low^-p.beta - c_hi^-p.beta);
    unique_sigs = fliplr(sqrt(p.sigma_c_low^2 - alpha * c_low^-p.beta + alpha*contrasts.^-p.beta)); % low to high sigma. should line up with contrast id
    
elseif isfield(p,'sigma_0') % old style
    unique_sigs = fliplr(sqrt(p.sigma_0 + p.alpha * contrasts.^-p.beta));
end
end