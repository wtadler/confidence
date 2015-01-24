function lp = logprior_fcn(x,o)
lapse_params = strncmpi(o.parameter_names,'lambda',6);
uniform_range = o.ub(~lapse_params)-o.lb(~lapse_params);
a=1; % beta dist params
b=20;
lp=sum(log(1./uniform_range))+sum(log(betapdf(x(lapse_params),a,b)));