function retval = f_ext(y,s,sigma,sq_flag)
retval              = zeros(size(s)); % length of all trials
if sq_flag
    idx           = find(y>0);      % find all trials where y is greater than 0. y is either positive or imaginary. so a non-positive y would indicate negative a or b
    s                   = s(idx);
    sigma               = sigma(idx);
    y                   = y(idx);
else
    idx = true(size(s));
end
retval(idx)   = 0.5 * (erf((s+y)./(sigma*sqrt(2))) - erf((s-y)./(sigma*sqrt(2)))); % erf is faster than normcdf.
end
