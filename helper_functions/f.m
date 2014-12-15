function retval = f(y,s,sigma)
retval              = zeros(size(y)); % length of all trials
pos_y_idx           = find(y>0);      % find all trials where y is greater than 0. y is either positive or imaginary. so a non-positive y would indicate negative a or b
s                   = s(pos_y_idx);
sigma               = sigma(pos_y_idx);
y                   = y(pos_y_idx);
%retval(pos_y_idx)   = normcdf(y,s,sigma)-normcdf(-y,s,sigma);
retval(pos_y_idx)   = .5 * (erf((s+y)./(sigma*sqrt(2))) - erf((s-y)./(sigma*sqrt(2)))); % erf is faster than normcdf.
end