function y = my_normcdf(x, mu, sigma)

if any(size(x)==1)
%     y = 0.5 * (1 + erf((k-mu) ./ (sigma*sqrt(2))));
    y = 0.5 * erfc((mu-x) ./ (sigma*sqrt(2))); % erfc produces more accurate near-zero results for large negative x than erf. same speed
    
else
    y = bsxfun_normcdf(x, mu, sigma);
end