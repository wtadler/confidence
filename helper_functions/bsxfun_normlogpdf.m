function y = bsxfun_normlogpdf(x,mu,sigma)
%BSXFUN_NORMLOGPDF Vectorized normal log probability density func (log pdf).
%   Y = BSXFUN_NORMLOGPDF(X,MU,SIGMA) returns the log pdf of the normal 
%   distribution with mean MU and standard deviation SIGMA, evaluated at 
%   the values in X. Dimensions of X, MU, and SIGMA must either match, or 
%   be equal to one. Computation of the pdf is performed with singleton
%   expansion enabled via BSXFUN. The size of Y is the size of the input 
%   arguments (expanded to non-singleton dimensions).
%
%   All elements of SIGMA are assumed to be non-negative (no checks).
%
%   See also BSXFUN, NORMPDF.

%   Author: Luigi Acerbi
%   Release date: 15/06/2015

if nargin<3
    error('bmp:bsxfun_normlogpdf:TooFewInputs','Input argument X, MU or SIGMA are undefined.');
end

try
    nf = -0.5*log(2*pi);    
    if isscalar(mu)
        y = bsxfun(@minus, -0.5*bsxfun(@rdivide, x - mu, sigma).^2, log(sigma)) + nf;
    elseif isscalar(sigma)
        y = -0.5*(bsxfun(@minus, x, mu)/sigma).^2 - log(sigma) + nf;
    else
        y = bsxfun(@minus, -0.5*bsxfun(@rdivide, bsxfun(@minus, x, mu), sigma).^2, log(sigma)) + nf;
    end
catch
    error('bmp:bsxfun_normlogpdf:InputSizeMismatch',...
          'Non-singleton dimensions must match in size.');
end
