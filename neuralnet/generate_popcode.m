function [R, P, D, gains, sigmas] = generate_popcode(C, s, sigmas, varargin);

sig1_sq = 3^2;
sig2_sq = 12^2;
tc_precision = .01;
baseline = 0;
sprefs = linspace(-40, 40, 50);
ds = .1;
assignopts(who, varargin);

nneuron = length(sprefs);

K = sum(exp(-sprefs.^2 * tc_precision / 2));

if ~(all(size(sigmas) == size(C)) && all(size(sigmas) == size(s)))
    error('sizes of C, s, and sigmas need to match.')
end
nTrials = length(C);

vars = sigmas.^2;
gains = 1 ./ (tc_precision*vars*K);

R = repmat(gains,1,nneuron) .* exp(-(repmat(s,1,nneuron) - repmat(sprefs,nTrials,1)).^2 * tc_precision / 2) + baseline;
R = poissrnd(R);

if nargout > 1
    if baseline==0
        AR1 = sum(R,2) * tc_precision;
        BR1 = sum(R.*repmat(sprefs,nTrials,1),2) * tc_precision;
        
        P  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));
        D = -log(1./P - 1);
    else
        s = (-40:ds:40)';
        
        tc = bsxfun(@times, permute(gains, [3 2 1]), exp(-bsxfun(@minus, s, sprefs).^2 * tc_precision / 2)) + baseline; % f_i(s) for every s and neuron i
        neural_likelihood = squeeze(prod(bsxfun(@power, tc, permute(R, [3 2 1])), 2));
        cat1_likelihood = sum(bsxfun(@times, normpdf(s, 0, 3), neural_likelihood))*ds;
        cat2_likelihood = sum(bsxfun(@times, normpdf(s, 0, 12), neural_likelihood))*ds;
        D = log(cat1_likelihood ./ cat2_likelihood)';
        P = 1 ./ (1 + exp(-D));
    end
end