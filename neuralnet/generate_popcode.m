function [R, P, D, gains, sigmas] = generate_popcode(C, s, sigmas, varargin);

nneuron = 50;
sig1_sq = 3^2;
sig2_sq = 12^2;
tc_precision = .01;
baseline = 0;
assignopts(who, varargin);

sprefs = linspace(-40, 40, nneuron);
assignopts(who, varargin);

K = sum(exp(-sprefs.^2 * tc_precision / 2));

if ~(all(size(sigmas) == size(C)) && all(size(sigmas) == size(s)))
    error('sizes of C, s, and sigmas need to match.')
end
nTrials = length(C);

vars = sigmas.^2;
gains = 1 ./ (tc_precision*vars*K);

R = repmat(gains,1,nneuron) .* exp(-(repmat(s,1,nneuron) - repmat(sprefs,nTrials,1)).^2 * tc_precision / 2) + baseline;
R  = poissrnd(R); 

AR1 = sum(R,2) * tc_precision;
BR1 = sum(R.*repmat(sprefs,nTrials,1),2) * tc_precision;

P  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));

D = -log(1./P - 1);

%% test

load('~/Google Drive/Will - Confidence/Analysis/neuralnet/test_sigmas.mat')

nn_dataset(1e4, .001, .01, 1, test_sigmas(:, end));


