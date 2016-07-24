function [R, P, s, C, gains, sigmas] = generate_popcode_simple_training(nTrials, nneuron, sig1_sq, sig2_sq, tc_precision, sigma, baseline, K, sprefs)

sigmas = sigma*ones(nTrials, 1);
vars = sigmas.^2;
gains  = 1 ./ (tc_precision*vars.*K);

s  = [sqrt(sig1_sq) * randn(nTrials/2,1); sqrt(sig2_sq) * randn(nTrials/2,1)];
R  = repmat(gains,1,nneuron) .* exp(-(repmat(s,1,nneuron) - repmat(sprefs,nTrials,1)).^2 * tc_precision / 2) + baseline;
R  = poissrnd(R); 
AR1 = sum(R,2) * tc_precision;
BR1 = sum(R.*repmat(sprefs,nTrials,1),2) * tc_precision;
P  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));

C = [ones(nTrials/2,1); zeros(nTrials/2,1)];