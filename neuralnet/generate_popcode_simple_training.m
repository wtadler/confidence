function [R, P, s, C, gains, sigmas] = generate_popcode_simple_training(ndatapergain, nneuron, sig1_sq, sig2_sq, sigtc_sq, sigma, baseline)

sigmas = sigma*ones(ndatapergain, 1);
gains = 100./(15.3524*sigmas.^2);

sprefs = linspace(-40,40,nneuron);

s  = [sqrt(sig1_sq) * randn(ndatapergain/2,1); sqrt(sig2_sq) * randn(ndatapergain/2,1)];
R  = repmat(gains,1,nneuron) .* exp(-(repmat(s,1,nneuron) - repmat(sprefs,ndatapergain,1)).^2 / (2*sigtc_sq)) + baseline;
R  = poissrnd(R); 
AR1 = sum(R,2) / sigtc_sq;
BR1 = sum(R.*repmat(sprefs,ndatapergain,1),2) / sigtc_sq;
P  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));

C = [ones(ndatapergain/2,1); zeros(ndatapergain/2,1)];