function [R,P,S,C,K] = generate_popcode_noisy_data_allgains_6(ndatapergain, nneuron, sig1_sq, sig2_sq, sigtc_sq)

sprefs    = linspace(-40,40,nneuron);
contrasts = [.018 .03 .05 .082 0.135 .223]';
K         = repmat(contrasts,ndatapergain/6,1);

% noise params
alpha = 9.9828;
beta  = 2.3454;
gamma = 4.2008;   

gains  = (alpha.*K).^(-beta) + gamma;
gains  = 100./(gains.*15.3524);

S1  = [sqrt(sig1_sq) * randn(ndatapergain/2,1); sqrt(sig2_sq) * randn(ndatapergain/2,1)];
R1  = repmat(gains,1,nneuron) .* exp(-(repmat(S1,1,nneuron) - repmat(sprefs,ndatapergain,1)).^2 / (2*sigtc_sq));
R1  = poissrnd(R1); 
AR1 = sum(R1,2) / sigtc_sq;
BR1 = sum(R1.*repmat(sprefs,ndatapergain,1),2) / sigtc_sq;
P1  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));

C = [ones(ndatapergain/2,1); zeros(ndatapergain/2,1)];
R = R1;
S = S1;
P = P1;
