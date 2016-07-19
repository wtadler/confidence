function [RMSEtrain, data, perf_train, perf_test] = nn_dataset(nTrainingTrials, eta_0, gamma_e, sigma_train, sigmas_test, varargin)

train_on_test_noise = true;
baseline = 0;
quantile_type = 'weak';
nEpochs      = 10;
alpha = [0 1e-4];
batch_size      = 5;
assignopts(who, varargin);

% if train_on_test_noise
    nTrainingTrials = round(nTrainingTrials/6)*6;
% end

% network parameters
nhu       = 200;
L         = 3;
nneuron   = 50;
nnode     = [nneuron nhu 1];
ftype     = 'relu';
objective = 'xent';

% training parameters
mu         = 0.0;
lambda_eff = 0.0;
% eta_0      = 0.05; % optimize this
% gamma_e    = 0.0001; % and this
eta        = eta_0 ./ (1 + gamma_e*(0:(nTrainingTrials-1))); % learning rate policy
% lambda     = 0.0705; % lapse

% generate data
sig1_sq     = 3^2;
sig2_sq     = 12^2;
tc_precision    = .01; % aka tau_t. formerly .01

% sum activity for unit gain. i'm sure this is a dumb way to do it. also,
% have a check to ensure that the space is uniformly covered with neural
% response
K = 0;
sprefs = linspace(-40, 40, nneuron);
for i = 1:nneuron
    K = K + exp(-(0-sprefs(i)).^2 * tc_precision / 2);
end

if ~train_on_test_noise
    [R, P, ~, C_train, ~] = generate_popcode_simple_training(nTrainingTrials, nneuron, sig1_sq, sig2_sq, tc_precision, sigma_train, baseline, K, sprefs);
else
    [R, P, ~, C_train, ~] = generate_popcode_noisy_data_allgains_6(nTrainingTrials, nneuron, sig1_sq, sig2_sq, tc_precision, sigmas_test, baseline, K, sprefs);
end
% fprintf('Generated training data\n');

Xdata      = R';
Ydata      = C_train';

% initialize network parameters
W_init = cell(L,1);
b_init = cell(L,1);

W_init{2} = 0.05*randn(nnode(2),nnode(1));
b_init{2} = 0.00*randn(nnode(2),1);

W_init{3} = 0.05*randn(nnode(3),nnode(2));
b_init{3} = 0.00*randn(nnode(3),1);

% Evaluate network at the end of epoch
nTestTrials = 2160; % should this be 2160 to match the experiment? or need more

perf_test = zeros(1, nEpochs);
perf_train = perf_test;

%% Train network with SGD
for e = 1:nEpochs
    
    pp = randperm(nTrainingTrials);
    
    for bi = 1:(nTrainingTrials/batch_size)
        
        bbegin = (bi-1)*batch_size+1;
        bend   = bi*batch_size;
        X      = Xdata(:,pp(bbegin:bend));
        Y      = Ydata(:,pp(bbegin:bend));
        
        if (e == 1) && (bi == 1)
            W = W_init;
            b = b_init;
        end
        
        [W, b] = do_backprop_on_batch(X, Y, W, b, eta(bi), mu, lambda_eff, L, ftype, 0, objective, alpha);
        
    end
    
    % Performance over training set
    Yhattrain           = zeros(1,nTrainingTrials);
    for ti = 1:nTrainingTrials
        [a, ~]          = fwd_pass(Xdata(:,ti),W,b,L,ftype);
        Yhattrain(1,ti) = a{end};
    end
    RMSEtrain = sqrt(mean((Yhattrain-P').^2)); % use this as objective
    perf_train(e) = mean((Yhattrain > .5) == C_train');
%     fprintf('\nEpoch %i: %.1f%% training performance', e, perf_train(e)*100)
    
    % Evaluate network at the end of epoch
    [Rinf, Pinf, s, C_test, gains, sigmas] = generate_popcode_noisy_data_allgains_6(nTestTrials, nneuron, sig1_sq, sig2_sq, tc_precision, sigmas_test, baseline, K, sprefs);
    Xinfloss                   = Rinf';
    Yinfloss                   = Pinf';
    Yhatinf                    = zeros(1,nTestTrials);
    for ti = 1:nTestTrials
        [a, ~]        = fwd_pass(Xinfloss(:,ti),W,b,L,ftype);
        Yhatinf(1,ti) = a{end};
    end
    
    Yinfloss(Yhatinf==0) = .5;
    Yhatinf(Yhatinf==0)  = .5;
    
    InfLoss = nanmean(Yinfloss.*(log(Yinfloss./Yhatinf)) + (1-Yinfloss).*(log((1-Yinfloss)./(1-Yhatinf)))) ...
        / nanmean(Yinfloss.*log(2*Yinfloss) + (1-Yinfloss).*log(2*(1-Yinfloss)));
    
    RMSE = sqrt(mean((Yhatinf-Yinfloss).^2));
    
    perf_test(e) = mean(C_test'==real(Yhatinf > .5));
%     fprintf('\nEpoch %i: %.1f%% test performance\n', e, perf_test(e)*100)
    
    %     fprintf('Epoch: %i done, InfLoss on test: %f, RMSE on test: %f, NoAcceptedTrials: %i, RMSE on training data: %f \n', e, InfLoss, RMSE, length(Yinfloss), RMSEtrain);
    
end

data.C = C_test';
data.C(data.C==1) = -1;
data.C(data.C==0) = 1;

data.gains = gains';
data.sigmas = sigmas';

data.s = s';

data.nTrainingTrials = nTrainingTrials;

data.prob = Yhatinf;

switch quantile_type
    case 'ultraweak'
        % quantile for confidence and choice (analogous to Bayes_Weak)
        edges = [0 quantile(data.prob, linspace(1/8, 7/8, 7)) 1];
    case 'weak'
        % quantile confidences but fix choice boundary at .5
        edges = [0, quantile(data.prob(data.prob<=.5), linspace(1/4, 3/4, 3)), 1/2, quantile(data.prob(data.prob>.5), linspace(1/4, 3/4, 3)), 1];
    case 'lowconfonly'
        edges = [zeros(1,4) 1/2 ones(1,4)];
end

for i = 2:length(edges)
    if isnan(edges(i))
        edges(i) = edges(i-1);
    end
end
[~, data.resp] = histc(data.prob, edges);



data.resp = 9-data.resp;
data.Chat = real(data.resp >= 5);
data.Chat(data.Chat==0) = -1;
data.tf = data.C==data.Chat;
fprintf('\n%.1f%% final test performance\n', perf_test(e)*100)

c1 = data.Chat == -1;
c2 = data.Chat == 1;
data.g = [];
data.g(c1) = 5 - data.resp(c1);
data.g(c2) = -4 + data.resp(c2);

end
