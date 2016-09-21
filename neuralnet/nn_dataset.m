function [RMSEtrain, data, perf_train, perf_test] = nn_dataset(nTrainingTrials, eta_0, gamma_e, sigma_train, sigma_test, varargin)

train_on_test_noise = true;
baseline = 0;
quantile_type = 'weak';
nEpochs = 10;
alpha = [0 1e-4];
batch_size = 5;
assignopts(who, varargin);

sigma_test = reshape(sigma_test, 1, length(sigma_test));

% network parameters
nhu = 200;
L = 3;
nneuron = 50;
nnode = [nneuron nhu 1];
ftype = 'relu';
objective = 'xent';

% training parameters
mu = 0.0;
lambda_eff = 0.0;
% eta_0      = 0.05; % optimize this
% gamma_e    = 0.0001; % and this
eta = eta_0 ./ (1 + gamma_e*(0:(nTrainingTrials-1)))'; % learning rate policy
% lambda     = 0.0705; % lapse

tc_precision = .01; % aka tau_t. formerly .01

sprefs = linspace(-40, 40, nneuron);

% generate data
category_params.sigma_1 = 3;
category_params.sigma_2 = 12;

[C_train, s_train] = generate_stimuli(nTrainingTrials, category_params);

if ~train_on_test_noise
    sigmas_train = sigma_train * ones(nTrainingTrials, 1);
else
    sigmas_train = randsample(sigma_test, nTrainingTrials, true)';
end

[R_train, optimal_P_train, optimal_D_train, gains_train] = generate_popcode(C_train, s_train, sigmas_train,...
    'sig1_sq', category_params.sigma_1^2, ...
    'sig2_sq', category_params.sigma_2^2, ...
    'tc_precision', tc_precision, 'baseline', baseline, ...
    'sprefs', sprefs);

% fprintf('Generated training data\n');

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
RMSEtrain = nan;
%% Train network with SGD
for e = 1:nEpochs
    
    pp = randperm(nTrainingTrials)';
    if nTrainingTrials==0
        W = W_init;
        b = b_init;
    else
        for bi = 1:(nTrainingTrials/batch_size)
            
            bbegin = (bi-1)*batch_size+1;
            bend = bi*batch_size;
            X = R_train(pp(bbegin:bend),:)';
            Y = C_train(pp(bbegin:bend),:)';
            
            if (e == 1) && (bi == 1)
                W = W_init;
                b = b_init;
            end
            
            [W, b] = do_backprop_on_batch(X, Y, W, b, eta(bi), mu, lambda_eff, L, ftype, 0, objective, alpha);
            
        end
        
        % Performance over training set
        Yhattrain = zeros(nTrainingTrials,1);
        for ti = 1:nTrainingTrials
            [a, ~] = fwd_pass(R_train(ti,:)',W,b,L,ftype);
            Yhattrain(ti) = a{end};
        end
        RMSEtrain = sqrt(mean((Yhattrain-optimal_P_train).^2)); % use this as objective
        perf_train(e) = mean((Yhattrain > .5) == C_train);
        %     fprintf('\nEpoch %i: %.1f%% training performance', e, perf_train(e)*100)
    end
    
    % Evaluate network at the end of epoch
    [C_test, s_test] = generate_stimuli(nTestTrials, category_params);
    sigmas_test = randsample(sigma_test, nTestTrials, true)';
    [R_test, optimal_P_test, optimal_D_test, gains_test] = generate_popcode(C_test, s_test, sigmas_test,...
        'nneuron', nneuron, 'sig1_sq', category_params.sigma_1^2, ...
        'sig2_sq', category_params.sigma_2^2, ...
        'tc_precision', tc_precision, 'baseline', baseline, ...
        'sprefs', sprefs);
    
    Yhatinf = zeros(nTestTrials,1);
    for ti = 1:nTestTrials
        [a, ~] = fwd_pass(R_test(ti,:)',W,b,L,ftype);
        Yhatinf(ti) = a{end}; % output unit
    end
    
    % set 0 spike trials to .5 prob. do we need this? not justified with
    % quantiles?
    optimal_P_test(Yhatinf==0) = .5; 
    Yhatinf(Yhatinf==0)  = .5;
    
    InfLoss = nanmean(optimal_P_test.*(log(optimal_P_test./Yhatinf)) + (1-optimal_P_test).*(log((1-optimal_P_test)./(1-Yhatinf)))) ...
        / nanmean(optimal_P_test.*log(2*optimal_P_test) + (1-optimal_P_test).*log(2*(1-optimal_P_test)));
    
    RMSE = sqrt(mean((Yhatinf-optimal_P_test).^2));
    
    perf_test(e) = mean(C_test==real(Yhatinf > .5));
    %     fprintf('\nEpoch %i: %.1f%% test performance\n', e, perf_test(e)*100)
    
    %     fprintf('Epoch: %i done, InfLoss on test: %f, RMSE on test: %f, NoAcceptedTrials: %i, RMSE on training data: %f \n', e, InfLoss, RMSE, length(Yinfloss), RMSEtrain);
    
end

data.C = C_test';
data.C(data.C==1) = -1;
data.C(data.C==0) = 1;

data.gains = gains_test';
data.sigmas = sigmas_test';

data.s = s_test';

data.nTrainingTrials = nTrainingTrials;

data.prob = Yhatinf';

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



data.resp = 9-data.resp';
data.Chat = real(data.resp >= 5)';
data.Chat(data.Chat==0) = -1;
data.tf = data.C==data.Chat;
fprintf('\n%.1f%% final test performance\n', perf_test(e)*100)

c1 = data.Chat == -1;
c2 = data.Chat == 1;
data.g = [];
data.g(c1) = 5 - data.resp(c1);
data.g(c2) = -4 + data.resp(c2);

    function [C,s] = generate_stimuli(N, category_params)
        % generate data
        C = [ones(N/2, 1); zeros(N/2, 1)];
        s = [];
        s(C == 1) = stimulus_orientations(category_params, 1, sum(C == 1), 'same_mean_diff_std');
        s(C == 0) = stimulus_orientations(category_params, 2, sum(C == 0), 'same_mean_diff_std');
        s = s'; 
    end

end
