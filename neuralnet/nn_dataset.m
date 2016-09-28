function [data, perf_train, perf_test] = nn_dataset(nTrainingTrials, eta_0, gamma_e, sigma_train, sigma_test, varargin)

train_on_test_noise = true;
baseline = 0;
quantile_type = 'weak';
nEpochs = 1;
alpha = [0 1e-4];
batch_size = 10;

C_test = [];
s_test = [];
sigmas_test = [];
R_test = [];
optimal_p_test = [];

W_init_sd = .05;
assignopts(who, varargin);

sigma_test = reshape(sigma_test, 1, length(sigma_test));

% network parameters
nhu = 200;
nLayers = 3;
nneuron = 50;
nnode = [nneuron nhu 1];
hidden_unit_type = 'relu';
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

R_train = generate_popcode(C_train, s_train, sigmas_train,...
    'sig1_sq', category_params.sigma_1^2, ...
    'sig2_sq', category_params.sigma_2^2, ...
    'tc_precision', tc_precision, 'baseline', baseline, ...
    'sprefs', sprefs);

% fprintf('Generated training data\n');

% initialize network parameters
W_init = cell(nLayers,1);
b_init = cell(nLayers,1);

W_init{2} = W_init_sd*randn(nnode(2),nnode(1));
b_init{2} = 0.00*randn(nnode(2),1);

W_init{3} = W_init_sd*randn(nnode(3),nnode(2));
b_init{3} = 0.00*randn(nnode(3),1);

% Evaluate network at the end of epoch
nTestTrials = 2160; % should this be 2160 to match the experiment? or need more

perf_test = zeros(1, nEpochs);
perf_train = perf_test;

%% Train network with SGD
        
for epoch = 1:nEpochs
    
    trial_idx = randperm(nTrainingTrials)';
    if nTrainingTrials==0
        W = W_init;
        b = b_init;
    else
        for batch = 1:(nTrainingTrials/batch_size)
            batch_begin = (batch-1)*batch_size+1;
            batch_end = batch*batch_size;
            spikes = R_train(trial_idx(batch_begin:batch_end),:)';
            C = C_train(trial_idx(batch_begin:batch_end),:)';
            
            if (epoch == 1) && (batch == 1)
                W = W_init;
                b = b_init;
            end
            
            [W, b] = do_backprop_on_batch(spikes, C, W, b, eta(batch), mu, lambda_eff, nLayers, hidden_unit_type, 0, objective, alpha);
            
        end
        
        % Performance over training set
%         [output_p_train, RMSE_train, info_loss_train, perf_train(epoch)] = fwd_pass_all(R_train, W, b, nLayers, hidden_unit_type, optimal_p_train, C_train);
    end
    
    % Evaluate network at the end of epoch
    if isempty(R_test) && isempty(optimal_p_test) && isempty(C_test) && isempty(s_test) && isempty(sigmas_test)
        [C_test, s_test] = generate_stimuli(nTestTrials, category_params);
        sigmas_test = randsample(sigma_test, nTestTrials, true)';
        [R_test, optimal_p_test] = generate_popcode(C_test, s_test, sigmas_test,...
            'nneuron', nneuron, 'sig1_sq', category_params.sigma_1^2, ...
            'sig2_sq', category_params.sigma_2^2, ...
            'tc_precision', tc_precision, 'baseline', baseline, ...
            'sprefs', sprefs);
    else
        if all(unique(C_test') == [-1 1])
            C_test(C_test==1) = 0;
            C_test(C_test==-1) = 1;
        end
    end
    
    [output_p_test, RMSE_test, info_loss_test, perf_test(epoch)] = fwd_pass_all(R_test, W, b, nLayers, hidden_unit_type, optimal_p_test, C_test);

    %     fprintf('\nEpoch %i: %.1f%% test performance\n', e, perf_test(e)*100)
    
    %     fprintf('Epoch: %i done, InfLoss on test: %f, RMSE on test: %f, NoAcceptedTrials: %i, RMSE on training data: %f \n', e, InfLoss, RMSE, length(Yinfloss), RMSEtrain);
    
end

data.C = C_test';
if all(unique(data.C) == [0 1])
    data.C(data.C==1) = -1;
    data.C(data.C==0) = 1;
end
data.sigmas = sigmas_test';
data.s = s_test';
data.R = R_test';

data.C_train = C_train';
data.C_train(data.C_train==1) = -1;
data.C_train(data.C_train==0) = 1;
% data.gains_train = gains_train';
data.sigmas_train = sigmas_train';
data.s_train = s_train';

data.nTrainingTrials = nTrainingTrials;

data.output_prob_test = output_p_test';
data.opt_prob_test = optimal_p_test';
data.info_loss_test = info_loss_test;
data.RMSE_test = RMSE_test;
data.perf_test = perf_test(end);

% data.output_prob_train = output_p_train';
% data.opt_prob_train = optimal_p_train';
% data.info_loss_train = info_loss_train;
% data.RMSE_train = RMSE_train;
data.perf_train = perf_train(end);

switch quantile_type
    case 'ultraweak'
        % quantile for confidence and choice (analogous to Bayes_Weak)
        edges = [0 quantile(data.output_prob_test, linspace(1/8, 7/8, 7)) 1];
    case 'weak'
        % quantile confidences but fix choice boundary at .5
        edges = [0, quantile(data.output_prob_test(data.output_prob_test<=.5), linspace(1/4, 3/4, 3)), 1/2, quantile(data.output_prob_test(data.output_prob_test>.5), linspace(1/4, 3/4, 3)), 1];
    case 'lowconfonly'
        edges = [zeros(1,4) 1/2 ones(1,4)];
end

for i = 2:length(edges)
    if isnan(edges(i))
        edges(i) = edges(i-1);
    end
end
[~, data.resp] = histc(data.output_prob_test, edges);



data.resp = 9-data.resp;
data.Chat = real(data.resp >= 5);
data.Chat(data.Chat==0) = -1;
data.tf = data.C==data.Chat;
% fprintf('\n%.1f%% final test performance\n', data.perf_test*100)

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
