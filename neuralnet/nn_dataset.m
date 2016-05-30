function [InfLoss, data] = nn_dataset(nTrainingTrials, eta_0, gamma_e, sigma_train, sigmas_test)

% rng(int_indx);
% 
% nTrials = floor(logspace(2,5,16)); % cumsum(repmat([72 48 48],1,5)); % 
% nDatasets = 30;
% 
% [nTrials, dataset_ids] = meshgrid(nTrials, 1:nDatasets);
% 
% dataset   = dataset_ids(int_indx);
% nTrials     = nTrials(int_indx);
% nTrials     = nTrials + (rem(nTrials,2)==1); 

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
nepch      = 1; 
bsize      = 1; 
% eta_0      = 0.05; % optimize this
% gamma_e    = 0.0001; % and this
eta        = eta_0 ./ (1 + gamma_e*(0:(nTrainingTrials-1))); % learning rate policy
% lambda     = 0.0705; % lapse

% generate data
sig1_sq     = 3^2;
sig2_sq     = 12^2;
sigtc_sq    = 10^2;
[R,P,~,C]   = generate_popcode_simple_training(nTrainingTrials, nneuron, sig1_sq, sig2_sq, sigtc_sq, sigma_train); % figure out how to set sigma. it's not fit, because we don't fit the sigma on training trials.
% fprintf('Generated training data\n');

Xdata      = R';
Ydata      = C';

% initialize network parameters
W_init = cell(L,1);
b_init = cell(L,1);

W_init{2} = 0.05*randn(nnode(2),nnode(1));
b_init{2} = 0.00*randn(nnode(2),1);

W_init{3} = 0.05*randn(nnode(3),nnode(2));
b_init{3} = 0.00*randn(nnode(3),1);

% Evaluate network at the end of epoch
ninfloss = 12000;

%% Train network with SGD
for e = 1:nepch
    
    pp = randperm(nTrainingTrials);
    
    for bi = 1:(nTrainingTrials/bsize)
    
        bbegin = (bi-1)*bsize+1;
        bend   = bi*bsize;
        X      = Xdata(:,pp(bbegin:bend));
        Y      = Ydata(:,pp(bbegin:bend));
        
        if (e == 1) && (bi == 1)
            W = W_init;
            b = b_init;
        end
        
        [W, b] = do_backprop_on_batch(X, Y, W, b, eta(bi), mu, lambda_eff, L, ftype, 0, objective);
    
    end
    
    % Performance over training set
    Yhattrain           = zeros(1,nTrainingTrials);
    for ti = 1:nTrainingTrials
        [a, ~]          = fwd_pass(Xdata(:,ti),W,b,L,ftype);
        Yhattrain(1,ti) = a{end};
    end
    RMSEtrain = sqrt(mean((Yhattrain-P').^2));
    
    % Evaluate network at the end of epoch
    [Rinf, Pinf, s, C] = generate_popcode_noisy_data_allgains_6(ninfloss, nneuron, sig1_sq, sig2_sq, sigtc_sq, sigmas_test);
    Xinfloss                   = Rinf';
    Yinfloss                   = Pinf';
    Yhatinf                    = zeros(1,ninfloss);
    for ti = 1:ninfloss
        [a, ~]        = fwd_pass(Xinfloss(:,ti),W,b,L,ftype);
        Yhatinf(1,ti) = a{end};
    end

    Yinfloss(Yhatinf==0) = .5;
    Yhatinf(Yhatinf==0)  = .5;

    InfLoss = nanmean(Yinfloss.*(log(Yinfloss./Yhatinf)) + (1-Yinfloss).*(log((1-Yinfloss)./(1-Yhatinf)))) ... 
                           / nanmean(Yinfloss.*log(2*Yinfloss) + (1-Yinfloss).*log(2*(1-Yinfloss)));
                     
    RMSE = sqrt(mean((Yhatinf-Yinfloss).^2));
    
%     fprintf('Epoch: %i done, InfLoss on test: %f, RMSE on test: %f, NoAcceptedTrials: %i, RMSE on training data: %f \n', e, InfLoss, RMSE, length(Yinfloss), RMSEtrain);

end

data.prob = Yhatinf;
data.Chat = (Yhatinf'>0.5)' + 0.0; % category choice
data.Chat_opt = (Yinfloss'>0.5)' + 0.0; % optimal category choice
data.s = s';
data.C = C';

end
