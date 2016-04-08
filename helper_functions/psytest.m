%PSYTEST Test of PSYBAYES

%--------------------------------------------------------------------------
% Definitions for PSYBAYES

% PSYBAYES initialization structure
psyinit = [];

% Set change level (for PCORRECT psychometric functions)
psyinit.gamma = 0.5;        
% psyinit.gamma = [];   % Leave it empty for YES/NO psychometric functions

% Define range for stimulus and for parameters of the psychometric function
% (lower bound, upper bound, number of points)
psyinit.range.x = [-6,0,61];
psyinit.range.mu = [-6,0,51];
psyinit.range.sigma = [0.05,1,25];      % The range for sigma is automatically converted to log spacing
psyinit.range.lambda = [.15,0.5,25];

% Define priors over parameters
psyinit.priors.mu = [-2,1.2];                  % mean and std of (truncated) Gaussian prior over MU
psyinit.priors.logsigma = [log(0.1),Inf];   % mean and std of (truncated) Gaussian prior over log SIGMA (Inf std means flat prior)
psyinit.priors.lambda = [20 39];             % alpha and beta parameter of beta pdf over LAMBDA

% Units -- used just for plotting in axis labels and titles
psyinit.units.x = 'cm';
psyinit.units.mu = 'cm';
psyinit.units.sigma = 'cm';
psyinit.units.lambda = [];

method = 'ent';     % Minimize the expected posterior entropy
% vars = [1 0 0];     % Minimize posterior entropy of the mean only
vars = [1 1 1];     % This choice minimizes joint posterior entropy of mean, sigma and lambda
plotflag = 1;       % Plot visualization

%--------------------------------------------------------------------------

% Parameters of simulated observer
mu = -2;
sigma = log(2);
lambda = 0.3;

% Psychometric function for the simulated observer
psychofun = @(x) lambda/2 + (1-lambda).*0.5*(1+erf((x-mu)./(sqrt(2)*sigma)));

%--------------------------------------------------------------------------
% Start running

display(['Simulating an observer with MU = ' num2str(mu) ' cm; SIGMA = ' num2str(sigma) ' cm; LAMBDA = ' num2str(lambda) '.']);
display('Press a key to simulate a trial.')


% Get first recommended point under the chosen optimization method
% (last argument is a flag for plotting)
[x,post] = psybayes(psyinit, method, vars, [], [], plotflag);

pause;

Ntrials = 200;

for iTrial = 1:Ntrials
    % Simulate observer's response given stimulus x
    r = rand < psychofun(x);

    % Get next recommended point that minimizes predicted entropy 
    % given the current posterior and response r at x
    tic
    [x, post] = psybayes(post, method, vars, x, r, plotflag);
    toc
        
    %--------------------------------------------------------------------
    % This part is just for additional plotting also the predicted posterior 
    % entropy placement bar

    if plotflag
        % Compute alternative placement under minimum predicted posterior
        % entropy for all variables
        x2 = psybayes(post, 'ent', [1 1 1]);
        
        % Add entropy prediction and legend to the graphs
        subplot(2,2,1);
        hold on;
        hl(1) = plot([-1 -1], [-1 -1], ':r', 'LineWidth', 2);
        hl(2) = plot([x2 x2],[0,1],':b', 'LineWidth', 2);
        hold off;
        h = legend(hl, 'Ent1','Ent');
        set(h,'Location','NorthWest','Box','off');

        subplot(2,2,3);
        plot([x2 x2],get(gca,'Ylim'),':b', 'LineWidth', 2);

        % Add bars of true value
        subplot(2,2,3);
        plot(mu*[1 1],get(gca,'Ylim'),'k','LineWidth',2);
        subplot(2,2,2);
        plot(sigma*[1 1],get(gca,'Ylim'),'k','LineWidth',2);
        axis([get(gca,'Xlim'),get(gca,'Ylim')]);
        hl = plot([-1 -1],[-1 -1],'k','LineWidth',2);
        h = legend(hl, 'True parameter value');
        set(h,'Location','NorthEast','Box','off');

        subplot(2,2,4);
        plot(lambda*[1 1],get(gca,'Ylim'),'k','LineWidth',2);
    end
    
    drawnow;
    %---------------------------------------------------------------------
    
    % pause;
end
% Once you are done, clean posterior from temporary variables
[~,post] = psybayes(post);

% Save posterior, can be used in following sessions
% save('posterior.mat','post');

