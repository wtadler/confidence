function [samples, loglikes, logpriors, abortflag] = slice_sample(N, logdist, xx, widths, varargin)
%SLICE_SAMPLE simple axis-aligned implementation of slice sampling for vectors
%
%     [samples loglikes] = slice_sample(N, burn, logdist, xx, widths, step_out, verbose, varargin)
%
% Inputs:
%             N  1x1  Number of samples to gather
%          burn  1x1  after burning period of this length
%       logdist  @fn  function logprobstar = logdist(xx, varargin{:})
%            xx  Dx1  initial state (or array with D elements)
%        widths  Dx1  or 1x1, step sizes for slice sampling
%      step_out bool  set to true if widths may sometimes be far too small
%                     (default 0)
%       verbose bool  set to true prints log to screen (default 1)
%      varargin   -   any extra arguments are passed on to logdist
%
% Outputs:
%      samples  DxN   samples stored in columns (regardless of original shape)
%      loglikes 1xN   log-likelihood of samples (optional)
%
% Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
% Luigi Acerbi inteface tweaks Jan 2012, loglikes output Feb 2013
% See Pseudo-code in David MacKay's text book p375

% set defaults
burn = 0; % no burnin
thin = 1; % no thinning
step_out = 0; % no stepping out
progress_report_interval = Inf; % no printing
chain_id = 0;
logpriordist = []; % assume logdist is the posterior. if logpriordist is specified, assume logdist is the log likelihood
time_lim = Inf; % no time limit
log_fid = [];
static_dims = [];

assignopts(who,varargin);

if isempty(log_fid)
    print_fcn = @fprintf;
else
    print_fcn = @(s) fprintf(log_fid,'%s\n',s);
end

start_t = tic;
abortflag = 0;

% startup stuff
D = numel(xx);
samples = nan(D, N);
if nargout > 1; loglikes = nan(1, N); end
if nargout > 2; logpriors = nan(1, N); end

if numel(widths) == 1
    widths = repmat(widths, D, 1);
end

if ~isempty(logpriordist)
    logprior = feval(logpriordist, xx);
    if logprior==-Inf
        log_Px=-Inf;
    else
        loglik = feval(logdist, xx);
        log_Px = loglik + logprior;
    end
else
    log_Px = feval(logdist, xx);
end

N = N*thin;
len_iter = floor(N/20);

% Main loop
for ii = 1:(N+burn)
if toc(start_t)>time_lim
    print_fcn(sprintf('slice_sample aborted after %.1f minutes to avoid walltime_limit\n', toc(start_t)/60));
    abortflag = 1;
    return
end
    
    if mod(ii,progress_report_interval)==0
        if ii<=burn
            print_fcn(sprintf('Chain %g: Burnin sample %g/%g',chain_id,ii,burn));
        else
            print_fcn(sprintf('Chain %g: Sample %g/%g',chain_id,ii-burn,N));
        end
		print_fcn(sprintf('%.1f minutes elapsed', toc(start_t)/60));
		print_fcn(sprintf('%i/',fix(clock)));
        print_fcn(sprintf('\n'));
    end
        
    old_xx = xx;
    % old_log_Px = log_Px;
    log_uprime = log(rand) + log_Px;

    % Sweep through axes (simplest thing)
    for dd = setdiff(1:D, static_dims);%1:setD
        x_l = xx;
        x_r = xx;
        xprime = xx;

        % Create a horizontal interval (x_l, x_r) enclosing xx
        rr = rand;
        x_l(dd) = xx(dd) - rr*widths(dd);
        x_r(dd) = xx(dd) + (1-rr)*widths(dd);
        if step_out
            % Typo in early editions of book. Book said compare to u, but it should say u'
            while (feval(logdist, x_l) > log_uprime)
                x_l(dd) = x_l(dd) - widths(dd);
            end
            while (feval(logdist, x_r) > log_uprime)
                x_r(dd) = x_r(dd) + widths(dd);
            end
        end

        % Inner loop:
        % Propose xprimes and shrink interval until good one found
        zz = 0;
        while 1
            zz = zz + 1;

            xprime(dd) = rand()*(x_r(dd) - x_l(dd)) + x_l(dd);
            
            
            if ~isempty(logpriordist)
                logprior = feval(logpriordist, xprime);
                if logprior==-Inf
                    log_Px=-Inf;
                else
                    loglik = feval(logdist, xprime);
                    log_Px = loglik + logprior;
                end
                
            else
                log_Px = feval(logdist, xprime);
            end
%             log_Px = feval(logdist, xprime, varargin{:});

            if log_Px > log_uprime
                break % this is the only way to leave the while loop
            else
                % Shrink in
                if xprime(dd) > xx(dd)
                    x_r(dd) = xprime(dd);
                elseif xprime(dd) < xx(dd)
                    x_l(dd) = xprime(dd);
                else
                    % old_log_Px
                    old_xx
                    xx
                    errorstr = 'BUG DETECTED: Shrunk to current position and still not acceptable. Current position: ';
                    for k = 1:(length(xx)-1); errorstr = [errorstr num2str(xx(k)) ', ']; end
                    errorstr = [errorstr num2str(xx(end)) '; Log p = ' num2str(log_Px), ',' num2str(log_uprime) '.'];
                    error(errorstr);
                    
                end
            end
        end
        xx(dd) = xprime(dd);
    end
    % Record samples
    if ii > burn && mod(ii-burn,thin)==0
        idx = (ii-burn)/thin;
        samples(:, idx) = xx(:);
        if nargout > 1
            if isempty(logpriordist)
                loglikes(idx) = log_Px;
            else
                loglikes(idx) = loglik;
                logpriors(idx) = logprior;
            end
        end
    end
end