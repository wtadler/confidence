function [xmin,tab,output] = psybayes(tab,method,vars,xi,yi,plotflag)
%PSYBAYES Adaptive Bayesian method for measuring threshold of psychometric function
%
% Author: Luigi Acerbi

if nargin < 1; tab = []; end
if nargin < 2; method = []; end
if nargin < 3; vars = []; end
if nargin < 4; xi = []; yi = []; end
if nargin < 6 || isempty(plotflag); plotflag = 0; end

xmin = [];

% Default method is expected entropy minimization
if isempty(method); method = 'ent'; end

if isempty(vars)
    switch lower(method)
        case 'ent'; vars = [1 1 1];
        case 'var'; vars = [1 0 0];
        otherwise
            error('Unknown optimization method.');
    end
end
if numel(vars) ~= 3; error('VARS need to be a 3-element array for MU, SIGMA and LAMBDA.'); end

if isempty(tab) || ~isfield(tab,'post')  % First call, initialize everything
    tabinit = tab;
    
    tab.ntrial = 0;     % Trial number
    tab.data = [];      % Record of data
    
    % Default grid sizes
    nx = 61;
    nmu = 51;
    nsigma = 25;
    nlambda = 25;
    
    tab.mu = [];
    tab.logsigma = [];
    tab.lambda = [];
    tab.x = [];
        
    % Grid over parameters of psychometric function
    if isfield(tabinit,'range')
        
        % Get grid sizes (third element in initialization range field)
        if isfield(tabinit.range,'mu') && numel(tabinit.range.mu > 2)
            nmu = tabinit.range.mu(3);
        end
        if isfield(tabinit.range,'sigma') && numel(tabinit.range.sigma > 2)
            nsigma = tabinit.range.sigma(3);
        elseif isfield(tabinit.range,'logsigma') && numel(tabinit.range.logsigma > 2)
            nsigma = tabinit.range.logsigma(3);            
        end
        if isfield(tabinit.range,'lambda') && numel(tabinit.range.lambda > 2)
            nlambda = tabinit.range.lambda(3);
        end
        if isfield(tabinit.range,'x') && numel(tabinit.range.x > 2)
            nx = tabinit.range.x(3);
        end
        
        % Prepare ranges
        if isfield(tabinit.range,'mu')
            tab.mu(:,1,1) = linspace(tabinit.range.mu(1),tabinit.range.mu(2),nmu);
        else
            error('Cannot find a field for MU in initialization range struct.');
        end
        if isfield(tabinit.range,'sigma')
            tab.logsigma(1,:,1) = linspace(log(tabinit.range.sigma(1)), log(tabinit.range.sigma(2)), nsigma);
        elseif isfield(tabinit.range,'logsigma')
            tab.logsigma(1,:,1) = linspace(tabinit.range.logsigma(1), tabinit.range.logsigma(2), nsigma);
        else
            error('Cannot find a field for SIGMA in initialization range struct.');
        end
        if isfield(tabinit.range,'lambda')
            tab.lambda(1,1,:) = linspace(tabinit.range.lambda(1),tabinit.range.lambda(2),nlambda);
        end
        if isfield(tabinit,'x') && ~isempty(tabinit.x)
            tab.x(1,1,1,:) = tabinit.x(:);
        elseif isfield(tabinit.range,'x') && ~isempty(tabinit.range.x)
            tab.x(1,1,1,:) = linspace(tabinit.range.x(1),tabinit.range.x(2),nx);
        else
            error('Test grid X not provided in initialization struct.');
        end
    end
    
    % Default ranges
    if isempty(tab.mu)
        tab.mu(:,1,1) = linspace(2,4,nmu);
    end
    if isempty(tab.logsigma)
        tab.logsigma(1,:,1) = linspace(log(0.01), log(1), nsigma);
    end
    if isempty(tab.lambda)
        tab.lambda(1,1,:) = linspace(0, 0.2, nlambda);
    end
    if isempty(tab.x)
        tab.x(1,1,1,:) = linspace(tab.mu(1),tab.mu(end),nx);
    end
    
    if isfield(tabinit,'units')
        tab.units = tabinit.units;
    else
        tab.units.x = [];
        tab.units.mu = [];
        tab.units.sigma = []; 
        tab.units.lambda = []; 
    end
    
    % Enforce symmetry (left/right with equal probability)
    if isfield(tabinit,'forcesymmetry') && ~isempty(tabinit.forcesymmetry)
        tab.forcesymmetry = tabinit.forcesymmetry;
    else
        tab.forcesymmetry = 0;        
    end
    
    % By default, very wide Gaussian prior on mu with slight preference for
    % the middle of the stimulus range
    muprior = [mean(tab.mu),tab.mu(end)-tab.mu(1)];    % mean and std
    if isfield(tabinit,'priors') && ~isempty(tabinit.priors)
        if isfield(tabinit.priors,'mu') && ~isempty(tabinit.priors.mu)
            muprior = tabinit.priors.mu;        
        end
    end
    priormu = exp(-0.5.*((tab.mu-muprior(1))/muprior(2)).^2);  
    
    % By default flat prior on log sigma (Jeffrey's 1/sigma prior in sigma 
    % space); more in general log-normal prior
    logsigmaprior = [mean(tab.logsigma),Inf];    % mean and std
    if isfield(tabinit,'priors') && ~isempty(tabinit.priors)
        if isfield(tabinit.priors,'sigmaprior') && ~isempty(tabinit.priors.sigmaprior)
            logsigmaprior = tabinit.priors.logsigmaprior;        
        end
    end
    priorlogsigma = exp(-0.5.*((tab.logsigma-logsigmaprior(1))/logsigmaprior(2)).^2);  
    
    % Beta(a,b) prior on lambda, with correction
    lambdaprior = [1,19];
    if isfield(tabinit,'priors') && ~isempty(tabinit.priors)
        if isfield(tabinit.priors,'lambda') && ~isempty(tabinit.priors.lambda)
            lambdaprior = tabinit.priors.lambda;        
        end
    end
    
    temp = tab.lambda(:)';
    temp = [0, temp + 0.5*[diff(temp),0]];
    a = lambdaprior(1); b = lambdaprior(2);
    priorlambda(1,1,:) = betainc(temp(2:end),a,b) - betainc(temp(1:end-1),a,b);
    
    priormu = priormu./sum(priormu);
    priorlogsigma = priorlogsigma./sum(priorlogsigma);
    priorlambda = priorlambda./sum(priorlambda);
    
    % Prior (posterior at iteration zero) over parameter vector theta
    tab.post = bsxfun(@times,bsxfun(@times,priormu,priorlogsigma),priorlambda);
    
    % Define sigma in addition to log sigma
    tab.sigma = exp(tab.logsigma);
    
    % Guess rate for PCORRECT psychometric functions
    if isfield(tabinit,'gamma')
        tab.gamma = tabinit.gamma;
    else
        tab.gamma = [];
    end
    
    tab.f = [];
end

% Choose correct psychometric function (YES/NO or PCORRECT)
if ~isempty(tab.gamma)
    psychofun = @(x_,mu_,sigma_,lambda_) psychofun_pcorrect(x_,mu_,sigma_,lambda_,tab.gamma);
else    
    psychofun = @(x_,mu_,sigma_,lambda_) psychofun_yesno(x_,mu_,sigma_,lambda_);
end

% Precompute psychometric function and its logarithm
if isempty(tab.f)
    tab.f = psychofun(tab.x,tab.mu,tab.sigma,tab.lambda);
end

% Update log posterior given the new data points XI, YI
if ~isempty(xi) && ~isempty(yi)    
    for i = 1:numel(xi)
        if yi(i) == 1
            like = psychofun(xi(i),tab.mu,tab.sigma,tab.lambda);
        else
            like = 1 - psychofun(xi(i),tab.mu,tab.sigma,tab.lambda); 
        end
        tab.post = tab.post.*like;
    end
    tab.post = tab.post./sum(tab.post(:));
    
    tab.ntrial = tab.ntrial + numel(xi);
    tab.data = [tab.data; xi(:) yi(:)];
end

% Compute mean of the posterior of mu
postmu = sum(sum(tab.post,2),3);
emu = sum(postmu.*tab.mu);

% Randomly remove half of the x
if tab.forcesymmetry
    if rand() < 0.5; xindex = tab.x < emu; else xindex = tab.x >= emu; end
else
    xindex = true(size(tab.x));
end

% Compute best sampling point X that minimizes variance of MU
if nargin > 0
        
    xred = tab.x(xindex);
    
    % Compute posteriors at next step for R=1 and R=0
    [post1,post0,r1] = nextposterior(tab.f(:,:,:,xindex),tab.post);

    % Marginalize over unrequested variables
    index = find(~vars);
    for iTheta = index
        post1 = sum(post1,iTheta);
        post0 = sum(post0,iTheta);
    end
    
    switch lower(method)
        case {'var','variance'}
            post1 = squeeze(post1);
            post0 = squeeze(post0);
            index = find(vars,1);
            switch index
                case 1; qq = tab.mu(:);
                case 2; qq = tab.logsigma(:);
                case 3; qq = tab.lambda(:);
            end
            mean1 = sum(bsxfun(@times,post1,qq),1);
            mean0 = sum(bsxfun(@times,post0,qq),1);
            var1 = sum(bsxfun(@times,post1,qq.^2),1) - mean1.^2;
            var0 = sum(bsxfun(@times,post0,qq.^2),1) - mean0.^2;
            target = r1(:).*var1(:) + (1-r1(:)).*var0(:);
                        
        case {'ent','entropy'}
            temp1 = -post1.*log(post1);
            temp0 = -post0.*log(post0);            
            temp1(~isfinite(temp1)) = 0;
            temp0(~isfinite(temp0)) = 0;
            H1 = temp1;     H0 = temp0;
            for iTheta = find(vars)
                H1 = sum(H1,iTheta);
                H0 = sum(H0,iTheta);
            end
            target = r1(:).*H1(:) + (1-r1(:)).*H0(:);
            
        otherwise
            error('Unknown method. Allowed methods are ''var'' and ''ent'' for, respectively, predicted variance and predicted entropy minimization.');
    end

    % Location X that minimizes target metric
    [~,index] = min(target);
    xmin = xred(index);
end

% Plot marginal posteriors and suggested next point X
if plotflag
    % Arrange figure panels in a 2 x 2 vignette
    rows = 2; cols = 2;
    % rows = 1; cols = 4;   % Alternative horizontal arrangement
    
    x = tab.x(:)';
    
    % Plot psychometric function
    subplot(rows,cols,1);
    
    psimean = zeros(1,numel(x));    psisd = zeros(1,numel(x));
    post = tab.post(:);
    for ix = 1:numel(x)
        f = psychofun(x(ix),tab.mu,tab.sigma,tab.lambda);
        psimean(ix) = sum(f(:).*post);
        psisd(ix) = sqrt(sum(f(:).^2.*post) - psimean(ix)^2);
    end    
    hold off;
    fill([x, fliplr(x)], [psimean + psisd, fliplr(psimean - psisd)], 0.8*[1 1 1], 'edgecolor', 'none');
    hold on;
    plot(x, psimean,'k','LineWidth',1);
    plot([xmin,xmin],[0,1],':r', 'LineWidth', 2);
    if ~isempty(tab.data)
        scatter(tab.data(:,1),tab.data(:,2),20,'ko','MarkerFaceColor','r','MarkerEdgeColor','none');
    end
    box off; set(gca,'TickDir','out');    
    if ~isempty(tab.units.x); string = [' (' tab.units.x ')']; else string = []; end
    xlabel(['x' string]);
    if isempty(tab.gamma)
        ylabel('Pr(response = 1)');
    else
        ylabel('Pr(response correct)');        
    end
    axis([min(x) max(x), 0 1]);
    title(['Psychometric function (trial ' num2str(num2str(tab.ntrial)) ')']);
    
    % Plot posterior for mu
    subplot(rows,cols,3);
    y = sum(sum(tab.post,2),3);
    y = y/sum(y*diff(tab.mu(1:2)));    
    hold off;
    plot(tab.mu(:), y(:), 'k', 'LineWidth', 1);
    hold on;
    box off; set(gca,'TickDir','out');
    if ~isempty(tab.units.mu); string = [' (' tab.units.mu ')']; else string = []; end
    xlabel(['\mu' string]);
    ylabel('Posterior probability');
    % Compute SD of the posterior
    y = y/sum(y);
    ymean = sum(y.*tab.mu);
    ysd = sqrt(sum(y.*tab.mu.^2) - ymean^2);    
    if ~isempty(tab.units.mu); string = [' ' tab.units.mu]; else string = []; end
    title(['Posterior \mu = ' num2str(ymean,'%.2f') ' ± ' num2str(ysd,'%.2f') string])
    yl = get(gca,'Ylim'); axis([get(gca,'Xlim'),0,yl(2)]);
    plot([xmin,xmin],get(gca,'Ylim'),':r', 'LineWidth', 2);

    % Plot posterior for sigma
    subplot(rows,cols,2); hold off;
    y = sum(sum(tab.post,1),3);
    y = y/sum(y*diff(tab.sigma(1:2)));
    plot(tab.sigma(:), y(:), 'k', 'LineWidth', 1); hold on;
    box off; set(gca,'TickDir','out','XScale','log');
    %box off; set(gca,'TickDir','out','XScale','log','XTickLabel',{'0.1','1','10'});
    if ~isempty(tab.units.sigma); string = [' (' tab.units.sigma ')']; else string = []; end
    xlabel(['\sigma' string]);
    ylabel('Posterior probability');
    % title(['Marginal posterior distributions (trial ' num2str(tab.ntrial) ')']);
    % Compute SD of the posterior
    y = (y.*tab.sigma)/sum(y.*tab.sigma);
    ymean = sum(y.*tab.sigma);
    ysd = sqrt(sum(y.*tab.sigma.^2) - ymean^2);
    if ~isempty(tab.units.sigma); string = [' ' tab.units.sigma]; else string = []; end
    title(['Posterior \sigma = ' num2str(ymean,'%.2f') ' ± ' num2str(ysd,'%.2f') string]);
    yl = get(gca,'Ylim'); axis([tab.sigma(1),tab.sigma(end),0,yl(2)]);

    % Plot posterior for lambda
    subplot(rows,cols,4); hold off;
    y = sum(sum(tab.post,1),2);    
    y = y/sum(y*diff(tab.lambda(1:2)));
    plot(tab.lambda(:), y(:), 'k', 'LineWidth', 1); hold on;
    box off; set(gca,'TickDir','out');
    if ~isempty(tab.units.lambda); string = [' (' tabs.units.lambda ')']; else string = []; end
    xlabel(['\lambda' string]);
    ylabel('Posterior probability');    
    % Compute SD of the posterior
    y = y/sum(y);    
    ymean = sum(y.*tab.lambda);
    ysd = sqrt(sum(y.*tab.lambda.^2) - ymean^2);    
    if ~isempty(tab.units.lambda); string = [' ' tabs.units.lambda]; else string = []; end
    title(['Posterior \lambda = ' num2str(ymean,'%.2f') ' ± ' num2str(ysd,'%.2f') string])
    yl = get(gca,'Ylim'); axis([get(gca,'Xlim'),0,yl(2)]);

    
    set(gcf,'Color','w');
end

% Compute parameter estimates
if nargout > 2
    
    % Compute mean and variance of the estimate of MU
    postmu = sum(sum(tab.post,2),3);
    postmu = postmu./sum(postmu,1);
    emu = sum(postmu.*tab.mu,1);
    estd = sqrt(sum(postmu.*tab.mu.^2,1) - emu.^2);
    output.mu.mean = emu;
    output.mu.std = estd;
    
    % Compute mean and variance of the estimate of LOGSIGMA and SIGMA
    postlogsigma = sum(sum(tab.post,1),3);    
    postlogsigma = postlogsigma./sum(postlogsigma,2);    
    emu = sum(postlogsigma.*tab.logsigma,2);
    estd = sqrt(sum(postlogsigma.*tab.logsigma.^2,2) - emu.^2);
    output.logsigma.mean = emu;
    output.logsigma.std = estd;
    
    postsigma = postlogsigma./tab.sigma;
    postsigma = postsigma./sum(postsigma,2);    
    emu = sum(postsigma.*tab.sigma,2);
    estd = sqrt(sum(postsigma.*tab.sigma.^2,2) - emu.^2);
    output.sigma.mean = emu;
    output.sigma.std = estd;
    
    % Compute mean and variance of the estimate of LAMBDA
    postlambda = sum(sum(tab.post,1),2);
    postlambda = postlambda./sum(postlambda,3);    
    emu = sum(postlambda.*tab.lambda,3);
    estd = sqrt(sum(postlambda.*tab.lambda.^2,3) - emu.^2);
    output.lambda.mean = emu;
    output.lambda.std = estd;
end

% Only one argument assumes that this is the final call
if nargin < 2
    % Empty some memory
    tab.f = [];
end

end


%--------------------------------------------------------------------------
function f = psychofun_yesno(x,mu,sigma,lambda)
%PSYCHOFUN_YESNO Psychometric function for YES/NO tasks

f = bsxfun(@plus, lambda/2, ...
    bsxfun(@times,1-lambda,0.5*(1+erf(bsxfun(@rdivide,bsxfun(@minus,x,mu),sqrt(2)*sigma)))));

end

function f = psychofun_pcorrect(x,mu,sigma,lambda,gamma)
%PSYCHOFUN_PCORRECT Psychometric function for percent correct (with guessing)

f = bsxfun(@plus, gamma, ...
    bsxfun(@times,1-gamma-lambda,0.5*(1+erf(bsxfun(@rdivide,bsxfun(@minus,x,mu),sqrt(2)*sigma)))));

end

%--------------------------------------------------------------------------
function [post1,post0,r1] = nextposterior(f,post)
%NEXTPOSTERIOR Compute posteriors on next trial depending on possible outcomes

    mf = 1-f;
    post1 = bsxfun(@times, post, f);
    r1 = sum(sum(sum(post1,1),2),3);
    post0 = bsxfun(@times, post, mf);
    post1 = bsxfun(@rdivide, post1, sum(sum(sum(post1,1),2),3));
    post0 = bsxfun(@rdivide, post0, sum(sum(sum(post0,1),2),3));    
end