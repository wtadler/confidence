load('/Users/bluec/Documents/GitHub/confidence/data/backup/GB_notrain_final.mat')
st(1) = psybayes_struct.valid;
st(2) = psybayes_struct.neutral;
st(3) = psybayes_struct.invalid;

%PSYCHOFUN_PCORRECT Psychometric function for percent correct (with guessing)
gamma = .5;

psychofun_pcorrect = @(x, mu, sigma, lambda)...
    bsxfun(@plus, gamma, ...
    bsxfun(@times,1-gamma-lambda,0.5*(1+erf(bsxfun(@rdivide,bsxfun(@minus,x,mu),sqrt(2)*sigma)))));
%%
nCurves = length(st);
try
    colors = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
  colors = colors.attention_colors;
catch
    colors = [0 .7 0; .6 .6 .6; .7 0 0];
end

figure(1)
clf

yl = [0 1];
xl = [-6 0];
for i = 1:nCurves
    tab = st(i);
    
    x = squeeze(tab.x);
    
    psimean = zeros(1, numel(x)); psisd = zeros(1, numel(x));
    post = tab.post(:);
    for ix = 1:numel(x)
        f = psychofun_pcorrect(x(ix), tab.mu, tab.sigma, tab.lambda);
        psimean(ix) = sum(f(:).*post);
        psisd(ix) = sqrt(sum(f(:).^2.*post) - psimean(ix)^2);
    end
    
    c = colors(i, :);
    figure(1)
    hold on
    
    
    l = plot(x, psimean, 'color', c, 'linewidth', 3)
    uistack(l, 'bottom')
    
    f = fill([x; flipud(x)], [psimean+psisd fliplr(psimean-psisd)], c, 'edgecolor', 'none', 'facealpha', .5);
    uistack(f, 'top')
    
    contrasts = tab.data(:,1);
    nBins = 6;
    binranges = linspace(min(contrasts),max(contrasts),nBins);
    [bincounts,ind] = histc(contrasts,binranges);
    outcome = tab.data(:,2);

%Most efficient version; plots error bars for each bin:
%std = (max-min)/6
    for i = 1:nBins
        y(i) = mean(outcome(ind==i));
    end
    miss = zeros(1,6);
    var = zeros(1,6);
    EB = zeros(1,6);
    for i = 1:length(y)
        miss(i) = 1-y(i);
        var(i) = (y(i)*miss(i))/((y(i)+miss(i))^2*(y(i)+miss(i)+1));% variance calc for each binned point
        EB(i) = sqrt(var(i)); %standard deviations (used as the length of each error bar)
    end
    p = errorbar(binranges, y, EB, '.', 'color', c, 'markersize', 12);

%Loop version (less efficient, but plots size-dependent points for each bin):    
%    x(1:nBins-1) = binranges(1:end-1) + diff(binranges)/2;
%    for i = 1:nBins
%        y = mean(outcome(ind==i));
%        if i < nBins
%            p = plot (x(i), y, '.', 'color', c, 'markersize', 10 + .1*bincounts(i));
%        else
%            p = plot (binranges(i), y, '.', 'color', c, 'markersize', 10 + .1*bincounts(i));
%        end
%    end

%To plot trials at the top/bottom of the screen (as opposed or in addition to binned data):
%    y = tab.data(:, 2);
%    y(y==1) = yl(2);
%    y(y==0) = yl(1);
%    n = length(contrasts);
%    noise = .005;
%    p = plot(contrasts+noise*randn(n, 1), y+noise*randn(n, 1), '.', 'color', c, 'markersize', 20)
    
end

set(gca, 'tickdir', 'out', 'ylim', yl, 'xlim', xl, 'clipping', 'off')
xlabel('log contrast')
ylabel('% correct')