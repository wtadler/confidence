load('/Local/Users/adler/Google Drive/Will - Confidence/Analysis/attention3/notrain_20160309_155331.mat')
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

yl = [.5 .75];
xl = [-4 0];
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
    
    y = tab.data(:, 2);
    y(y==1) = yl(2);
    y(y==0) = yl(1);
    contrasts = tab.data(:,1);
    n = length(contrasts);
    noise = .005;
    p = plot(contrasts+noise*randn(n, 1), y+noise*randn(n, 1), '.', 'color', c, 'markersize', 20)
    
end

set(gca, 'tickdir', 'out', 'ylim', yl, 'xlim', xl, 'clipping', 'off')
xlabel('log contrast')
ylabel('% correct')