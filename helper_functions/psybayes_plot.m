st(1) = load('1.mat');
st(2) = load('2.mat');

%PSYCHOFUN_PCORRECT Psychometric function for percent correct (with guessing)
gamma = .5;

psychofun_pcorrect = @(x, mu, sigma, lambda)...
    bsxfun(@plus, gamma, ...
    bsxfun(@times,1-gamma-lambda,0.5*(1+erf(bsxfun(@rdivide,bsxfun(@minus,x,mu),sqrt(2)*sigma)))));
%%
nCurves = length(st);
colors = load('/Users/will/Google Drive/MATLAB/utilities/MyColorMaps.mat');
colors = colors.attention_colors;

figure(1)
clf

for i = 1:nCurves
    tab = st(i).post;
    
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

        f = fill([x; flipud(x)], [psimean+psisd fliplr(psimean-psisd)], c, 'edgecolor', 'none', 'facealpha', .5)
    uistack(f, 'top')

end

set(gca, 'tickdir', 'out')
xlabel('contrast')
ylabel('% correct')