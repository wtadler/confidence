function psybayes_plot(filename, nBins)

load(filename)

st(1) = psybayes_struct.valid;
st(2) = psybayes_struct.neutral;
st(3) = psybayes_struct.invalid;

%PSYCHOFUN_PCORRECT Psychometric function for percent correct (with guessing)
gamma = .5;

psychofun_pcorrect = @(x, mu, sigma, lambda)...
    bsxfun(@plus, gamma, ...
    bsxfun(@times,1-gamma-lambda,0.5*(1+erf(bsxfun(@rdivide,bsxfun(@minus,x,mu),sqrt(2)*sigma)))));
%%
nBins = nBins+1;

nCurves = length(st);
try
    colors = load('~/Google Drive/MATLAB/utilities/MyColorMaps.mat');
    colors = colors.attention_colors;
catch
    colors = [0 .7 0; .6 .6 .6; .7 0 0];
end

running_minimum_x = 0;
yl = [0 1];

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
    hold on
    
    l = plot(x, psimean, 'color', c, 'linewidth', 3);
    uistack(l, 'bottom')
    
    f = fill([x; flipud(x)], [psimean+psisd fliplr(psimean-psisd)], c, 'edgecolor', 'none', 'facealpha', .5);
    uistack(f, 'top')
    
    contrasts = tab.data(:,1);
    outcome = tab.data(:, 2);
    
    edges = linspace(min(contrasts), max(contrasts)+.01, nBins);
    centers = edges(1:end-1)+diff(edges)/2;
    
    [~, ind] = histc(contrasts, edges);
    
    bin_mean = zeros(1, nBins-1);
    EB = zeros(1, nBins-1);
    
    %Plot each of the first four bins individually:
    for bin = 1:length(bin_mean)
        bin_mean(bin) = mean(outcome(ind==bin));
        
        a = sum(outcome(ind==bin)==1);
        b = sum(outcome(ind==bin)==0);
        
        beta_var = (a*b)/((a+b)^2*(a+b+1));
        EB(bin) = sqrt(beta_var);
    end
        
    %And finally plot everything:
    p = errorbar(centers, bin_mean, EB, '.', 'color', c, 'markersize', 12, 'linewidth', 1.5);

    %To plot trials at the top/bottom of the screen (as opposed or in addition to binned data):
    y = tab.data(:, 2);
    noise = .02;
    y(y==1) = yl(2)-noise*randn(sum(y==1), 1);
    y(y==0) = yl(1)+noise*randn(sum(y==0), 1)
    
    p = plot(contrasts, y, '.', 'color', c, 'markersize', 10);
    
    running_minimum_x = min([running_minimum_x, min(contrasts)])
end
plot_horizontal_line(.5)
set(gca, 'tickdir', 'out', 'ylim', yl, 'xlim', [running_minimum_x 0], 'clipping', 'on', 'xgrid', 'on', 'ygrid', 'on')
xlabel('log contrast')
ylabel('prop. correct')