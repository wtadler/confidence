load('/Users/bluec/Documents/GitHub/confidence/data/backup/willstaircase3.mat')
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

%Most computationally efficient version; plots error bars for each bin:
    yy= zeros(1,nBins-1);
    plotpoints = zeros(1,nBins-1);
    %Plot each of the first four bins individually:
    for i = 1:length(yy)
        yy(i) = mean(outcome(ind==i));
        plotpoints(i) = (binranges(i)+binranges(i+1))/2;
        a(i) = sum(outcome(ind==i)==1);
        b(i) = sum(outcome(ind==i)==0);
    end
   %Combine values for the final two bins:
   if isnan(yy(end)) == 1
       yy(end) = mean(outcome(ind==nBins)); %Resets NaN resulting from mean calculation for an empty bin
       plotpoints(end) = binranges(end); %Reset the plotting point to reflect only the bin with data
   else
       yy(end) = (yy(end)+mean(outcome(ind==nBins)))/2;
   end
    a(end) = (a(end)+sum(outcome(ind==nBins)==1));
    b(end) = (b(end)+sum(outcome(ind==nBins)==0));
   %Get variance and SD (error bar length)
    var = zeros(1,length(yy));
    EB = zeros(1,length(yy));
    for i = 1:length(yy)
        var(i) = (a(i)*b(i))/((a(i)+b(i))^2*(a(i)+b(i)+1));% variance calc for each binned point
        EB(i) = sqrt(var(i)); %standard deviations (used as the length of each error bar)  
    end
    %And finally plot everything:
    p = errorbar(plotpoints, yy, EB, '.', 'color', c, 'markersize', 12);

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
    y = tab.data(:, 2);
    y(y==1) = yl(2);
    y(y==0) = yl(1);
    n = length(contrasts);
    noise = .005;
    p = plot(contrasts+noise*randn(n, 1), y+noise*randn(n, 1), '.', 'color', c, 'markersize', 20)
    
end

set(gca, 'tickdir', 'out', 'ylim', yl, 'xlim', xl, 'clipping', 'off')
xlabel('log contrast')
ylabel('% correct')