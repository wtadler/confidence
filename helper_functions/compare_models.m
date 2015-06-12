function compare_models(models, varargin)
% make model comparison grid
figure

MCM = 'dic';
sort_subjects = true;
fontsize = 10;
mark_best_and_worst = false;
color_switch_threshold = .5;
assignopts(who, varargin)

nModels = length(models)
nDatasets = length(models(1).extracted);

score = nan(nModels, nDatasets);
for m = 1:nModels
    for d = 1:nDatasets
        score(m,d) = models(m).extracted(d).(MCM);
    end
end

range = max(max(score))-min(min(score));
color_threshold = min(min(score)) + color_switch_threshold * range;

subject_names = {models(1).extracted.name};

if sort_subjects
    [~, sort_idx] = sort(mean(score,1));
    score = score(1:nModels, sort_idx);
    subject_names = subject_names(sort_idx);
end

model_names = strrep({models.name}, '_', ' ');

i = imagesc(score);
set(gca,'looseinset',get(gca,'tightinset')) % get rid of white space
colorsteps = 256;
colormap(flipud(gray(colorsteps)));

for m = 1:nModels
    for d = 1:nDatasets
        Lmargin = -.45;
        Tmargin = -.3;
        t=text(d+Lmargin,m+Tmargin,num2str(score(m,d),'%.0f'),'fontsize',fontsize,'horizontalalignment','left');
        if score(m,d)>color_threshold, set(t,'color','white'), end
    end
end


pbaspect([nDatasets nModels 1])
set(gca,'box','on','xaxislocation','top','ytick',1:length(models),'ticklength',[0 0],'linewidth',1,'xtick',1:nDatasets, 'xticklabels', upper(subject_names),'fontweight','bold')
set(gca,'yticklabels', model_names)
xlabel('Subject','interpreter','none')

% c=colorbar;
% ticks=11000:2000:15000;
% set(c,'ydir','reverse','box','on','ticklength',.035,'linewidth',1,'ticks',ticks,'ticklabels',{'11000','13000','15000'});


if mark_best_and_worst
    % mark best and worse with checks and crosses
    [~,best_idx]=min(score);
    [~,worst_idx]=max(score);
    
    divisor = 60; % set size of crosses/checks
    
    xpatchx=[15 5 11 21 31 37 27 37 31 21 11 5]/divisor;
    margin = (1-max(xpatchx)+min(xpatchx))/2;
    shift = margin-min(xpatchx);
    xpatchx=xpatchx+shift;
    xpatchy=[21 11 5 15 5 11 21 31 37 27 37 31]/divisor+shift;
    
    vpatchx=[16 3 9 16 32 39]/divisor;
    margin= (1-max(vpatchx)+min(vpatchx))/2;
    shift = margin-min(vpatchx);
    vpatchx=vpatchx+shift;
    vpatchy=[39 26 20 26 10 17]/divisor;
    margin= (1-max(vpatchy)+min(vpatchy))/2;
    shift = margin-min(vpatchy);
    vpatchy=vpatchy+shift;
    
    crosses = zeros(1,nDatasets);
    checks = zeros(1,nDatasets);
    for subject = 1:nDatasets
        crosses(subject) = patch(subject-.5+xpatchx, worst_idx(subject)-.5+xpatchy,'k');
        checks(subject)  = patch(subject-.5 +vpatchx, best_idx(subject)-.5+vpatchy,'k');
    end
    set(crosses,'facecolor',[.6 0 0],'edgecolor','w','linewidth',1)
    set(checks,'facecolor',[0 .6 0],'edgecolor','w','linewidth',1)
end
