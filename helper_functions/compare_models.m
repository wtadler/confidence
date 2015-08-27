function score=compare_models(models, varargin)
% make model comparison grid
figure

MCM = 'dic';
sort_subjects = false;

fig_type = 'grid'; % 'grid' or 'bar' or 'mean'

% BAR OPTIONS
inter_group_gutter=.2;
intra_group_gutter= 0.02;
show_names = true;
fontname = 'Helvetica Neue';

% GRID OPTIONS
mark_best_and_worst = true;
color_switch_threshold = .5; % point in the MCM range where the text color switches from black to white

fontsize = 10;    

mark_grate_ellipse = false;
assignopts(who, varargin)

if strcmp(MCM, 'laplace')
    flip_sign = true; % if higher number indicates better fit
else
    flip_sign = false; % if lower number indicates better fit (most MCMs)
end

nModels = length(models)
nDatasets = length(models(1).extracted);

score = nan(nModels, nDatasets);
for m = 1:nModels
    for d = 1:nDatasets
        m
        d
        score(m,d) = models(m).extracted(d).(MCM);
    end
end

if flip_sign
    score = -score;
end

if ~isreal(score)
    score = real(score);
end

range = max(max(score))-min(min(score));
color_threshold = min(min(score)) + color_switch_threshold * range;

subject_names = {models(1).extracted.name};

if sort_subjects
    [~, sort_idx] = sort(mean(score,1));
    score = score(1:nModels, sort_idx);
    subject_names = subject_names(sort_idx);
end

model_names = rename_models({models.name});

switch fig_type
    case 'grid'
        
        i = imagesc(score);
        set(gca,'looseinset',get(gca,'tightinset')) % get rid of white space
        colorsteps = 256;
        colormap(flipud(gray(colorsteps)));
        
        for m = 1:nModels
            for d = 1:nDatasets
                Lmargin = -.45;
                Tmargin = -.3;
                if ~flip_sign
                    score_string = num2str(score(m,d),'%.0f');
                else
                    score_string = num2str(-score(m,d), '%.0f');
                end
                t=text(d+Lmargin,m+Tmargin, score_string,'fontsize',fontsize,'horizontalalignment','left');
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
            set(crosses, 'facecolor', [.6 0 0], 'edgecolor', 'w', 'linewidth', 1)
            set(checks,  'facecolor', [0 .6 0], 'edgecolor', 'w', 'linewidth', 1)
        end

    case 'bar'
                
        [~,best_model] = min(mean(score, 2));
        
        MCM_delta = bsxfun(@minus, score, score(best_model,:));
        
        
        
        if show_names
            mybar(-MCM_delta, 'barnames', upper({models(1).extracted.name}), 'show_mean', true, ...
                'inter_group_gutter', inter_group_gutter, 'intra_group_gutter', intra_group_gutter, ...
                'mark_grate_ellipse', mark_grate_ellipse)
        else
            mybar(-MCM_delta, 'show_mean', true, ...
                'inter_group_gutter', inter_group_gutter, 'intra_group_gutter', intra_group_gutter, ...
                'mark_grate_ellipse', mark_grate_ellipse)
        end

        yl = get(gca,'ylim')
        set(gca,'ticklength',[0 0],'box','off','xtick',1:nModels,'xticklabel',model_names,...
            'xaxislocation','top','fontweight','bold','fontname', fontname,'ytick', round(yl(1),-2):500:round(yl(2),-2), ...
            'fontsize', fontsize, 'xticklabelrotation', 30)
        
        if strcmp(MCM, 'waic2') || strcmp(MCM, 'waic1')
            MCM_name = 'WAIC';
        else
            MCM_name = upper(MCM);
        end
        
        ylabel(sprintf('%s_{%s} - %s', MCM_name, model_names{best_model}, MCM_name))
end