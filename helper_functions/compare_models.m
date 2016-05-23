function [score, group_mean, group_sem, MCM_delta, subject_names] = compare_models(models, varargin)

if isfield(models(1).extracted(1), 'waic2')
    MCM = 'waic2';
else
    MCM = 'dic';
end
sort_subjects = false;

fig_type = 'bar'; % 'grid' or 'bar'

% BAR OPTIONS
group_gutter=.02;
bar_gutter= 0.001;
show_names = true;
    anonymize = true;
fontname = 'Helvetica';
show_model_names = true;

% GRID OPTIONS
mark_best_and_worst = true;
color_switch_threshold = .5; % point in the MCM range where the text color switches from black to white

fontsize = 10;

mark_grate_ellipse = false;

ref_model = [];
ref_value = [];

assignopts(who, varargin)

if isempty(ref_model)
    if isempty(ref_value)
        normalize_by = 'best_model';
    else
        normalize_by = 'specific_value';
    end
else
    if isempty(ref_value)
        normalize_by = 'specific_model';
    else
        error('ref_model and ref_value are both set. Pick one!')
    end
end
    

if strcmp(MCM, 'laplace')
    flip_sign = true; % if higher number indicates better fit
else
    flip_sign = false; % if lower number indicates better fit (most MCMs)
end

nModels = length(models);
nDatasets = length(models(1).extracted);

score = nan(nModels, nDatasets);
for m = 1:nModels
    for d = 1:nDatasets
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
        
        imagesc(score);
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
        switch normalize_by
            case 'best_model'
                [~, ref_model] = min(mean(score, 2));
                MCM_delta = bsxfun(@minus, score(ref_model,:), score);
            case 'specific_model'
                MCM_delta = bsxfun(@minus, score(ref_model,:), score);
            case 'specific_value'
                MCM_delta = bsxfun(@minus, ref_value, score);
        end
        
        MCM_delta = -MCM_delta;
        
        if show_names
            if anonymize
                subject_names = strcat('S', strread(num2str(1:nDatasets ), '%s'));
            else
                subject_names = upper({models(1).extracted.name});
            end
        else
            subject_names = [];
        end
        
        [group_mean, group_sem] = mybar(MCM_delta, 'barnames', subject_names, 'show_mean', true, ...
            'group_gutter', group_gutter, 'bar_gutter', bar_gutter, ...
            'mark_grate_ellipse', mark_grate_ellipse, 'bootstrap', false);
        
        
        set(gca,'ticklength',[0.018 0.018],'box','off','xtick',(1/nModels/2):(1/nModels):1,'xticklabel',model_names,...
            'xaxislocation','top','fontname', fontname,...%'ytick', round(yl(1),-2):500:round(yl(2),-2), ...
            'fontsize', fontsize, 'xticklabelrotation', 30, 'ygrid', 'on')
        
        if ~show_model_names
            set(gca, 'xticklabel', '')
            set(gca, 'xcolor', 'w')
        end
        
        if strcmp(MCM, 'waic2') || strcmp(MCM, 'waic1')
            MCM_name = 'WAIC';
        else
            MCM_name = upper(MCM);
        end
        
        ylabel(sprintf('%s - %s_{%s}', MCM_name, MCM_name, model_names{ref_model}))
end