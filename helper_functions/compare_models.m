function [score, group_mean, group_sem, MCM_delta, subject_names] = compare_models(models, varargin)

if isfield(models(1).extracted(1), 'waic2')
    MCM = 'waic2';
else
    MCM = 'dic';
end
sort_subjects = false;

fig_type = 'bar'; % 'grid' or 'bar' or 'sum' or '

% BAR OPTIONS
group_gutter=.02;
bar_gutter= 0.001;
show_names = true;
anonymize = true;
fontname = 'Helvetica';
show_model_names = true;
CI = .95;
barcolor = [0 0 0];

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
        try
            score(m,d) = models(m).extracted(d).(MCM);
        end
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

if strcmp(MCM, 'waic2') || strcmp(MCM, 'waic1')
    MCM_name = 'WAIC';
else
    MCM_name = upper(MCM);
end

if strcmp(fig_type, 'grid')
    
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
    
else
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
    
    LL = score'/-2;
    
    if show_names
        if anonymize
            subject_names = strcat('S', strread(num2str(1:nDatasets ), '%s'));
        else
            subject_names = upper({models(1).extracted.name});
        end
    else
        subject_names = [];
    end
    
    if strcmp(fig_type, 'bar')
        [group_mean, group_sem] = mybar(MCM_delta, 'barnames', subject_names, 'show_mean', true, ...
            'group_gutter', group_gutter, 'bar_gutter', bar_gutter, ...
            'mark_grate_ellipse', mark_grate_ellipse, 'bootstrap', true, 'show_mean', true, 'show_errorbox', true);
        
        
        set(gca,'ticklength',[0.018 0.018],'box','off','xtick',(1/nModels/2):(1/nModels):1,'xticklabel',model_names,...
            'xaxislocation','top','fontname', fontname,...%'ytick', round(yl(1),-2):500:round(yl(2),-2), ...
            'fontsize', fontsize, 'xticklabelrotation', 30, 'ygrid', 'on')
        
        if ~show_model_names
            set(gca, 'xticklabel', '')
            set(gca, 'xcolor', 'w')
        end
        
        ylabel(sprintf('%s - %s_{%s}', MCM_name, MCM_name, model_names{ref_model}))
    else
        if any(strcmp(fig_type, {'mean', 'sum'}))
            quantiles = quantile(bootstrp(1e4, eval(sprintf('@%s', fig_type)), MCM_delta'), [.5 - CI/2, .5, .5 + CI/2]);
            [~, sort_idx] = sort(quantiles(2,:), 2)
            quantiles = quantiles(:, sort_idx);
            bar(quantiles(2,:), 'facecolor', barcolor, 'edgecolor', 'none')
            hold on
            errorbar(2:nModels, quantiles(2,2:end), diff(quantiles(1:2,2:end)), diff(quantiles(2:3,2:end)), 'linestyle', 'none', 'linewidth', 2, 'color', [.75 .75 .75])
            yl = get(gca, 'ylim');
            yl(1) = 0;
            ylim(yl);
            ylabel(['\Delta ' MCM_name])
            set(gca, 'xdir', 'reverse')
        else
            [alpha, exp_r, xp, pxp, bor] = spm_BMS(LL);
            
            [bars, sort_idx] = sort(eval(fig_type));
            
            bar(bars, 'facecolor', barcolor, 'edgecolor', 'none')
            
            if strcmp(fig_type, 'exp_r')
                ylabel('model probability for random subject')
            else
                ylabel('probability that model is better than all others')
            end
            ylim([0 1])
            set(gca, 'ytick', 0:.25:1);
        end
        
        view(90,-90);
        set(gca, 'box', 'off', 'tickdir', 'out', 'xticklabel', model_names(sort_idx), 'xtick', 1:nModels)        
    end
    
end