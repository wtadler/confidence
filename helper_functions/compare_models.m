function [score, group_mean, group_sem, MCM_delta, subject_names, quantiles, sort_idx] = compare_models(models, varargin)
[score,group_mean,group_sem,MCM_delta,subject_names,quantiles]=deal([]);

MCM_priority = {'loopsis', 'waic2', 'dic', 'aic'};
for m = 1:length(MCM_priority)
    if isfield(models(1).extracted(1), MCM_priority{m}) && ~isempty(models(1).extracted(1).(MCM_priority{m}))
        MCM = MCM_priority{m};
        break
    end
end

sort_subjects = false;
sort_model = 1;
keep_subjects_sorted = true;

fig_type = 'bar'; % 'grid' or 'bar' or 'sum' or ''

LL_scale = true; % if false, it's on *IC scale

ticklength = .02;

% GRID OPTIONS
mark_best_and_worst = true;
color_switch_threshold = .5; % point in the MCM range where the text color switches from black to white

% BAR/MEAN/SUM OPTIONS
region_color = 'b';
region_alpha = .4;
xticklabelrotation = 30;
fig_orientation = 'horz'; %'vert' or 'horz'

% BAR OPTIONS
group_gutter=.02;
bar_gutter= 0.001;
show_names = true;
anonymize = true;
fontname = 'Helvetica';
show_model_names = true;
CI = .95;
barcolor = [0 0 0];
sort_idx = [];

% MEAN/SUM OPTIONS
bar_type = 'bar'; % 'bar' or 'fill'
fill_gutter = .3;

xy_label_fontsize = 10;
tick_label_fontsize = 10;


mark_grate_ellipse = false;

ref_model = [];
ref_value = [];
normalize_by = [];

model_name_short = true;
model_name_abbrev = true;
model_name_task = true;
model_name_choice = true;

assignopts(who, varargin)

if strcmp(MCM, 'waic')
    MCM = 'waic2';
end

if isempty(normalize_by)
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
end

switch MCM
    case {'waic1', 'waic2'}
        MCM_name = 'WAIC*';
        multiplier = -.5;
    case 'loopsis'
        MCM_name = 'LOO';
        multiplier = 1;
    case 'laplace'
        MCM_name = '-Laplace approximation';
        multiplier = 1; % not sure if this is correct, but we're not really using this MCM
    case 'min_nll'
        MCM_name = 'LL';
        multiplier = -1;
    otherwise % AIC, DIC, etc.
        MCM_name = [upper(MCM) '*'];
        multiplier = -.5;
end

if ~LL_scale
    multiplier = -2*multiplier;
    if any(strfind(MCM_name, '*')) % remove star if it's there
        MCM_name = strrep(MCM_name, '*', '');
    else
        MCM_name = [MCM_name '*']; % add star if not, to indicate that LL is in IC scale
    end
end

nModels = length(models);

% if nModels > 15
%     tick_label_fontsize = tick_label_fontsize-2;
% end

nDatasets = length(models(1).extracted);



score = nan(nModels, nDatasets);
for m = 1:nModels
    for d = 1:nDatasets
        try
            score(m,d) = models(m).extracted(d).(MCM);
        catch
            error(sprintf('MCM %s not found in models struct', MCM));
        end
    end
end

score = real(multiplier*score); % real is for laplace approximation, which sometimes fails and has a small imaginary component

range = max(max(score))-min(min(score));
color_threshold = min(min(score)) + color_switch_threshold * range;

if show_names
    subject_names = {models(1).extracted.name};
end


model_names = rename_models(models, 'short', model_name_short, 'abbrev', model_name_abbrev, 'task', model_name_task, 'choice', model_name_choice);

if strcmp(fig_type, 'grid')
    if sort_subjects
        [~, sort_idx] = sort(mean(score,1));
        score = score(1:nModels, sort_idx);
        if show_names
            subject_names = subject_names(sort_idx);
        end
    end

    imagesc(score);
    set(gca,'looseinset',get(gca,'tightinset')) % get rid of white space
    colorsteps = 256;
    colormap(flipud(gray(colorsteps)));
    
    for m = 1:nModels
        for d = 1:nDatasets
            Lmargin = -.45;
            Tmargin = -.3;
            score_string = num2str(score(m,d),'%.0f');
            t=text(d+Lmargin,m+Tmargin, score_string,'fontsize',xy_label_fontsize,'horizontalalignment','left');
            if score(m,d)>color_threshold, set(t,'color','white'), end
        end
    end
    
    
    pbaspect([nDatasets nModels 1])
    set(gca,'box','on','xaxislocation','top','ytick',1:length(models),'ticklength',[0 0],'xtick',1:nDatasets, 'xticklabels', upper(subject_names),'fontweight','bold')
    set(gca,'yticklabels', model_names)
    xlabel('Subject','interpreter','none')
    
    % c=colorbar;
    % ticks=11000:2000:15000;
    % set(c,'ydir','reverse','box','on','ticklength',.035,'ticks',ticks,'ticklabels',{'11000','13000','15000'});
    
    
    if mark_best_and_worst
        % mark best and worse with checks and crosses
        [~,best_idx]=max(score);
        [~,worst_idx]=min(score);
        
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
    
elseif ~strcmp(fig_type, '')
    switch normalize_by
        case 'best_model'
            if LL_scale
                [~, ref_model] = max(mean(score, 2));
            else
                [~, ref_model] = min(mean(score, 2));
            end
            MCM_delta = bsxfun(@minus, score(ref_model,:), score);
        case 'worst_model'
            if LL_scale
                [~, ref_model] = min(mean(score, 2));
            else
                [~, ref_model] = max(mean(score, 2));
            end
            MCM_delta = bsxfun(@minus, score(ref_model,:), score);
        case 'specific_model'
            MCM_delta = bsxfun(@minus, score(ref_model,:), score);
        case 'specific_value'
            MCM_delta = bsxfun(@minus, ref_value, score);
    end
    if ~LL_scale
        MCM_delta = -MCM_delta;
    end
    
    if show_names
        if anonymize
            subject_names = strcat('O', strread(num2str(1:nDatasets ), '%s'));
        else
            subject_names = upper({models(1).extracted.name});
        end
    else
        subject_names = [];
    end
    
    if LL_scale
        ylabel_str = sprintf('%s_{%s} %s %s', MCM_name, model_names{ref_model}, char(8722), MCM_name);
    else
        ylabel_str = sprintf('%s %s %s_{%s}', MCM_name, char(8722), MCM_name, model_names{ref_model});
    end
    
    if strcmp(fig_type, 'bar')
        if sort_subjects
            if keep_subjects_sorted
                [~, sort_idx]=sort(MCM_delta(sort_model, :), 'descend');
                MCM_delta = MCM_delta(:, sort_idx);
                if show_names
                    subject_names = subject_names(sort_idx);
                end
            else
                MCM_delta = sort(MCM_delta, 2,'descend');
                subject_names = [];
            end
            
        end
        
        [group_mean, group_sem] = mybar(MCM_delta, 'barnames', subject_names, 'show_mean', true, ...
            'group_gutter', group_gutter, 'bar_gutter', bar_gutter, ...
            'mark_grate_ellipse', mark_grate_ellipse, 'bootstrap', true, ...
            'show_mean', true, 'show_errorbox', true,...
            'fontsize', tick_label_fontsize, 'region_color', region_color,...
            'region_alpha', region_alpha, 'fig_orientation', fig_orientation,...
            'CI', CI);
        
        
        set(gca,'ticklength', [ticklength, ticklength],'box','off','xtick',(1/nModels/2):(1/nModels):1,'xticklabel',model_names,...
            'xaxislocation','top','fontname', fontname,...%'ytick', round(yl(1),-2):500:round(yl(2),-2), ...
            'fontsize', tick_label_fontsize, 'xticklabelrotation', 30, 'ygrid', 'on')
        
        if ~show_model_names
            set(gca, 'xticklabel', '')
            set(gca, 'xcolor', 'w')
        end
        
        ylabel(ylabel_str, 'fontsize', xy_label_fontsize)
    else
        if any(strcmp(fig_type, {'mean', 'sum'}))
            quantiles = quantile(bootstrp(1e4, eval(sprintf('@%s', fig_type)), MCM_delta'), [.5 - CI/2, .5, .5 + CI/2]);
            
            if isempty(sort_idx)
                [~, sort_idx] = sort(quantiles(2,:), 2);
            end
            
            plotbars = setdiff(1:nModels, find(sort_idx==ref_model));
            quantiles = quantiles(:, sort_idx);
            hold on
            
            if strcmp(fig_type, 'sum')
                ylabel_str = sprintf('\\Sigma(%s)', ylabel_str);
            elseif strcmp(fig_type,'mean')
                ylabel_str = sprintf('\\bf{E}\\rm{(%s)}', ylabel_str);
            end
            ylabel(ylabel_str, 'fontsize', xy_label_fontsize)
            
            medians = quantiles(2,:);
            
        else
            [alpha, exp_r, xp, pxp, bor] = spm_BMS(score');
            
            if isempty(sort_idx)
                [~, sort_idx] = sort(eval(fig_type), 2, 'descend');
            end
            
            medians = eval(fig_type);
            medians = medians(:, sort_idx);
            
        end
        
        
        if strcmp(bar_type, 'bar')
            bar(medians, 'facecolor', barcolor, 'edgecolor', 'none')
            if any(strcmp(fig_type, {'mean', 'sum'}))
                errorbar(plotbars, quantiles(2,plotbars), diff(quantiles(1:2,plotbars)), diff(quantiles(2:3,plotbars)), 'linestyle', 'none', 'linewidth', 2, 'color', [.75 .75 .75])
            end
        elseif strcmp(bar_type, 'fill')
            for m = 1:nModels
                startpt = m-.5+fill_gutter/2;
                endpt = startpt+1-fill_gutter;
                
                plot([startpt endpt], [medians(m) medians(m)], '-', 'color', region_color, 'linewidth', 3);
                
                error_box = [quantiles(1, m), quantiles(1, m), quantiles(3, m), quantiles(3, m)];
                f = fill([startpt endpt endpt startpt], error_box, region_color,...
                    'edgecolor', 'none', 'facealpha', region_alpha);
            end
        end
        
        if ~any(strcmp(fig_type, {'mean', 'sum'}))
            if strcmp(fig_type, 'exp_r')
                ylabel('expected posterior probability of model')
            elseif strcmp(fig_type, 'pxp')
                ylabel('protected exceedance probability')
            elseif strcmp(fig_type, 'xp')
                ylabel('exceedance probability')
            end
            ylim([0 1])
            set(gca, 'ytick', 0:.25:1);
        end
        % plot connecting line
%         yl = get(gca, 'ylim');
%         for m = 1:nModels
%             plot([m m], [yl(1) medians(m)], 'k-', 'linewidth', 2)
%         end
        
        hline=plot_horizontal_line(0,'k-', 'linewidth', 1);
        uistack(hline,'bottom')
        
        set(gca, 'box', 'off', 'tickdir', 'out', 'xticklabel', model_names(sort_idx), ...
            'xtick', 1:nModels, 'xlim', [0 nModels+1], 'fontsize', tick_label_fontsize,...
            'xdir','reverse', 'ticklength', [ticklength, ticklength], 'ygrid', 'on',...
            'xlim', .5+[0 nModels])
        if strcmp(fig_orientation, 'horz')
            set(gca, 'view', [90 -90])
        elseif strcmp(fig_orientation, 'vert')
            set(gca, 'ydir','normal','xaxislocation','top','xticklabelrotation',xticklabelrotation)
        end
        
        set(gcf, 'position', [184 660 261 202])%[184 490 466 372])
    end
    
end