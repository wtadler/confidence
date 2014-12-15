function m=model_comparison
close all

stats = {'aic','bic','aicc','laplace','min_sum_test_nll'}; % 'aic' or 'bic' or 'aicc' or 'min_nll' or 'laplace' or 'min_sum_test_nll'

for i = 1:length(stats)
    stat = stats{i};
cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations')
m = struct;
% 
% % models with old style names
% load('opt_hpc_dgnoise_2000opts_1000sets.mat') % might need to redo the d noise models. ub for sigma_d was at 1.
% m(1).extracted = model.extracted;
% m(1).name = 'optP d noise conf';
% 
% load('opt_hpc_Chat_d_noise_600opts_1000sets.mat')
% m(2).extracted = model.extracted;
% m(2).name = 'optP d noise';
% 
% load('1000opts_hpc_real_all_except_d_noise') % fixed family is bad here. re-do the fixed ones
% m = loadup(m, opt_models([1:4]), model([1:4])); %just [1 3] before
% 
% load('400opts_lin_quad')
% m = loadup(m, opt_models, st.model);
% 
% load('200opts_hpc_real_opt_optconf_d_noise')
% m = loadup(m, opt_models(1), model(1));
% 
% load('1000opts_real_hpc_lin_quad_conf.mat')
% m = loadup(m, opt_models([1 3]), model([1 3]));
% 
% load('2000opts_fixed') % fixed partial, fixed no partial, fixed. this is better score for choice, but not for conf. why? keep it. more up to date.
% m = loadup(m, opt_models([1 3]), gen.opt([1 3]));
% 
% load('1000opts_asym_partial_lapse_and_no_partial_lapse')
% m = loadup(m, opt_models, gen.opt); % was just 1 before
% 
% load('1000opts asym bounds d noise conf')
% m = loadup(m, opt_models, model);
% 
% %load('2000opts_lin2_conf')
% %m = loadup(m, opt_models, gen.opt(1));
% 
% %load('2000opts_quad2_conf')
% %m = loadup(m, opt_models, gen.opt(1));
% 
% %load('multilapse')
% %m = loadup(m, opt_models([2 3]), gen.opt([2 3]));
% 
%load('new_quad2_lin2_multilapse_partial_no'); % nondecreasing lambda_i...
% m = loadup(m, opt_models, model);
% 
% load('100opts_opt_asym_bounds_d_noise_partial_lapse_conf');
% m = loadup(m, opt_models, model);
%load('300opts_multilapse_battle_monotonic_no_d_noise');% decreasing lamda_i
%m=loadup(m,opt_models,gen.opt);
% % % 
% % % load('20000opts_hpc_good_models_conf_no_d_noise')
% % % m=loadup(m, opt_models, model)
% % % 
% % % load('750opts_hpc_good_models_conf_d_noise')
% % % m=loadup(m, opt_models, model);
% % % 
% % % %load('20000opts_hpc_good_models_choice_no_d_noise')
% % % %m=loadup(m, opt_models, model);
% % % 
% % % %load('1000opts_hpc_good_models_choice_d_noise') % why are these worse?
% % % %m=loadup(m, opt_models, model);
% % % load('20000opts_lin_quad_fixed_repeat')
% % % m=loadup(m, opt_models, model);
% % % 
% % % load('200opts_opt_asym_optP_d_noise_repeat')
% % % m=loadup(m, opt_models, model);
% % % 
% % % 
% % % [~,sort_idx] = sort({m(:).name});
% % % m = m(sort_idx); % sort alphabetically
% % % 
% % % fixed_idx = find(~cellfun(@isempty, regexp({m.name}, 'fixed')));
% % % m = m([setdiff(1:length(m), fixed_idx) fixed_idx]); % push fixed to the back
% % % 
% % % for i = 1 : length(m)
% % %     i
% % %     m(i).name
% % % end
% % % 
% % % conf_models = ~cellfun(@isempty, regexp({m.name}, 'conf$'));
% % % %conf_models = ~cellfun(@isempty, regexp({m.name},'(?<!no partial lapse.*)conf$')); % leave out the no partial lapse models, they're all bad, it seems.
% % % choice_models = cellfun(@isempty, regexp({m.name}, 'conf$'));
% % % m=m([9 10 4 5 17 18 1 2 14 15])
% load('v2_no_noise.mat')
% m=loadup(m,opt_models,model)
% 
% load('v2_noise.mat')
% m=loadup(m,opt_models,model)
% m.name
% m=m([3 5 4 1 2]); % good for colorful bars lined up fixed, optp, opt a, lin, quad
% %m=m([1 5 4 3 2]); %Fixed, Quad, Lin, Opt A, Opt good for the weird grid thing. 
% m.name

load('v2_small.mat')
m=loadup(m,{model.name},model)
%m=m([4 1 2 3]);
%m=m([1 2 3 4]); % fixed lin quad opt
m=m([1 4 2 3]); % fixed opt lin quad

save mctest1.mat

if strcmp(stat,'min_sum_test_nll')
    load('v2_crossval.mat')
    m=loadup(m,{model.name},model)
    %m=model([1 2 3 4]) % fixed lin quad opt
    m=model([1 4 2 3]) % fixed opt lin quad
end

figure
set(gcf,'position',[54 261 719 545]);

%plotstuff(m(conf_models), 1, 1, 1, stat)
plotstuff(m, 1, 1, 1, stat)
savefig(sprintf('%s.fig',stat))
%plotstuff(m(choice_models), 2, 1, 2, stat)
%title(sprintf('choice model %s scores (lower is better)', upper(stat)))
end
end



function m = loadup(m, opt_models, model)
if isempty(fieldnames(m))
    l = 0;
else
    l = length(m); % take out -1 at some point
end
for opt_model_id = 1 : length(opt_models)
    m(opt_model_id+l).extracted = model(opt_model_id).extracted;
    m(opt_model_id+l).name = strrep(opt_models{opt_model_id}, '_', ' ');
end
end

function plotstuff(m, nRows, nCols, index, stat)
%% BAR PLOT
subplot(nRows, nCols, index)
nModels = length(m);
nDatasets = length(m(1).extracted);
barinfo = zeros(nDatasets, nModels);
for i = 1 : nModels;
    barinfo(:,i) = real([m(i).extracted.(stat)]');
end
if strcmp(stat,'min_sum_test_nll')
    barinfo = -barinfo;
end

% set in reference to some model
% ref_model = 2;
% for i = 1 : nDatasets
%     barinfo(i,:) = - barinfo_raw(i,:) + barinfo_raw(i,ref_model);
% end
%

bh = barh(fliplr(barinfo));
%set(bh,'edgecolor','none')
a = min(min(barinfo));
b = max(max(barinfo));
range = b - a;
xlim([a - range / 8, b + range / 4]);

%lh = legend(m.name);
lh = legend(bh([4 3 2 1]), 'Fixed','Optimal','Lin','Quad')
legend('boxoff')
set(lh,'location','eastoutside')
ylabel('Subjects')

% make these generate smarter
%set(bh(1),'facecolor',[0 0 1])
%set(bh(2),'facecolor',[0 0 .8])
%set(bh(3),'facecolor',[0 0 .6])
%set(bh(4),'facecolor',[1 0 0])
%set(bh(5),'facecolor',[.6 0 0])
save mctest1.mat
% bc = get(bh, 'children');
% bp = get([bc{:}], 'ydata');
% barheight = diff(bp{1}([1,3]));

% for group = 1 : length(bh);
%     for dataset = 1 : nDatasets;
%         %th=text(bp{group}(1, dataset) + barwidth/2, m(group).extracted(dataset).(stat), sprintf('%.0f', m(group).extracted(dataset).(stat)));
%         th=text(barinfo(dataset,5-group), bp{group}(1, dataset) + barheight/2, sprintf('%.0f', barinfo(dataset,5-group)));
%         tp=get(th,'position');
%         if barinfo(dataset,5-group) < 0
%             set(th,'fontweight','bold','color','black', 'horizontalalignment', 'right','position',tp-[10 0 0])
%         else
%             set(th,'fontweight','bold','color','black', 'horizontalalignment', 'left','position',tp+[10 0 0])
%         end
% 
%     end
% end
title(sprintf('%s scores', upper(stat)))
return

save mctest.mat
%% HEATMAP
close all
figure
set(gcf,'position',[440 322 402 476])
set(0,'defaultaxesfontsize',12,'defaulttextinterpreter','none')

%dataset_by_model = barinfo_raw./repmat(sum(barinfo_raw,2),1,size(barinfo_raw,2));
%imagesc(dataset_by_model)
nDatasets = size(barinfo_raw,1);
nModels = size(barinfo_raw,2);

% this is a nice idea, but it's better to be consistent across plots.
% if strcmp(stat, 'laplace') | strcmp(stat, 'min_sum_test_nll')
%     'hello'
%     [~,idx]=sort(sum(barinfo_raw,2), 1, 'descend')
% else
%     [~,idx]=sort(sum(barinfo_raw,2), 1, 'ascend')
% end
idx = [7 5 8 4 2 3 10 1 6 9 11]';

barinfo_sort = barinfo_raw(idx,:);
i=imagesc(barinfo_sort);
pbaspect([nModels nDatasets 1]);
%mnames={'Optimal','Fixed','Lin.','Quad.'};
mnames ={'Fix.','Bayes.','Lin.','Quad.'};
set(0,'DefaultTextInterpreter', 'latex');
c=colorbar;
set(c,'box','on','ticklength',[.05 0],'linewidth',1);
%set(get(c,'ylabel'),'string','log marginal likelihood','interpreter','none','rot',-90,'verticalalignment','bottom')
set(gca,'box','on','xaxislocation','top','xtick',1:4,'xticklabel',mnames,'ticklength',[0 0],'linewidth',1)
xlabel('Model','interpreter','none')
ylabel('Subject','interpreter','none')
colorsteps = 256;
rrr = [1 0 0];
ggg = [0 1 .1];
greenredmap = [linspace(ggg(1),rrr(1),colorsteps)' linspace(ggg(2),rrr(2),colorsteps)' linspace(ggg(3),rrr(3),colorsteps)'];
if strcmp(stat, 'laplace') | strcmp(stat, 'min_sum_test_nll')
    colormap(gray(colorsteps))
else
    colormap(flipud(gray(colorsteps)))
end

if strcmp(stat,'laplace') | strcmp(stat, 'min_sum_test_nll')
    set(c,'ydir','normal')
else
    set(c,'ydir','reverse')
end


hold on
%[~,min_idx]=min(dataset_by_model');
%[~,max_idx]=max(dataset_by_model');
if strcmp(stat, 'laplace') | strcmp(stat, 'min_sum_test_nll')
    [~,best_idx]=max(barinfo_sort');
    [~,worst_idx]=min(barinfo_sort');
else
    [~,best_idx]=min(barinfo_sort');
    [~,worst_idx]=max(barinfo_sort');
end

%
%imx=imread('/Users/will/Google Drive/Will - Confidence/Presentations/1YT/checks/x.png','BackGroundColor',[1 1 .95]);
%imv=imread('/Users/will/Google Drive/Will - Confidence/Presentations/1YT/checks/v.png','BackGroundColor',[1 1 .95]);
%maskx=bsxfun(@eq,imx,reshape([255 255 242],1,1,3));
%maskv=bsxfun(@eq,imv,reshape([255 255 242],1,1,3));
%image(imx,'alphadata',1-double(all(mask,3)));

%l1=plot(min_idx,1:nDatasets,'o','color',[.3 .87 .3],'markersize',20,'linewidth',4);
%l2=plot(max_idx,1:nDatasets,'x','color',[.45 0 0],'markersize',20,'linewidth',4);

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
    crosses(subject) = patch(worst_idx(subject)-.5+xpatchx, subject-.5+xpatchy,'k');
    checks(subject)  = patch(best_idx(subject)-.5 +vpatchx, subject-.5+vpatchy,'k');
end
set(crosses,'facecolor',[.6 0 0],'edgecolor','w','linewidth',1)
set(checks,'facecolor',[0 .6 0],'edgecolor','w','linewidth',1)
if strcmp(stat, 'laplace')
    title({'Log marginal likelihood'; 'Laplace approximation'},'interpreter','none','interpreter','none','fontsize',15)
elseif strcmp(stat,'min_sum_test_nll')
    title({'Log likelihood';'10-fold cross-validation'},'interpreter','none','fontsize',15)
else
    title(sprintf('%s score',upper(stat)),'interpreter','none','fontsize',15)
end
export_fig(sprintf('model_comparison_%s.pdf',upper(stat)))
return
%% MEAN MODEL SCORES WITH SEM
figure
meanscores = mean(barinfo);
semscores = std(barinfo)/sqrt(nDatasets);
%bh=barh(fliplr(meanscores));
bh=barh(repmat(fliplr(meanscores),2,1));
ylim([.5 1.5])
set(gca,'ylim',[.5 1.5],'ytick',[])
lh = legend(bh([5 4 3 2 1]), 'Fixed','Lin','Quad','Opt','Opt Asym')
legend('boxoff')
set(lh,'location','eastoutside')
hold on
for i = 1:5
    chi = get(bh(6-i),'children');
    yd = get(chi,'ydata');
    plot([meanscores(i)-semscores(i) meanscores(i)+semscores(i)], [mean(yd([1 3])) mean(yd([1 3]))],'-','color',[.8 .8 .8], 'linewidth',3)
end
title(sprintf('\\Delta %s scores', upper(stat)))

%% CONFUSING PAIRWISE MODEL COMPARISON
%reorder everything
m=m([1 3 2 4 5]); %Fixed, Quad, Lin, Opt A, Opt
barinfo_raw = [barinfo_raw(:,1) barinfo_raw(:,3) barinfo_raw(:,2) barinfo_raw(:,4) barinfo_raw(:,5)];

nModels = length(m);
titles = {'Fixed','Quad.','Lin.','Opt. Asym.', 'Optimal'}
[mmm,sss]=deal(NaN(nModels-1,nModels-1));
for i = 1:nModels-1;
    i
    for j = 1:nModels-i;
        mmm(i,j) = mean(barinfo_raw(:,nModels+1-j)-barinfo_raw(:,i));
        sss(i,j) = std(barinfo_raw(:,nModels+1-j)-barinfo_raw(:,i))./sqrt(nDatasets);
    end
end

big = max(max(mmm+sss));
lil = min(min(mmm-sss));
f=figure;
set(0,'defaultaxesfontsize',15)
for i = 1:nModels-1
    for j = 1:nModels-i
        subplot_tight(nModels-1,nModels-1, (nModels-1)*(i-1) + j)
        bar(mmm(i,j),'k')
        hold on
        plot([1 1],[mmm(i,j) - sss(i,j) mmm(i,j) + sss(i,j)],'-','linewidth',3,'color',[.8 .8 .8])
        set(gca,'xtick',[],'ylim',[lil big],'visible','on','box','on','xcolor','k','ticklength',[.03 .03])
        if i == 1
            title(sprintf('+%s',titles{nModels+1-j}));
        end
        if j == 1
            yy=ylabel(sprintf('-%s',titles{i}));
            set(yy,'rot',0,'horizontalalignment','right')
        else
            set(gca,'yticklabel','')
        end
        
    end
end
sl=suplabel(sprintf('\\Delta %s scores',upper(stat)),'t')
tightfig




end