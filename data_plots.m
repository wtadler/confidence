clear all
tstart=tic;
tic

cdsandbox
%cd('~/Ma lab/Confidence theory/analysis plots/')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% LOAD MODEL FIT PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
show_fit = true;
if show_fit
    cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations/v3')
    load('v3_combined_and_cleaned_POSTER_DATA.mat')
    plot_model = 1 : length(opt_models); % can make figs for each model or just one
else
    plot_model = 1;
end

%contrasts = exp(-4:.5:-1.5);
contrasts = exp(linspace(-5.5,-2,6));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% PLOTTING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set(0,'DefaultLineLineWidth',1)
set(0,'DefaultLineLineWidth','remove')

nBins = 13; %13 for cosyne poster
% [real_bins_rt, real_axis_rt] = bin_generator(n_bins_real,'binstyle','rt');

ori_labels = [-16 -8 -4 -2 -1 0 1 2 4 8 16]; % make sure that this only has nBins entries or fewer
ori_label_bin_value = interp1(axis, 1:nBins, ori_labels);
% o_bound = [1 nBins]; % does this make any sense?????
o_bound = [-20 20];


show_fit = true;
n_samples = 2160; % 1e7
% n_bins_model = nBins; %45
% [model_bins, model_axis] = bin_generator(n_bins_model);

% EVERYTHING BELOW THIS needs to be a separate function, to make plotting easier.

nPlotSamples = 2; % for MCMC fits. number of fake datasets to generate, from each subject. 100 might be enough. used 100 for cosyne poster

nHyperplots = 1000; % number of times to take a random fake dataset from each subject. has only small effect on computation time.
%
%gen_stat = 'Chat'; % plot choice and % correct
% gen_stat = 'g'; % plot conf and % correct
gen_stat = 'resp'; % plot choice/confidence response and % correct

ms = 12; %marker size


set(0,'defaultaxesbox','off','defaultaxesfontsize',10,'defaultlinemarkersize',ms,'defaultlinelinewidth',2)

cdsandbox


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% LOAD REAL DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

streal.A = compile_data('datadir','/Users/will/Google Drive/Will - Confidence/Data/v3/taskA')
streal.B = compile_data('datadir','/Users/will/Google Drive/Will - Confidence/Data/v3/taskB')
%streal = st; % use this if you've generated fake data in optimize

if length(streal.A.data)==length(streal.B.data) % sanity check, not foolproof.
    nDatasets = length(streal.A.data);
else
    error('uh oh')
end
data_tasks = {'A','B'}; % tasks for which we have data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% ANALYZE REAL DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for task = 1:length(data_tasks)
    [bins, axis] = bin_generator(nBins, 'task', data_tasks{task});

    % analyze individual data
    for dataset = 1 : nDatasets
        [streal.(data_tasks{task}).data(dataset).stats, streal.(data_tasks{task}).data(dataset).sorted_raw] = indiv_analysis_fcn(streal.(data_tasks{task}).data(dataset).raw, bins);
    end
    
    % group summary stats
    streal.(data_tasks{task}).sumstats = sumstats_fcn(streal.(data_tasks{task}).data);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% GENERATE/ANALYZE FAKE DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for model_id = plot_model;
    for subject = 1:length(model(model_id).extracted)
        model(model_id).extracted(subject).fake = dataset_generator(model(model_id), ...
            model(model_id).extracted(subject).p, nPlotSamples, ...
            'nBins', nBins, 'raw_A', streal.A.data(subject).raw, 'raw_B', streal.B.data(subject).raw);
    end
    
    model(model_id).fake_sumstats = hyperplot(model(model_id), nHyperplots);
end

%%

cd('/Users/will/Google Drive/Will - Confidence/Presentations/cosyne')
load cosyne_poster_hyperplots

%% MODEL FITS FOR ALL SUBJECTS

set(0,'defaultaxesfontsize', 12, 'defaultaxesfontname', 'Helvetica Neue')
contrast_colors = [10 94 0;...
    181 172 69;...
    189 189 189]...
    /256;
yl = [0 1;1 8;1 4;0 1];
outputs = {'tf','resp','g','Chat'};
x = [1:nBins fliplr(1:nBins)];
alpha = .7;
models = [5 4 3 1 2];
model_names = {'Linear','Fixed','Bayes_{Free}','Bayes_{Sym}','Bayes_{Sym,Joint}'};

figure(1)
clf
active_contrasts = [2 4 6];

for o = 2
    for m = 1:5
        model_id = models(m);
        for t = 1:2
            task_id = tasks{t};
            
            figure(1)
            hyperplot = model(model_id).fake_sumstats.(tasks{t});
            
            tight_subplot(3,6,t+1,1+m,[.042 .016]);
            loopvar = 0;
            for c=active_contrasts
                loopvar = loopvar+1;
                errorbar(1:nBins, streal.(task_id).sumstats.all.mean.(outputs{o})(c,:), streal.(task_id).sumstats.all.edgar_sem.(outputs{o})(c,:), 'color',contrast_colors(loopvar,:),'linewidth',1.5);
                hold on
                
                y = [hyperplot.mean.(outputs{o})(c,:)        + hyperplot.std.(outputs{o})(c,:), ...
                    fliplr(hyperplot.mean.(outputs{o})(c,:)) - hyperplot.std.(outputs{o})(c,:)];
                f = fill(x,y,contrast_colors(loopvar,:));
                set(f,'edgecolor','none','facealpha',alpha);
                
            end
            
            plot([0 nBins+1], [4.5 4.5],'k', 'linewidth', 1)
            set(gca,'xtick', ori_label_bin_value,'xticklabel',ori_labels,'ylim',yl(o,:),'xlim',[0 nBins+1],'box','off','tickdir','out','ticklength',[.025 .2],'yticklabel','', 'ydir', 'reverse');
            %
            %             % model fits as fill
            if t==1
                % %                 title(model_names{model_id})
                set(gca,'xticklabel','');
                % %                 set(gca,'yticklabel','')
                %
                %             elseif t==2
                % %                 xlabel('stimulus orientation')
                %                 if m ==1
                % %                     ylabel('category and confidence response, Task B')
                %                 else
                %                     set(gca,'yticklabel','')
                %                 end
                %
            end
        end
    end
end

set(gcf,'position',[870 34 1656 783]);


%% plot conf as function of choice. 5/12/15 for VSS
cd('/Users/will/Google Drive/Will - Confidence/Presentations/cosyne')
load 9bin_hyperplots
%%
active_contrasts = [2 4 6];
contrast_colors = [10 94 0;...
    181 172 69;...
    189 189 189]...
    /256;
xl = [0 1; .5 1];
outputs = {'Chat', 'tf'}; %plotting these on the x axis
xlabels = {'prop. report "category 1"', '% correct'};

for m = 1:5
    figure(m)
    clf
    for o = 1:length(outputs)
        
        for t = 1:2
            task_id = tasks{t};
            
            tight_subplot(3,length(outputs)+1,t,o+1,[.042 .016]);
            loopvar = 0;
            for c = active_contrasts
                loopvar = loopvar+1;
                x = streal.(task_id).sumstats.all.mean.(outputs{o})(c,:);
                y = streal.(task_id).sumstats.all.mean.g(c,:);
                x_sem = streal.(task_id).sumstats.all.edgar_sem.(outputs{o})(c,:);
                y_sem = streal.(task_id).sumstats.all.edgar_sem.g(c,:);
                
                [x, sort_idx] = sort(x);
                y = y(sort_idx);
                x_sem = x_sem(sort_idx);
                y_sem = y_sem(sort_idx);
                
                x_fit = model(model_id).(tasks{t}).hyperplot.mean.(outputs{o})(c,:);
                y_fit = model(model_id).(tasks{t}).hyperplot.mean.g(c,:);
                x_fit_std = model(model_id).(tasks{t}).hyperplot.std.(outputs{o})(c,:);
                y_fit_std = model(model_id).(tasks{t}).hyperplot.std.g(c,:);
                x_fit = x_fit(sort_idx);
                y_fit = y_fit(sort_idx);
                x_fit_std = x_fit_std(sort_idx);
                y_fit_std = y_fit_std(sort_idx);
                
                
                hold on
                            plot(x, y, '-', 'color',contrast_colors(loopvar,:),'linewidth',1.5); % means
                   % 2 dimensional error bars:
%                 for pt = 1:length(x)
%                     x_pt1 = x(pt) - x_sem(pt);
%                     x_pt2 = x(pt) + x_sem(pt);
%                     y_pt1 = y(pt) - y_sem(pt);
%                     y_pt2 = y(pt) + y_sem(pt);
%                     
%                     plot([x_pt1 x_pt2], [y(pt) y(pt)], '-', 'color', contrast_colors(loopvar,:),'linewidth',1.5);
%                     plot([x(pt) x(pt)], [y_pt1 y_pt2], '-', 'color', contrast_colors(loopvar,:),'linewidth',1.5);
%                      
                    % ellipse fits:
% %                     e=plot_ellipse(x_fit_std(pt), y_fit_std(pt), 0, x_fit(pt), y_fit(pt), contrast_colors(loopvar,:));
%                 end
                
            end
            set(gca, 'box','off','tickdir','out','xlim',xl(o,:),'ylim',[1 4],'ytick',[1 2 3 4])
            if t == 1
                set(gca,'xticklabels','')
            elseif t == 2
                xlabel(xlabels{o}, 'fontsize', 20, 'fontname','Helvetica Neue')
            end
            
            if o == 1
                ylabel('mean confidence', 'fontsize', 20', 'fontname', 'Helvetica Neue')
            elseif o > 1
                set(gca,'yticklabels','')
            end
        end
    end
end

set(gcf, 'position', [40 1 913 804]); % 5/12

%% plot choice as function of conf (discrete version of the above). 5/12/15 for VSS.
streal.A = compile_data('datadir','/Users/will/Google Drive/Will - Confidence/Data/v3/taskA')
streal.B = compile_data('datadir','/Users/will/Google Drive/Will - Confidence/Data/v3/taskB')
tasks = {'A', 'B'};

figure(1)
clf

active_contrasts = [2 4 6];
contrast_colors = [10 94 0;...
    181 172 69;...
    189 189 189]...
    /256;


% compile all trials
for task = 1:2
    tight_subplot(3, 2, task, 2, [.042 .016]);
    all_g = [];
    all_Chat = [];
    all_tf = [];
    all_contrast_id = [];
    for subject = 1:length(streal.(tasks{task}).data)
        all_g           = [all_g     streal.(tasks{task}).data(subject).raw.g];
        all_Chat        = [all_Chat  streal.(tasks{task}).data(subject).raw.Chat];
        all_tf          = [all_tf  streal.(tasks{task}).data(subject).raw.tf];
        all_contrast_id = [all_contrast_id streal.(tasks{task}).data(subject).raw.contrast_id];
    end
    
    mean_Chat = zeros(length(active_contrasts), 4);
    sem_Chat  = zeros(length(active_contrasts), 4);
    
    mean_tf = zeros(length(active_contrasts), 4);
    sem_tf  = zeros(length(active_contrasts), 4);
    
    nCorrect = zeros(length(active_contrasts), 4);
    nIncorrect = zeros(length(active_contrasts), 4);
    beta_var = zeros(length(active_contrasts), 4);
    
    loopvar = 0;
    for c = active_contrasts
        loopvar = loopvar+1;
        for g = 1:4
            idx = (all_g == g) & (all_contrast_id == c);
            mean_Chat(loopvar, g) = mean(all_Chat(idx));
            std_Chat(loopvar, g)  = std (all_Chat(idx));
            sem_Chat(loopvar, g)  = std_Chat(g) / sqrt(sum(idx));
            
            mean_tf(loopvar, g) = mean(all_tf(idx));
            std_tf(loopvar, g)  = std (all_tf(idx));
            sem_tf(loopvar, g)  = std_tf(g) / sqrt(sum(idx));
            
            a = sum(all_tf(idx)==1); % number of correct trials
            b = sum(all_tf(idx)==0); % number of incorrect trials
            
            beta_var(loopvar, g) = a*b/(a+b)^2/(a+b+1); % can use sqrt(beta_var(loopvar,:) below instead of sem_tf?
        end
        errorbar(1:4, mean_tf(loopvar, :), sem_tf(loopvar,:), 'color', contrast_colors(loopvar,:),'linewidth', 2)
        hold on
    end
    set(gca,'tickdir','out','box','off','xtick',1:4,'ylim',[.5 .9],'ytick',.5:.2:.9,'xlim',[.5 4.5])
    if task == 1
        set(gca,'xticklabels','')
    elseif task == 2
        xlabel('confidence', 'fontname','Helvetica Neue','fontsize',20)
    end
    ylabel('prop. correct', 'fontname','Helvetica Neue','fontsize',20)
end


%% model comparison square grid. now replaced by compare_models.m

%close all
figure(1)
clf
% set(gcf,'position',[440 322 402 476])
set(0,'defaultaxesfontsize',12,'defaulttextinterpreter','none','defaultaxesfontname','Helvetica Neue')

nModels = length(model);
nDatasets = length(model(1).extracted);
dic = nan(nDatasets,nModels);
for m = 1:nModels
    for d = 1:nDatasets
        dic(d,m) = model(m).extracted(d).dic;
    end
end
models = 1:length(model);%[5 4 3 1 2];
model_names = {'lin A', 'fixed A', 'bayes A', 'lin B', 'fixed B', 'bayes B free', 'bayes B sym'};%{model.name};%{'Linear','Fixed','Bayes_{Free}','Bayes_{Sym}','Bayes_{Sym,Joint}'}; % these correspond with model(1:5)
[~,sort_idx] = sort(mean(dic,2)); % sort by best overall fit subject
dic = dic(sort_idx,models);

i=imagesc(dic);
for m = 1:nModels
    for d = 1:nDatasets
        t=text(m+.475,d+.45,num2str(dic(d,m),'%.0f'),'fontsize',13,'horizontalalignment','right');
        if d>=2, set(t,'color','white'), end
    end
end
pbaspect([nModels nDatasets 1])
set(gca,'box','on','xaxislocation','top','xtick',1:length(model),'xticklabel',{model_names{models}},'ticklength',[0 0],'linewidth',1,'ytick',1:nDatasets)
ylabel('Subject','interpreter','none')

c=colorbar;
colorsteps = 256;
colormap(flipud(gray(colorsteps)));
ticks=11000:2000:15000;
set(c,'ydir','reverse','box','on','ticklength',.035,'linewidth',1,'ticks',ticks,'ticklabels',{'11000','13000','15000'});


% mark best and worse with checks and crosses
rrr = [1 0 0];
ggg = [0 1 .1];
greenredmap = [linspace(ggg(1),rrr(1),colorsteps)' linspace(ggg(2),rrr(2),colorsteps)' linspace(ggg(3),rrr(3),colorsteps)'];
[~,best_idx]=min(dic');
[~,worst_idx]=max(dic');

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



%% model comparison bar plot

% replaced by mybar.m (needs better name)
cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations/v3')
load('v3_combined_and_cleaned_POSTER_DATA.mat')


dic_delta = dic_sorted-repmat(min(dic_sorted,[],2),1,5);
%%
% bar(-dic_delta','barwidth',.75)
nModels = 5;
nSubjects = 5;
inter_group_gutter=.2;
intra_group_gutter= 0.02;
barwidth = (1 - inter_group_gutter - (nDatasets-1)*intra_group_gutter)/nDatasets;
figure(2)
clf
for i = 1:20
    p=plot([.5 5.5],[-i*100 -i*100],'color',[.8 .8 .8],'linewidth',.8)
    uistack(p,'top')
    hold on
end

for m = 1:nModels
    for d = 1:nDatasets
        start = m-.5*(1-inter_group_gutter) + (barwidth+intra_group_gutter)*(d-1);
        f=fill([start start+barwidth start+barwidth start], [0 0 -dic_delta(d,m) -dic_delta(d,m)],'black','EdgeColor','none');
        hold on
    end
end

set(gca,'ticklength',[0 0],'box','off','xtick',1:nModels,'xticklabel',{'Bayes_{Sym, Joint}','Bayes_{Sym}','Bayes_{Free}','Linear','Fixed'},...
    'xaxislocation','top','fontweight','bold','fontname','Helvetica Neue','ytick',-2000:500:0,'ylim',[-2000 0])
set(gcf,'position',[1693 345 645 392])




%% find DIC contributions from the task A and B trials (for VSS 2015)
% cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations/v3')
% load('v3_combined_and_cleaned_POSTER_DATA.mat')

streal.A = compile_data('datadir','/Users/will/Google Drive/Will - Confidence/Data/v3/taskA')
streal.B = compile_data('datadir','/Users/will/Google Drive/Will - Confidence/Data/v3/taskB')

for m = 5%1:nModels
    sm = prepare_submodels(model(m));
    nSubjects = length(model(m).extracted);
    for subject = 1:nSubjects
        %         h = waitbar(0, sprintf('model %i, subject %i/%i', m, subject, nSubjects));
        
        all_p = vertcat(model(m).extracted(subject).p{:});
        p_A = all_p(:, sm.A_param_idx);
        p_B = all_p(:, sm.B_param_idx);
        
        nSamples = size(all_p, 1);
        LL_taskA = zeros(nSamples, 1);
        LL_taskB = zeros(nSamples, 1);
        
        progress_report_interval = 10;
        parfor sample = 1:nSamples
            if rand < 1/progress_report_interval
                fprintf('model %i, subject %i/%i, sample %i/%i\n', m, subject, nSubjects, sample, nSamples)
            end
            LL_taskA(sample) = -nloglik_fcn(p_A(sample, :), streal.A.data(subject).raw, sm.model_A, nDNoiseSets, category_params);
            LL_taskB(sample) = -nloglik_fcn(p_B(sample, :), streal.B.data(subject).raw, sm.model_B, nDNoiseSets, category_params);
        end
        %         close(h)
        
        model(m).extracted(subject).LL_taskA = LL_taskA;
        model(m).extracted(subject).LL_taskB = LL_taskB;
    end
end


%% big monster plot loop
trial_types = fieldnames(streal.data(1).stats);
trial_type_names = {'all trials','correct trials','incorrect trials','$\hat{C}=-1$ trials', '$\hat{C}=1$ trials', '$C=-1$ trials', '$C=1$ trials', 'trials after $\hat{C}=-1$', 'trials after $\hat{C}=1$'};
file_names = {'all','correct','incorrect','Chat_-1','Chat_1','C_-1','C_1','after_Chat_-1','after_Chat_1'};

switch gen_stat
    case 'resp'
        stats_over_s = {'Chat1_prop'; 'percent_correct'; 'resp_mean'};
        stats_over_c = {'Chat1_prop_over_c_and_g'; 'percent_correct_over_c_and_g'; 'resp_mean_over_c_and_Chat'};
        ylabels = {'Prop. $\hat{C}=-1$'; 'percent correct'; '$\langle\gamma\rangle$'};
        ylims = [0 1; 0 1; 1 8];
        
    case 'g'
        stats_over_s = {'Chat1_prop'; 'percent_correct'; 'g_mean'};
        stats_over_c = {'Chat1_prop_over_c_and_g'; 'percent_correct_over_c_and_g'; 'g_mean_over_c_and_Chat'};
        ylabels = {'Prop. $\hat{C}=-1$'; 'percent correct'; '$\langle\gamma\rangle$'};
        ylims = [0 1; 0 1; 1 4];
        
    case 'Chat'
        stats_over_s = {'Chat1_prop'; 'percent_correct'};
        stats_over_c = {'Chat1_prop_over_c'; 'percent_correct_over_c'};
        ylabels = {'Prop. $\hat{C}=-1$'; 'percent correct'};
        ylims = [0 1; 0.5 1];
end

nRows = 2*length(stats_over_s);
if show_fit
    nRows = nRows + 1;
end

% change default plotting colors.
% analytic expression over stimuli
% display more info over contrast
%make sure i can extract parameters. run optimize with lots of sets and
%optimizations. overnight
hhh = hot;
contrast_colors = hhh(round(linspace(1,40,6)),:); % black to orange indicate high to low contrast
sss = summer;
confidence_colors = [sss(round(linspace(55,15,3)),:); 0 .2 0]; % yellow to dark green indicate low to high confidence
Chat_colors = [0 .4 .8; .8 0 .8];

for model_id = plot_model
    %cd(paths{model_id})
    figure(model_id)
    for type = 1 %: length(trial_types)
        set(gcf,'position',[78 1 350*nDatasets 1705]) % [ULx ULy w h]
        
        for dataset = 1 : nDatasets
            for stat = 1 : length(stats_over_s)
                subplot(nRows,nDatasets,show_fit*nDatasets + (stat - 1) * nDatasets * 2 + dataset)
                
                set(0,'DefaultAxesColorOrder', contrast_colors)
                
                if show_fit
                    plot(1:n_bins_model, model(model_id).data(dataset).stats.(trial_types{type}).(stats_over_s{stat}))
                    hold on
                    plot(1:n_bins_real, streal.data(dataset).stats.(trial_types{type}).(stats_over_s{stat}), '.')
                else
                    plot(1:n_bins_real, streal.data(dataset).stats.(trial_types{type}).(stats_over_s{stat}), '-')
                    hold on
                end
                set(gca,'xtick', ori_label_bin_value,'xticklabel',ori_labels)
                
                if stat == 2 % plot chance line for % correct plots
                    plot(o_bound, [.5 .5], 'b--')
                end
                
                ylim(ylims(stat,:))
                xlim(o_bound)
                xlabel('stimulus (°)')
                if dataset == 1
                    %text(-.3, ylabel_y(stat), ylabels{stat},'units','normalized','rotation',90,'interpreter','latex','fontsize',17)
                    yl=ylabel(ylabels{stat});
                    pp=get(yl,'position');
                    set(yl,'interpreter','latex','position',[pp(1) ylims(stat,1)-.5 pp(3)])
                elseif dataset == nDatasets
                    for i = 1:length(contrasts)
                        legarray{i} = sprintf('c = %.1f%%', 100*contrasts(length(contrasts)+1-i));
                    end
                    lh=legend(legarray); % this works best if you plot model data first, so that it shows the lines rather than the dots in the legend. also, have to position the figure before you do this.
                    pp=get(lh,'position');
                    set(lh,'position',pp+[pp(3)+.01 0 0 0]); % move x position by width plus a little bit
                    
                end
                
                
                subplot(nRows, nDatasets, nDatasets * (show_fit + 1) + (stat - 1) * nDatasets * 2 + dataset);
                
                set(0,'DefaultAxesColorOrder', confidence_colors);
                if stat == 3
                    set(0, 'DefaultAxesColorOrder', Chat_colors);
                end
                
                if show_fit
                    semilogx(contrasts, model(model_id).data(dataset).stats.(trial_types{type}).(stats_over_c{stat}));
                    hold on
                    h=semilogx(contrasts, streal.data(dataset).stats.(trial_types{type}).(stats_over_c{stat}), '.');
                else
                    h=semilogx(contrasts, streal.data(dataset).stats.(trial_types{type}).(stats_over_c{stat}), '-');
                    hold on
                end
                
                if dataset == nDatasets
                    if stat == 3
                        lh = legend('$\hat{C}=-1$', '$\hat{C}=1$');
                        set(lh,'interpreter','latex');
                    else
                        lh = legend(h([4 3 2 1]),'\gamma = 4','\gamma = 3','\gamma = 2','\gamma = 1');
                    end
                    pp = get(lh,'position');
                    set(lh,'position',pp+[pp(3)+.01 0 0 0]);
                end
                
                if stat == 2 % plot chance line for % correct plots
                    curxlim = get(gca,'xlim');
                    plot(curxlim, [.5 .5], 'b--');
                end
                
                
                ylim(ylims(stat,:));
                xlim([min(contrasts)-.001 max(contrasts)+.03]); % THIS MIGHT BE A PROBLEM?
                set(gca, 'xtick', contrasts);
                set(gca, 'xticklabel', round(contrasts*1000)/10);
                xlabel('contrast (%)');
            end
            
        end
        
        %export_fig(sprintf('%s.pdf',file_names{type}),'-transparent')
        %close(gcf)
    end
    %cd(paths{model_id})
    % export_fig(sprintf('%s.pdf',opt_models{model_id}),'-transparent')
    
end
return

%% single subject/summary data
close all
figure
set(gcf,'position',[88 334 1218 472])

summary = 1;
plot_contrasts = [1 2 3 6];
stat_mean = {'Chat1_prop_mean', 'g_mean'};
stat_sem = {'Chat1_prop_sem','g_sem'};
alpha = .5;
dataset=1;

hhh=hot;
contrast_colors = hhh(round(linspace(1,40,6)),:); % black to orange indicate high to low contrast
set(0,'defaultaxesbox','off','defaultaxesticklength',[.02 .025],'DefaultAxesColorOrder', contrast_colors,'defaultlinemarkersize',12,'defaultlinelinewidth',2,'defaultaxeslinewidth',2,'defaulttextinterpreter','latex')

%p = panel();
stats_over_s = {'Chat1_prop','g_mean'};
ylims = [0 1;1 4];
yticks = {[0 .25 .5 .75 1],[1 2 3 4]};
xt = [-20 -10 0 10 20];
mnames = {'Optimal','Fixed','Lin.','Quad.'};
params = [15 14 21 21];
yl = {'prop. reports ``category 1"','mean confidence report'};
%p.pack(2,4);
for model_id = 1:length(model)
    model_id
    for stat = 1:2
        % p(stat,model_id).select();
        subplot(2,4,length(model)*(stat-1)+model_id)
        if summary
            for i = 1 : length(plot_contrasts)
                i
                contrast_id = plot_contrasts(i);
                eb=errorbar(real_axis, streal.sumstats.all.(stat_mean{stat})(contrast_id,:), streal.sumstats.all.(stat_sem{stat})(contrast_id,:), 'color', contrast_colors(contrast_id,:));
                hold on
                x = [model_axis fliplr(model_axis)];
                y = [model(model_id).sumstats.all.(stat_mean{stat})(contrast_id,:) + model(model_id).sumstats.all.(stat_sem{stat})(contrast_id,:) fliplr(model(model_id).sumstats.all.(stat_mean{stat})(contrast_id,:) - model(model_id).sumstats.all.(stat_sem{stat})(contrast_id,:))];
                h(i) = fill(x,y,contrast_colors(contrast_id,:));
                set(h(i),'edgecolor','none');
                set(h(i),'facealpha', alpha);
                hold on
                
                %uistack(eb,'top')
            end
            
        else
            plot(model_axis, model(model_id).data(dataset).stats.all.(stats_over_s{stat}))
            hold on
            plot(real_axis, streal.data(dataset).stats.all.(stats_over_s{stat}), '.')
        end
        set(gca,'ylim',ylims(stat,:),'ytick',yticks{stat},'xlim',[-20 20],'xtick',xt,'fontsize',16,'box','off')
        pbaspect([4 3 1])
        if stat == 1
            t = title({sprintf('\\makebox[4in][c]{\\fontsize{19}{0}\\selectfont{%s model}}',mnames{model_id}),sprintf('\\makebox[4in][c]{%i parameters}',params(model_id))},'interpreter','latex');
            %%t = title('\fontsize{20}{0}\selectfont{xxxx}' ,'interpreter','latex')
            set(gca,'xticklabel','')
        elseif stat==2
            xlabel('stimulus $s$ ($^\circ$)')
        end
        if model_id == 1
            yl{stat}
            ylabel(yl{stat})
        else
            set(gca,'yticklabel','')
        end
        
    end
end
export_fig('summary.pdf','-transparent')
%export_fig('singlesubject.pdf','-transparent')

%% RT. after running first cell.
hhh=hot;
contrast_colors = hhh(round(linspace(1,40,6)),:); % black to orange indicate high to low contrast
set(0,'defaultaxesbox','off','defaultaxesticklength',[.02 .025],'DefaultAxesColorOrder', contrast_colors,'defaultlinemarkersize',12,'defaultlinelinewidth',2,'defaultaxeslinewidth',2,'defaulttextinterpreter','latex')

% mean confidence as function of RT and contrast
plot(real_axis_rt,streal.sumstats.all.g_mean_rt)

plot(real_axis_rt,mean(streal.sumstats.all.g_mean_rt))
%%
clf
startup
gamp = zeros(2,4,11);
means = zeros(11,4);
means_r=zeros(11,8);
vars = zeros(11,4);
for subject = 1:11
    %subplot(3,4,subject)
    % g_mean_rt=streal.data(subject).stats.all.g_mean_rt;
    % g_mean_rt(find(isnan(g_mean_rt)))=0;
    % bin_counts_rt=streal.data(subject).stats.all.bin_counts_rt;
    % bin_counts_rt(find(isnan(bin_counts_rt)))=0;
    %
    % %weighted average over confidence
    % weighted_average=sum(g_mean_rt.*bin_counts_rt)./sum(bin_counts_rt);
    % plot(real_axis_rt,weighted_average,'-k')
    % hold on
    % scatter(real_axis_rt,weighted_average,sum(bin_counts_rt),'filled')
    %      set(gca,axes_defaults)
    % ylim([1 4])
    g=streal.data(subject).raw.g;
    rt=streal.data(subject).raw.rt;
    resp=streal.data(subject).raw.resp;
    for conf=1:4
        gamp(:,conf,subject) = gamfit(rt(g==conf));
        lnp(:,conf,subject) = lognfit(rt(g==conf));
        means(subject,conf) = mean(rt(g==conf));
        %vars(subject,conf) = var(rt(g==conf));
    end
    for r=1:8
        means_r(subject,r) = mean(rt(resp==r));
    end
    %     [x,idx]=sort(streal.data(subject).raw.g);
    %     y=streal.data(subject).raw.rt(idx);
    %     plot(x,y,'.','markersize',2)
    %     xlim([0 5])
    %     set(gca,'xtick',1:4,axes_defaults)
    %     [p,s,mu]=polyfit(x,y,1);
    %     yfit = polyval(p,x);
    % yresid = y - yfit;
    % SSresid = sum(yresid.^2);
    % SStotal = (length(y)-1) * var(y);
    % rsq = 1 - SSresid/SStotal
    %[p,table,stats]=anova1(y,x);
    %     hold on
    %     plot(x,yfit,'k-')
    %mdl = fitlm(g,rt);
    %plot(mdl)
    %     ylim([0 8]);
    %     xlabel('confidence rating')
    %     ylabel('rt (s)')
end

% gam_a = nanmean(gamp(1,:,:),3);
% gam_b = nanmean(gamp(2,:,:),3);
% lnp_mu= nanmean(lnp(1,:,:),3);
% lnp_sig=nanmean(lnp(2,:,:),3);

m = nanmean(means);
mr= nanmean(means_r);

sem = sqrt(nanvar(means))./sqrt(11);
semr= sqrt(nanvar(means_r))./sqrt(11);
% gam_mean=gam_a.*gam_b;
% lgn_mean=exp(lnp_mu+(lnp_sig.^2)./2);

%plot(1:4,[gam_mean;lgn_mean]','linewidth',3)
% hold on
%plot(1:4,[m])
%errorbar(1:4,m,sem,'linewidth',3)
errorbar(1:8,mr,semr,'k','linewidth',3)
hold on
for i = 1:8
    plot(i,mr(i),'.','markersize',60,'color',colors(i,:))
end
%legend('mean of fitted gamma','mean of fitted log normal')
xlabel('response')
ylabel('reaction time (s)')
%ylim([.65 1.35])
set(gca,'xtick',[],'fontsize',16,'ytick',.7:.2:1.3,'box','off','tickdir','out')
export_fig('rt_response.png','-m3')

%% confidence vs percent correct over subjects
clf
sss = summer;
confidence_colors = [sss(round(linspace(55,15,3)),:); 0 .2 0]; % yellow to dark green indicate low to high confidence
hhh=hot;
contrast_colors = flipud(hhh(round(linspace(1,40,6)),:)); % black to orange indicate high to low contrast

for subject = 1:11
    pcg(:,:,subject) = streal.data(subject).stats.all.percent_correct_over_c_and_g;
end

m = nanmean(pcg,3);
sem=sqrt(nanvar(pcg,0,3))./sqrt(11);
figure(2)
hold on

for i = 1:6
    l(i)=errorbar(1:4,m(i,:),sem(i,:),'linewidth',3,'color',contrast_colors(i,:));
end
xl=get(gca,'xlim')
plot(xl,[.5 .5],'--','color',[.2 .2 .2],'linewidth',2.5)
set(gca,'tickdir','out','fontsize',15,'ytick',.5:.25:1,'ylim',[.4 1],'xtick',1:4)
xlabel('confidence','interpreter','latex')
ylabel('proportion correct','interpreter','latex')

%legend(l([4 3 2 1]),'high conf.','somewhat high conf.','somewhat low conf.','low conf.','location','northwest')

%%
load('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations/v2_allsubjects_11bins_1e6samples_analyzed.mat')
%% 2x1 plot showing overall choice and confidence behavior
hhh=hot;
contrast_colors = hhh(round(linspace(1,40,6)),:); % black to orange indicate high to low contrast
set(0,'defaultaxesbox','off','defaultaxesticklength',[.02 .025],'DefaultAxesColorOrder', contrast_colors,'defaultlinemarkersize',12,'defaultlinelinewidth',2,'defaultaxeslinewidth',2,'defaulttextinterpreter','latex')

close all
figure
set(gcf,'position',[440 272 463 526])
for stat = 1:2
    subplot(2,1,stat)
    cla
    for i = 1 : 6
        %plot(real_axis, streal.sumstats.all.(stat_mean{stat})(i,:),'color',contrast_colors(i,:));
        plot(real_axis,streal.data(6).stats.all.(stats_over_s{stat}),'-')
        hold on
    end
    set(gca,'ylim',ylims(stat,:),'ytick',yticks{stat},'xlim',[-20 20],'xtick',xt,'fontsize',16,'box','off')
    yll=ylabel(yl{stat})
    pbaspect([4 3 1])
    if stat==1
        set(gca,'xticklabel','')
    elseif stat==2
        xlabel('stimulus $s$ ($^\circ$)')
        yypos = get(yll,'position')
        set(yll,'position',yypos-[3.7 0 0])
    end
end
break

%% scatter plot of all responses

clf
colors = [0 0 .4;
    .17 .17 .6;
    .33 .33 .8;
    .5 .5 1;
    1 .5 .5;
    .8 .33 .33;
    .6 .17 .17;
    .4 0 0];
figure(2)
hold on
set(gca,'ydir','reverse','yscale','log')
xlabel('s')
ylabel('contrast')
contrasts = exp(-4:.5:-1.5);
t = 0:pi/10:2*pi;
for c = 1:6
    for Chat = [-1 1]
        for g = 1:4
            raw=streal.data(3).sorted_raw.all{c};
            idx = find(raw.Chat == Chat & raw.g == g);
            plot(raw.s(idx), ones(1,length(idx))*contrasts(7-c)+contrasts(7-c)*.07*randn(1,length(idx)), '.', 'color',colors(Chat*g+5 - Chat/2 - 0.5,:),'markersize',6);
            %             for i = 1:length(idx)
            %                 patch((sin(t)+raw.s(idx(i))),(cos(t)+contrasts(7-c)),colors(Chat*g+5-Chat/2-0.5),'edgecolor','none')
            %             end
        end
    end
end

%% look at response hists
% columns are subjects
% rows are responses, following each possible response
%clear all
%close all
streal=compile_data('shuffle',true);
nDatasets = length(streal.data);
nTrials = length(streal.data(1).raw.C);
resp_levels = 8;


h1 = cell(nDatasets,1);
for dataset = 1:nDatasets
    %subplot(4,1,dataset)
    h1{dataset}=hist(streal.data(dataset).raw.resp,1:8);
    %ylim([0 1200])
end

h2=cell(nDatasets,1);
for dataset = 1 : nDatasets
    for r = 1 : resp_levels;
        idx = 1+find(streal.data(dataset).raw.resp==r);
        idx = idx(idx < nTrials);
        %subplot(nDatasets, resp_levels, resp_levels*(dataset-1)+r)
        h2{dataset}(r,:)=hist(streal.data(dataset).raw.resp(idx),1:8);
    end
end

%% wjm heat matp
X = NaN(8,8,4);

for i=1:4
    X(:,:,i)=bsxfun(@rdivide, h2{i}, sum(h2{i},2));
end
figure;
imagesc(mean(X,3));

%% histogram lines
figure
set(gcf,'position', [1024 1102 1537 404])
for dataset = 1 : nDatasets
    subplot(1,nDatasets,dataset)
    plot(1:resp_levels, h1{dataset}/nTrials,'k-o')
    xlim([1 8])
    hold on
    exact_resp = diag(h2{dataset});
    plot(1:resp_levels, exact_resp./sum(h2{dataset},2),'b-o')
    
    three_neighbors = (diag(h2{dataset})+[0;diag(h2{dataset},1)]+[diag(h2{dataset},-1);0]);
    d = conv(sum(h2{dataset},2), [1 1 1]);
    d = d(2:end-1);
    three_neighbors = three_neighbors./d;
    
    
    two_neighbors = [0;diag(h2{dataset},1)]+[diag(h2{dataset},-1);0]
    d = conv(sum(h2{dataset},2), [1 0 1]);
    d = d(2:end-1);
    two_neighbors = two_neighbors./d;
    
    [rsame, dsame, rdiff, ddiff] = deal(zeros(1,8));
    for r = 1:4
        rsame(r) = sum(h2{dataset}(setdiff(1:4,r),r));
        dsame(r) = sum(sum(h2{dataset}(setdiff(1:4,r),:)));
        rsame(r+4)=sum(h2{dataset}(setdiff(5:8,r+4),r+4));
        dsame(r+4)=sum(sum(h2{dataset}(setdiff(5:8,r+4),:)));
        
        rdiff(r) = sum(h2{dataset}(5:8,r));
        ddiff(r) = sum(sum(h2{dataset}(5:8,:)));
        rdiff(r+4)=sum(h2{dataset}(1:4,r+4));
        ddiff(r+4)=sum(sum(h2{dataset}(1:4,:)));
        
    end
    same_c_diff_g = rsame./dsame
    diff_c = rdiff./ddiff
    
    %tmp1 = h2{dataset}(1:4,5:8);
    %tmp2 = h2{dataset}(5:8,1:4);
    %diff_c = [sum(tmp1)/sum(sum(tmp1)) sum(tmp2)/sum(sum(tmp2))];
    
    
    plot(1:resp_levels, same_c_diff_g, 'r-o')
    plot(1:resp_levels, diff_c, 'g-o')
    plot(1:resp_levels, three_neighbors,'c-o')
    plot(1:resp_levels, two_neighbors,'m-o')
    
    
    if dataset == nDatasets
        lh=legend('$p(r_i=r)$','$p(r_i=r|r_{i-1}=r)$','$p(r_i=r|\hat{C}_{i-1}=\hat{C}_i,\gamma_{i-1}\neq\gamma_i)$','$p(r_i=r|\hat{C}_{i-1}\neq\hat{C}_i)$','$p(r_i=r|r_{i-1}\in\{r-1,r,r+1\}$','$p(r_i=r|r_{i-1}\in\{r-1,r+1\}$')
        p = get(lh,'position')
        set(lh,'interpreter','latex','location','south')%'position',[p(1)+p(3) p(2) p(3) p(4)])
        legend('boxoff')
    end
end

%% connection lines
%close all
figure
size = .05
set(gcf,'position',[51 528 1390 278])
for dataset = 1:nDatasets
    subplot(1,nDatasets,dataset)
    for i = 1:8
        for j = 1:8
            %plot([1 2],[i j])
            f=fill([1 2 2 1],[i-size j-size j+size i+size],'k');
            hold on;
            n=(h2{dataset}(i,j)/(sum(h2{dataset}(i,:))));
            %n=(h2{dataset}(i,j)/3240);
            set(f,'facealpha',n,'edgecolor','none');
            set(gca,'xticklabel','','xtick',[],'ylim',[.5 8.5],'ytick',[]);
            if dataset==1
                ylabel('r_i_-_1');
                set(gca,'ytick',1:8);
            elseif dataset==4
                l=ylabel('r_i');
                set(gca,'yaxislocation','right','ytick',1:8);
            end
            %             axis square
        end
    end
end
suplabel('dataset','x')
%% lots of intertrial histograms
figure
for dataset = 1:nDatasets
    subplot(4,1,dataset)
    streal.data(dataset).raw.Chat(streal.data(dataset).raw.Chat==1)=2;
    streal.data(dataset).raw.Chat(streal.data(dataset).raw.Chat==-1)=1;
    hist(streal.data(dataset).raw.Chat,1:2)
end

figure

for dataset = 1 : nDatasets
    for r = 1 : 2;
        idx = 1+find(streal.data(dataset).raw.Chat==r)
        idx = idx(idx < nTrials)
        subplot(nDatasets, 2, 2*(dataset-1)+r)
        hist(streal.data(dataset).raw.Chat(idx),1:2)
        %ylim([0 400])
    end
end

%% performance over sessions, blocks
streal = compile_data%('datadir','/Users/will/Ma lab/repos/qamar confidence/data/')

nTrialsPerBlock = 216;
nBlocksPerSession = 3;
nDatasets = length(streal.data);

for dataset = 1 : nDatasets
    nTrials = length(streal.data(dataset).raw.C);
    nSessions = nTrials / (nTrialsPerBlock * nBlocksPerSession);
    
    subplot(2,nDatasets, dataset);
    meanperf(dataset,:)=mean(reshape(streal.data(dataset).raw.tf,nTrialsPerBlock,nTrials/nTrialsPerBlock));
    plot(meanperf(dataset,:),'k-o');
    ylim([.5 .85]);
    xlim([0 16]);
    xl=get(gca,'xlim');
    yl=get(gca,'ylim');
    hold on
    for i = 1 : nSessions - 1
        plot([i*nBlocksPerSession+.5 i*nBlocksPerSession+.5], yl, 'b-')
    end
    xlabel('block');
    if dataset == 1
        ylabel('% correct');
    end
    
    subplot(2,nDatasets, dataset+nDatasets);
    meanconf(dataset,:)=mean(reshape(streal.data(dataset).raw.g,nTrialsPerBlock,nTrials/nTrialsPerBlock));
    plot(meanconf(dataset,:),'k-o');
    ylim([1 4]);
    xlim([0 16]);
    xl=get(gca,'xlim');
    yl=get(gca,'ylim');
    hold on
    for i = 1 : nSessions - 1
        plot([i*nBlocksPerSession+.5 i*nBlocksPerSession+.5], yl, 'b-');
    end
    xlabel('block')
    if dataset == 1
        ylabel('mean confidence');
    end
    
end
%% just look at % correct vs contrast, to see if the contrast/eccentricity levels are good. can run this cell alone
clear all
figure

st(1) = compile_data('datadir', '/Users/will/Ma lab/repos/qamar confidence/data/v1/');
training(1).tf = [.71 .75 .73 .69]; % calculated manually
st(2) = compile_data('datadir', '/Users/will/Ma lab/repos/qamar confidence/data/');
training(2).tf = [.75 .68 .77]; % calculated manually
nRows = length(st);

for row = 1 : nRows
    contrasts = fliplr(st(row).data(1).raw.contrast_values);
    nDatasets = length(st(row).data);
    real_bins = bin_generator(10);
    
    for dataset = 1 : nDatasets
        st(row).data(dataset).stats = indiv_analysis_fcn(st(row).data(dataset).raw, real_bins);
        
        subplot(nRows,nDatasets,nDatasets * (row - 1) + dataset)
        semilogx(contrasts, fliplr(st(row).data(dataset).stats.all.percent_correct_over_c), '.-');
        hold on
        plot(.45, training(row).tf(dataset), 'r.')
        curxlim = get(gca,'xlim');
        plot(curxlim, [.5 .5], 'b--');
        ylim([.4 1])
        xlim([.013 .6]);
        set(gca, 'xtick', contrasts);
        set(gca, 'xticklabel', round(contrasts*1000)/10);
        xlabel('contrast (%)');
        if dataset == 1
            ylabel('% correct')
        end
    end
end

%% choice plot (recreate qamar fig 2c)
clear all
close all

load('opt_g_same_bins_analyzed.mat')
%load('opt_Chat_same_bins_analyzed.mat')

figure
sig_levels = 6;
contrasts=exp(-4:.5:-1.5);
plot_contrasts = [1 2 3 6]
%default_colors=get(0,'DefaultAxesColorOrder'); % get default colors
qamar_colors = [0 0 .3; 1 0 0; 0 1 0; 0 0 1]; % this matches fig 2c in qamar.
hold on


for model_id = 1 : length(models);
    subplot(1,2,model_id)
    for i = 1:4;
        contrast_id = plot_contrasts(i);
        hold on
        errorbar(real_axis, streal.sumstats.all.Chat1_prop_mean(contrast_id,:), streal.sumstats.all.Chat1_prop_sem(contrast_id,:),'color', qamar_colors(i,:))
        
        x = [real_axis fliplr(real_axis)];
        y = [model(model_id).sumstats.all.Chat1_prop_mean(contrast_id,:) + model(model_id).sumstats.all.Chat1_prop_sem(contrast_id,:) fliplr(model(model_id).sumstats.all.Chat1_prop_mean(contrast_id,:)-model(model_id).sumstats.all.Chat1_prop_sem(contrast_id,:))];
        h(i) = fill(x,y,qamar_colors(i,:));
        set(h(i),'edgecolor','none');
        set(h(i),'facealpha', .3);
        legarray{i} = sprintf('$c=%.1f\\%%$', 100*contrasts(contrast_id));
    end
    ylim([0 1])
    ylabel('$p(\hat{C}=-1|s)$','interpreter','latex')
    xlabel('$s$', 'interpreter', 'latex')
    
    title(sprintf('model %s', models{model_id}))
end
lh=legend(h,legarray,'interpreter','latex');
lobjs = findobj(lh);
set(lobjs([2 4 6 8]), 'facealpha', .3);
set(gcf, 'position', [1160 1361 908 337]);

export_fig('fig2c_g.svg','-transparent','-m2')


%% confidence plot sumstats
%clear all
clear legarray h
close all
%load('opt_g_same_bins_analyzed.mat')
%load('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations/sumstats.mat')

figure
p = panel();
alpha = .5;
sig_levels = 6;
contrasts=exp(-4:.5:-1.5);
plot_contrasts = [1 3 4 6];

hhh = hot;
contrast_colors = hhh(round(linspace(1,40,length(plot_contrasts))),:); % black to orange indicate high to low contrast

nOptModels = length(opt_models);
%set(gcf,'position',[102 1007 500*nOptModels 500])
opt_models={'Optimal model','Fixed model','Quad model'};
parameters=[12 14 21];
stat_mean = {'Chat1_prop_mean', 'g_mean'};
stat_sem = {'Chat1_prop_sem','g_sem'};
yt = {[0 .5 1],1:4};

p.pack(length(stat_mean),length(opt_models));

for model_id = 1 : nOptModels;
    for stat = 1 : length(stat_mean);
        p(stat,model_id).select();
        pbaspect([4 3 1])
        set(gca,'visible','on','ytick',yt{stat})
        if stat==1
            set(gca,'xticklabel','')
            %t=title({opt_models{model_id}; sprintf('(%i params.)',parameters(model_id))})
            t = title({sprintf('\\makebox[4in][c]{%s}',opt_models{model_id}),sprintf('\\makebox[4in][c]{(%i params.)}',parameters(model_id))},'interpreter','latex');
        else
            xlabel('orientation $s$ ($^\circ$)','interpreter','latex')
        end
        
        if model_id~=1
            set(gca,'yticklabel','')
        end
        if model_id==1 && stat==1
            ylabel('prop. reports ``category 1"','interpreter','latex')
        elseif model_id==1 && stat==2
            ylabel('mean confidence','interpreter','latex')
        end
        for i = 1: length(plot_contrasts);
            contrast_id = plot_contrasts(i);
            hold on
            eb=errorbar(real_axis, streal.sumstats.all.(stat_mean{stat})(contrast_id,:), streal.sumstats.all.(stat_sem{stat})(contrast_id,:), 'color', contrast_colors(i,:),'linewidth',2);
            x = [model_axis fliplr(model_axis)];
            y = [model(model_id).sumstats.all.(stat_mean{stat})(contrast_id,:) + model(model_id).sumstats.all.(stat_sem{stat})(contrast_id,:) fliplr(model(model_id).sumstats.all.(stat_mean{stat})(contrast_id,:) - model(model_id).sumstats.all.(stat_sem{stat})(contrast_id,:))];
            h(i) = fill(x,y,contrast_colors(i,:));
            set(h(i),'edgecolor','none');
            set(h(i),'facealpha', alpha);
            legarray{i} = sprintf('$c=%.1f\\%%$', 100*contrasts(contrast_id));
            uistack(eb,'top')
            
        end
        axis([-20 20 min(yt{stat}) max(yt{stat})])
    end
end

set(gcf,'position',[95 921 727 461]);
%lh=legend(h,legarray,'interpreter','latex')
%lobjs = findobj(lh)
%set(lobjs([2:2:8]), 'facealpha', alpha)
%set(gcf, 'position', [1160 1361 908 337])
%export_fig('sumstats_g.svg','-transparent','-m2')


%% compare sigma-c fit for fake and extracted p
% f1=figure(1);
% set(f1,'position',[1441 1 1390 805]);
% for dataset = 1: nDatasets
% subplot(2,2,dataset);
% x = .015:.001:.226;
% contrasts = exp(-4:.5:-1.5);
% h1 = plot(x,sqrt(p(3,dataset)^2 + p(1,dataset) * x .^ -p(2,dataset)),'b-');
% hold on
% h2 = plot(contrasts, sqrt(p(3,dataset)^2 + p(1,dataset) * contrasts .^ -p(2,dataset)),'b.','markersize',20);
% h3 = plot(contrasts, sqrt(best_params(3,dataset)^2 + best_params(1,dataset) * contrasts .^ -best_params(2,dataset)),'b*','markersize',10);
% xlim([.015 .226])
% if dataset == nDatasets
%     suplabel('contrast','x')
%     suplabel('sigma','y')
%     lh=legend([h2 h3],{'true', 'extracted'});
%     set(lh,'position', [0.45, .955, 0.1, 0.05])
% end
% end



%% decision boundary plots

figure;
subplot(1,2,1)
errorbar(1:6,streal.sumstats.all.g_decb_mean, streal.sumstats.all.g_decb_sem)
xlabel('sigma')
ylabel('confidence')
xlim([.75 6.25])
ylim([1 4])
title('"Confidence reversal"')

subplot(1,2,2)
errorbar(1:6, streal.sumstats.all.Chat1_decb_prop, streal.sumstats.all.Chat1_decb_prop_sem)
xlabel('sigma')
ylabel('proportion of $\hat{C}=1$ reports','interpreter','latex')
xlim([.75 6.25])
ylim([0 1])
title('Shifting decision boundaries')

suptitle(sprintf('%g trials at ± %g ± %g°', sum(n), decb, window))

%% conf plot with error bars
figure
hold on

default_colors=get(0,'DefaultAxesColorOrder'); % get default colors

for i = 1:sig_levels;
    
    errorbar(real_axis, streal.sumstats.all.g_mean(i,:), streal.sumstats.all.g_sem(i,:),'color',default_colors(i,:))
end

xlim([-o_boundary o_boundary])
ylim([1 4])

%% conf plot with model fit!! %%%%%%%%%%
% model fits are to real stimuli presentation for each subject
figure
default_colors=get(0,'DefaultAxesColorOrder'); % get default colors

for i = 1 : 4
    subplot(2,2,i)
    plot(real_axis,streal.data(i).stats.all.g_mean,'o')
    ylim([1 4])
    hold on
    title(sprintf('subject %s',upper(streal.data(i).name)))
    
    plot(real_axis,model(model_id).data(i).stats.all.g_mean)
    
    ylabel('\gamma')
    xlabel('s')
    suptitle('dots=real data. lines=model data')
end

%% summary plots with model fit!!! %%%%%%%
% cluttered!
sig_levels = 6
figure
default_colors=get(0,'DefaultAxesColorOrder'); % get default colors
hold on
%plot(real_axis,streal.sumstats.all.g_mean)
for i = 1:sig_levels;
    %subplot(1,2,1)
    hold on
    errorbar(real_axis, streal.sumstats.all.g_mean(i,:), streal.sumstats.all.g_sem(i,:),'color',default_colors(i,:))
    
    %subplot(1,2,2)
    hold on
    x = [real_axis fliplr(real_axis)]
    y = [model(model_id).sumstats.all.g_mean(i,:)+model(model_id).sumstats.all.g_sem(i,:) fliplr(model(model_id).sumstats.all.g_mean(i,:)-model(model_id).sumstats.all.g_sem(i,:))];
    h = fill(x,y,default_colors(i,:));
    set(h,'edgecolor','none');
    set(h,'facealpha', .5);
end
ylim([1 4])





%% conf plot for each subject

contrasts = round(exp(-4.5:.5:-1.5)*1000)/10; % round and make percent
legendCell = eval(['{' sprintf('''c = %g%%'' ',contrasts) '}'])

for subject = 1 : length(st.data)
    figure
    plot(st.axis, st.data(subject).stats.all.g_mean)
    ylim([1 4])
    xlabel('s')
    ylabel('\gamma')
    title(['subject ' upper(st.data(subject).name)])
    legend(fliplr(legendCell))
    %export_and_reset(strcat('expectedval(confidence)_subject_',st.data(subject).name,'.pdf'))
end


%% hist of confidence reports per subject

for subject = 1 : length(st.data)
    subplot(length(st.data),2,subject*2-1)
    hist(st.data(subject).raw.g,4)
    ylim([0 1800])
    ylabel('Frequency')
    subplot(length(st.data),2,subject*2)
    hist(st.data(subject).raw.resp,8)
    ylim([0 1800])
    ylabel('Frequency')
end
subplot(length(st.data),2,2*length(st.data)-1)
xlabel('Confidence')
subplot(length(st.data),2,2*length(st.data))
xlabel('Confidence & Chat')
suptitle('Real responses')




%% shade behavioral data based on sem. this is in progress
%break
t=1;
figure
hold on

default_colors=get(0,'DefaultAxesColorOrder');

for i=1:length(sig);
    x=[data(t).o_axis(i,:) fliplr(data(t).o_axis(i,:))];
    y=[(data(t).g_mean(i,:)+data(t).g_sem(i,:)) fliplr(data(t).g_mean(i,:)-data(t).g_sem(i,:))];
    h=fill(x,y,default_colors(i,:));
    set(h,'edgecolor','none');
    set(h,'facealpha',.5);
end
xlim([-25 25])
ylim([1 4])

%% error bars instead of shading
%break
t=1;
figure
hold on

default_colors=get(0,'DefaultAxesColorOrder');

for i=1:length(sig);
    errorbar(data(t).o_axis(i,:), data(t).g_mean(i,:), data(t).g_sem(i,:),'color',default_colors(i,:),'linewidth',2)
end

xlim([-25 25])
ylim([1 4])
set(gca,'ytick',[1:.5:3])


%% histograms en route to roc



subplot(2,1,1)
hist([data(6).resp{1} data(6).resp{2} data(6).resp{3}],8)
n1=hist([data(6).resp{1} data(6).resp{2} data(6).resp{3}],8)
ylabel('Frequency of report')
set(gca,'xtick',[])
title('True class = 1')


subplot(2,1,2)
hist([data(7).resp{1} data(7).resp{2} data(7).resp{3}],8)
n2=hist([data(7).resp{1} data(7).resp{2} data(7).resp{3}],8)

ylabel('Frequency of report')
title('True class = 2')
set(gca,'xtick',[1.4:.88:8])
set(gca,'xticklabel',{'"VH 1"','"SH 1"','"SL 1"','"VL 1"','"VL 2"','"SL 2"','"SH 2"','"VH 2"'})
export_and_reset('response_histogram.pdf')


% not sure if this is right
x=[0 n2(1)/sum(n2) (n2(1)+n2(2))/sum(n2) (n2(1)+n2(2)+n2(3))/sum(n2) (n2(1)+n2(2)+n2(3)+n2(4))/sum(n2) (n2(1)+n2(2)+n2(3)+n2(4)+n2(5))/sum(n2) (n2(1)+n2(2)+n2(3)+n2(4)+n2(5)+n2(6))/sum(n2)  (n2(1)+n2(2)+n2(3)+n2(4)+n2(5)+n2(6)+n2(7))/sum(n2) 1];
y=[0 n1(1)/sum(n1) (n1(1)+n1(2))/sum(n1) (n1(1)+n1(2)+n1(3))/sum(n1) (n1(1)+n1(2)+n1(3)+n1(4))/sum(n1) (n1(1)+n1(2)+n1(3)+n1(4)+n1(5))/sum(n1) (n1(1)+n1(2)+n1(3)+n1(4)+n1(5)+n1(6))/sum(n1)  (n1(1)+n1(2)+n1(3)+n1(4)+n1(5)+n1(6)+n1(7))/sum(n2) 1];
plot(x,y,'linewidth',2)
xlabel('Miss rate')
ylabel('Hit rate')


%%
set(0,'defaultaxesfontsize',12,'defaulttextinterpreter','none','defaultaxesfontname','Helvetica Neue')

%mean of responses at contrast/orientation bins
% datadirs={'/Users/will/Google Drive/Will - Confidence/Data/v3/taskA/','/Users/will/Google Drive/Will - Confidence/Data/v3/taskB/'};
tasks = {'A','B'};
% bins = 13;
im = cell(2,1);
sort_idx = [4 1 2 5 3]; % sorted from best to worst fit.
for t = 1:2
    %     st = compile_data('datadir',datadirs{t});
    %     st.data = st.data(sort_idx);
    %     [real_bins, real_axis] = bin_generator(bins,'task',tasks{t});
    %     real_bins = [-Inf real_bins Inf];
    %             im{t} = zeros(6,bins,5);
    
    for d = 1:5
        
        %         raw = st.data(d).raw;
        %         for c = 1:6
        %             c_idx = raw.contrast_id==c;
        %             for b = 1:bins
        %                 idx = raw.s > real_bins(b) & raw.s < real_bins(b+1);
        %                 joint_idx = idx & c_idx;
        %                 n=hist(raw.resp(joint_idx),8);
        %                 im{t}(c,b,d)=mean(raw.resp(joint_idx));
        %             end
        %         end
        %         imagesc(flipud(im{t}(:,:,d)));
        tight_subplot(4,7,1+t,1+d,[.026 .0085])
        matrix = model(d).(tasks{t}).hyperplot.mean.resp;%(:,3:end-2);
        if t==1
            matrix(:,[1:2 end-1:end])=4.5;
        end
        imagesc(flipud(matrix)) % models instead of datasets
        
        load('MyColorMaps')
        colormap(confchoicemap)
        caxis([1 8 ])
        set(gca,'xticklabel',round(real_axis(1:2:end)),'xtick',1:2:bins,'ytick',[1 6],'yticklabel',{'low','high'},'box','off','tickdir','out','ticklength',[.02 .02])
        set(gca,'xtick',1:4:nBins)
        if t==1
            %             title(sprintf('subject %i',d))
            set(gca,'xticklabel','')
            if d==1
                %                 ylabel('contrast')
            else
                set(gca,'yticklabel','')
            end
        else
            set(gca,'xticklabel','')
            if d==1
                %                 ylabel('contrast')
            else
                set(gca,'yticklabel','')
            end
        end
        set(gca,'yticklabel','')
        set(gca,'linewidth',1)
        pause(.1)
    end
    %     tight_subplot(4,7,1+t,7,[.026 .0085]) % for average subject
    %     imagesc(flipud(mean(im{t},3))); % for average subject
    
    %     load('MyColorMaps')
    %     colormap(confchoicemap)
    %     caxis([1 8 ])
    %         set(gca,'xticklabel','','xtick',1:2:bins,'ytick',[1 6],'yticklabel','','box','off','tickdir','out','ticklength',[.02 .02])
    %         set(gcf,'position',[1329,288,1520,701])
    
end

%% mean parameter values
