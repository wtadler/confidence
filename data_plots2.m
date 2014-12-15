close all
clear all
tic

cdsandbox
%cd('~/Ma lab/Confidence theory/analysis plots/')

%set(0,'DefaultLineLineWidth',1)
set(0,'DefaultLineLineWidth','remove')
n_bins_model = 15;
n_bins_real = 15;
n_samples = 1e6;


% model fit
cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations')
%load('10_fake_model_x.mat')
%load('fmincon_fits_x_model.mat', 'extracted')
%load('fmincon_fits.mat', 'extracted') % d model
%load('parameter_recovery_x_Chat.mat') % 9 fake datasets

%load('4datasets.mat')
%load('opt_g.mat')
%load('opt_hpc_dgnoise_2000opts_1000sets.mat');
load('opt_hpc_d_x_g_partial_lapse_and_no'); % 4 confidence models

gen_stat = 'g';

nDatasets = length(model(1).extracted);


s = -20:.1:20;
sig1 = 3;
sig2 = 12;

streal = compile_data;
%streal = st; % use this if you've generated fake data in optimize

[model_bins, model_axis] = bin_generator(n_bins_model);
[real_bins, real_axis] = bin_generator(n_bins_real);

% generate fake trials based on extracted params, bin/analyze
for model_id = 1 : length(opt_models);
    
    for dataset = 1 : nDatasets;
        dataset
        tic
        model(model_id).data(dataset).raw = trial_generator(model(model_id).extracted(dataset).best_params,...
            'n_samples', n_samples, 'dist_type', 'qamar', 'contrasts', exp(-4:.5:-1.5),...
            'model', opt_models{model_id});%, 'model_fitting_data',streal.data(dataset).raw);
        [model(model_id).data(dataset).stats, model(model_id).data(dataset).sorted_raw] = indiv_analysis_fcn(model(model_id).data(dataset).raw, model_bins);
        
        [streal.data(dataset).stats, streal.data(dataset).sorted_raw]   = indiv_analysis_fcn(streal.data(dataset).raw, real_bins);
        toc
    end
    % SUMMARY STATS
    model(model_id).sumstats = sumstats_fcn(model(model_id).data); % big function that generates summary stats across subjects.
    streal.sumstats = sumstats_fcn(streal.data); % big function that generates summary stats across subjects.
    
end
% these files have already been analyzed as above, with both models, with 1e6 samples, 37
% model bins, 15 real bins. keep working on the analytical plots though


%%

%clear all
%close all
cd('/Users/will/Google Drive/Will - Confidence/Analysis')
%load('opt_g_analyzed.mat')
%load('opt_Chat_analyzed.mat')
%load('4datasets_75opt_dxmodels_dChatgen_analyzed.mat')
%load('parameter_recovery_x_Chat_analyzed')
%load('parameter_recovery_d_Chat_analyzed')


if ~exist('nDatasets')
    nDatasets = length(model(1).extracted)
end

% big plot loop! %%%%%%%%

trial_types = fieldnames(streal.data(1).stats);
trial_type_names = {'all trials','correct trials','incorrect trials','$\hat{C}=-1$ trials', '$\hat{C}=1$ trials', '$C=-1$ trials', '$C=1$ trials'};
file_names = {'all','correct','incorrect','Chat_-1','Chat_1','C_-1','C_1'};

switch gen_stat
    case 'g'
        stats_over_s = {'Chat1_prop'; 'percent_correct'; 'g_mean'};
        stats_over_c = {'Chat1_prop_over_c'; 'percent_correct_over_c_and_g'; 'g_mean_over_c'};
        ylabels = {'Prop. $\hat{C}=-1$'; 'percent correct'; '$\langle\gamma\rangle$'};
        ylabel_y = [-.6; -.7; -.3];
        ylims = [0 1; 0.5 1; 1 4];
        switch data_type
            case 'real'
                paths = {'/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model g fits real data D MODEL', '/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model g fits real data X MODEL'};
                nRows = 7;
            case 'fake'
                paths = {'/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model g fits fake data D MODEL', '/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model g fits fake data X MODEL'};
                nRows = 8;
        end
        
    case 'Chat'
        stats_over_s = {'Chat1_prop'; 'percent_correct'};
        stats_over_c = {'Chat1_prop_over_c'; 'percent_correct_over_c'};
        ylabels = {'Prop. $\hat{C}=-1$'; 'percent correct'};
        ylabel_y = [-1.0; -1.1];
        ylims = [0 1; 0.5 1];
        switch data_type
            case 'real'
                paths = {'/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model Chat fits real data D MODEL', '/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model Chat fits real data X MODEL'};
                nRows = 5;
            case 'fake'
                paths = {'/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model Chat fits fake data D MODEL', '/Users/will/Google Drive/Will - Confidence/Analysis/model fits/model Chat fits fake data X MODEL'};
                nRows = 6;
        end
end

contrasts = exp(-4:.5:-1.5);

% change default plotting colors.
% analytic expression over stimuli
% display more info over contrast
%make sure i can extract parameters. run optimize with lots of sets and
%optimizations. overnight
hhh = hot;
contrast_colors = hhh(round(linspace(1,40,6)),:); % black to orange indicate high to low contrast
sss = summer;
confidence_colors = [0 .2 0; sss(round(linspace(15,55,3)),:)]; % dark green to yellow indicate high to low confidence

%fpos = [91 -899 1395 1705; 1486 -899 1395 1705];


for model_id = 1 : length(opt_models);
    %cd(paths{model_id})
    figure
    
    for type = 1 : length(trial_types)
        for dataset = 1 : nDatasets
            switch data_type
                case 'real'
                    p_title = model(model_id).extracted(dataset).best_params;
                case 'fake'
                    p_title = p(:,dataset); % true p
            end
            
            [alpha, beta, sigma_0, prior, b_i, lambda, lambda_g, sigma_d, t_str] = parameter_variable_namer(p_title, opt_models{model_id});

            
            subplot(nRows,nDatasets,dataset)
            
            switch data_type
                case 'fake'
                    bar([p(:,dataset) model(model_id).extracted(dataset).best_params] ./ repmat(p(:,dataset),1,2), 'grouped')% normalize to true parameter
                    if dataset == 1
                        ylabel('norm. parameter extraction (red)')
                    end
                    %lhbar = legend('true (normalized)','extracted (normalized)')
                    title(sprintf('true params:\n%s', t_str));
                    subplot(nRows, nDatasets, nDatasets + dataset); % put contrast plot on next line
                    contrast_fit_plot(alpha, beta, sigma_0) % regular plot with true params
                    
                    alpha_extracted = model(model_id).extracted(dataset).best_params(1);
                    beta_extracted = model(model_id).extracted(dataset).best_params(2);
                    sigma_0_extracted = model(model_id).extracted(dataset).best_params(3);

                    contrast_fit_plot(alpha_extracted, beta_extracted, sigma_0_extracted,'color','r') % red stars to indicate extracted params
                    ylabel('contrast fitting (blue is true, red is extracted)');
                case 'real'
                    contrast_fit_plot(alpha, beta, sigma_0)
                    title(sprintf('%s\n%s', upper(streal.data(dataset).name), t_str));
            end

            
            sig = sqrt(sigma_0^2 + alpha * fliplr(contrasts) .^ -beta); % i've messed this up for fake data, I think
            
            % analytical stuff
            
            %f = @(y) (normcdf(y,s,sig(sigma)) - normcdf(-y,s,sig(sigma)));
            %analytical_model.g_mean = zeros(6,length(s));

            % analytical p(Chat = 1 | s)
%             switch models{model_id}
%                 case 'd'
%                     k1 = .5*log( (sig.^2 + sig2^2) ./ (sig.^2 + sig1^2)) + log(prior/(1-prior));
%                     k2 = (sig2^2 - sig1^2) ./ (2 .* (sig.^2 + sig1^2) .* (sig.^2 + sig2^2));
%                     k=sqrt(k1./k2);
%                     k2x2 = k2' * s.^2;
%                     d = repmat(k1',1,size(k2x2,2)) - k2x2;
%                     
%                     kminuss = repmat(k',1,size(s,2)) - repmat(s,length(k),1); % = k - s
%                     kpluss = repmat(k',1,size(s,2)) + repmat(s,length(k),1); % = k + s
%                     analytical_model.Chat1_prop =  lambda/2 + (1 - lambda) * .5 * (erf(kminuss./(sqrt(2)*repmat(sig',1,size(kminuss,2)))) + erf(kpluss./(sqrt(2)*repmat(sig',1,size(kpluss,2)))));
%                     
%                 case 'x'
%                     analytical_model.Chat1_prop = lambda/2 + (1 - lambda) * .5 * (erf((b_i(5) - repmat(s,length(sig),1)) ./ (sqrt(2)*repmat(sig',1,length(s)))) + erf((b_i(5) + repmat(s,length(sig),1)) ./ (sqrt(2)*repmat(sig',1,length(s)))));
%                     
%             end
            % analytical percent correct??
            % analytical <g> not working for d yet.
%             if strcmp(gen_stat,'g')
%                 switch models{model_id}
%                     case 'd'
%                         % this g_mean stuff isn't quite working yet.
%                         for sigma = 1:6
%                             for i = 1:4
%                                 a1 = (k1(sigma) - b_i(i+1)) / k2(sigma);
%                                 b1 = (k1(sigma) - b_i(i  )) / k2(sigma);
%                                 a2 = (k1(sigma) + b_i(i+1)) / k2(sigma);
%                                 b2 = (k1(sigma) + b_i(i  )) / k2(sigma);
%                                 if a1 > 0 && b1 > 0
%                                     '1'
%                                     sum1 = f(sqrt(b1)) + f(-sqrt(a1));
%                                 elseif a1 <= 0 && b1 > 0
%                                     '2'
%                                     sum1 = f(sqrt(b1));
%                                 elseif a1 < 0 && b1 < 0
%                                     '3'
%                                     sum1 = zeros(1,length(s));
%                                 end
%                                 
%                                 if a2 > 0 && b2 > 0
%                                     '1b'
%                                     sum2 = f(sqrt(a2)) + f(-sqrt(b2));
%                                 elseif b2 <= 0 && a2 > 0
%                                     '2b'
%                                     sum2 = f(sqrt(a2));
%                                 elseif b2 < 0 && a2 < 0
%                                     '3b'
%                                     sum2 = zeros(1,length(s));
%                                 end
%                                 p_nolapse = sum2+sum1;
%                                 analytical_model.g_mean(sigma,:) = analytical_model.g_mean(sigma,:) + i * (lambda_g/4 + (1 - lambda_g) * p_nolapse);
%                             end
%                         end
%                     case 'x'
%                         for sigma = 1:6
%                             for i = 1:4
%                                 p_nolapse = f(b_i(i+5)) + f(-b_i(-i+5)) + f(b_i(-i+6)) + f(-b_i(i+4));
%                                 analytical_model.g_mean(sigma,:) = analytical_model.g_mean(sigma,:) + i * (lambda_g/4 + (1 - lambda_g) * p_nolapse);
%                             end
%                         end
%                 end
%             end
            
            
            for stat = 1 : length(stats_over_s)
                switch data_type
                    case 'fake'
                        subplot(nRows,nDatasets, nDatasets * 2 + (stat-1) * nDatasets * 2 + dataset)
                    case 'real'
                        subplot(nRows,nDatasets, nDatasets + (stat-1) * nDatasets * 2 + dataset)
                end
                
                set(0,'DefaultAxesColorOrder', contrast_colors)

                plot(real_axis, streal.data(dataset).stats.(trial_types{type}).(stats_over_s{stat}), '.','markersize',9)
                hold on
                
                if stat == 1000 %|| stat == 3 % g_mean isn't ready for d yet.
                    plot(s, analytical_model.(stats_over_s{stat})) %???
                else
                    plot(model_axis, model(model_id).data(dataset).stats.(trial_types{type}).(stats_over_s{stat}))
                end
                
                ylim(ylims(stat,:))
                xlim([-20 20])
                xlabel('stimulus (°)')
                if dataset == 1
                    text(-.3, ylabel_y(stat), ylabels{stat},'units','normalized','rotation',90,'interpreter','latex','fontsize',17)
                end
                switch data_type
                    case 'fake'
                        subplot(nRows, nDatasets, nDatasets * 3 + (stat - 1) * nDatasets * 2 + dataset)

                    case 'real'
                        subplot(nRows, nDatasets, nDatasets * 2 + (stat - 1) * nDatasets * 2 + dataset)
                end
                
                set(0,'DefaultAxesColorOrder', confidence_colors)
                semilogx(contrasts, fliplr(streal.data(dataset).stats.(trial_types{type}).(stats_over_c{stat})), '.', 'markersize', 9);
                hold on
                semilogx(contrasts, fliplr(model(model_id).data(dataset).stats.(trial_types{type}).(stats_over_c{stat})));
                ylim(ylims(stat,:))
                xlim([.013 .3])
                set(gca, 'xtick', contrasts)
                set(gca, 'xticklabel', round(contrasts*1000)/10)
                xlabel('contrast (%)')
                
            end
            
        end
        [~,t]=suplabel(sprintf('%s\n%s', trial_type_names{type}, strrep(opt_models{model_id},'_','-')),'t');
        set(t, 'interpreter', 'latex', 'fontsize', 15);
        
        switch data_type
            case 'fake'
                set(gcf, 'position', [83 -899 1987 1705])
            case 'real'
                set(gcf,'position',[78 1 1401 1705])
        end
        %set(fh,'position', fpos(model_id,:))
        export_fig(sprintf('%s.pdf',file_names{type}),'-transparent')
        close(gcf)
    end
end



break


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
clear all
close all
load('opt_g_same_bins_analyzed.mat')

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
        errorbar(real_axis, streal.sumstats.all.g_mean(contrast_id,:), streal.sumstats.all.g_sem(contrast_id,:), 'color', qamar_colors(i,:))
        x = [real_axis fliplr(real_axis)];
        y = [model(model_id).sumstats.all.g_mean(contrast_id,:) + model(model_id).sumstats.all.g_sem(contrast_id,:) fliplr(model(model_id).sumstats.all.g_mean(contrast_id,:) - model(model_id).sumstats.all.g_sem(contrast_id,:))];
        h(i) = fill(x,y,qamar_colors(i,:));
        set(h(i),'edgecolor','none');
        set(h(i),'facealpha', .3);
        legarray{i} = sprintf('$c=%.1f\\%%$', 100*contrasts(contrast_id));
    end
    ylim([1 4])
    xlabel('$s$', 'interpreter', 'latex')
    ylabel('$\langle\gamma\rangle$', 'interpreter', 'latex');
    title(sprintf('model %s', models{model_id}))
end
lh=legend(h,legarray,'interpreter','latex')
lobjs = findobj(lh)
set(lobjs([2 4 6 8]), 'facealpha', .3)
set(gcf, 'position', [1160 1361 908 337])
export_fig('sumstats_g.svg','-transparent','-m2')
        
        
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







%% DEPRECATED PLOTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONFIDENCE
t=1;
plot(data(t).o_axis',data(t).g_mean') % Plot expected confidence
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
title('$\langle\gamma\rangle$','interpreter','latex')
legend_and_export('expectedval(confidence).pdf',sig,'percent',true)

t=2;
plot(data(t).o_axis',data(t).g_mean') % Plot <g>|Chat is correct
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat is correct.pdf',sig,'percent',true)

t=3;
plot(data(t).o_axis',data(t).g_mean') % Plot <g>|Chat is incorrect
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat is incorrect.pdf',sig,'percent',true)

t=4;
plot(data(t).o_axis',data(t).g_mean') % Plot <g>|Chat = 1
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}=1$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat=1.pdf',sig,'percent',true)

t=5;
plot(data(t).o_axis',data(t).g_mean') % Plot <g>|Chat = 2
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}=2$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat=2.pdf',sig,'percent',true)

t=6;
plot(data(t).o_axis',data(t).g_mean') % Plot <g>|C = 1
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
title('$\langle\gamma\rangle\mid$C=1','interpreter','latex')
legend_and_export('expectedval(confidence)|C=1.pdf',sig,'percent',true)

t=7;
plot(data(t).o_axis',data(t).g_mean') % Plot <g>|C = 2
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
title('$\langle\gamma\rangle\mid$C=2','interpreter','latex')
legend_and_export('expectedval(confidence)|C=2.pdf',sig,'percent',true)


% CHOICE

t=1;
plot(data(t).o_axis',data(t).Chat_mean') % Plot expected <Chat>
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle$','interpreter','latex')
legend_and_export('expectedval(Chat).pdf',sig,'percent',true)

t=2;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|Chat is correct
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('expectedval(Chat)|Chat is correct.pdf',sig,'percent',true)

t=3;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|Chat is incorrect
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('expectedval(Chat)|Chat is incorrect.pdf',sig,'percent',true)

t=6;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|C = 1
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid$C=1','interpreter','latex')
legend_and_export('expectedval(Chat)|C=1.pdf',sig,'percent',true)

t=7;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|C = 2
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid$C=2','interpreter','latex')
legend_and_export('expectedval(Chat)|C=2.pdf',sig,'percent',true)







%%

t=1;
plot(data(t).o_axis',data(t).g_std') % Plot std(confidence)
ylim([0 1])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)$','interpreter','latex')
legend_and_export('std(confidence).pdf',sig,'percent',true)

t=2;
plot(data(t).o_axis',data(t).g_std') % Plot std(g)|Chat is correct
ylim([0 1])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('std(confidence)|Chat is correct.pdf',sig,'percent',true)

t=3;
plot(data(t).o_axis',data(t).g_std') % Plot std(g)|Chat is incorrect
ylim([0 1])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('std(confidence)|Chat is incorrect.pdf',sig,'percent',true)

t=4;
plot(data(t).o_axis',data(t).g_std') % Plot std(g)|Chat = 1
ylim([0 1])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}=1$','interpreter','latex')
legend_and_export('std(confidence)|Chat=1.pdf',sig,'percent',true)

t=5;
plot(data(t).o_axis',data(t).g_std') % Plot std(g)|Chat = 2
ylim([0 1])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}=2$','interpreter','latex')
legend_and_export('std(confidence)|Chat=2.pdf',sig,'percent',true)

t=6;
plot(data(t).o_axis',data(t).g_std') % Plot std(g)|Chat = 1
ylim([0 1])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid$C=1','interpreter','latex')
legend_and_export('std(confidence)|C=1.pdf',sig,'percent',true)

t=7;
plot(data(t).o_axis',data(t).g_std') % Plot std(g)|Chat = 2
ylim([0 1])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid$C=2','interpreter','latex')
legend_and_export('std(confidence)|C=2.pdf',sig,'percent',true)

%% mean vs std

% you can do this one for qamar, but it's not useful
t=1;
plot(data(t).g_mean_sort',data(t).g_std_sort')
xlim([1 4])
ylim([0 1])
xlabel('$\langle\gamma\rangle$','interpreter','latex')
ylabel('$std(\gamma)$','interpreter','latex')
legend_and_export('mean_vs_std.pdf',sig,'percent',true)

% you can do this one for qamar, but it's not useful
t=2;
plot(data(t).g_mean_sort',data(t).g_std_sort')
xlim([1 4])
ylim([0 1])
xlabel('$\langle\gamma\rangle\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('mean_vs_std|correct.pdf',sig,'percent',true)

% you can do this one for qamar, but it's not useful
t=3;
plot(data(t).g_mean_sort',data(t).g_std_sort')
xlim([1 4])
ylim([0 1])
xlabel('$\langle\gamma\rangle\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('mean_vs_std|incorrect.pdf',sig,'percent',true)

t=4;
plot(data(t).g_mean_sort',data(t).g_std_sort')
xlim([1 4])
ylim([0 1])
xlabel('$\langle\gamma\rangle\mid\hat{C}=1$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}=1$','interpreter','latex')
legend_and_export('mean_vs_std|Chat=1.pdf',sig,'percent',true)

t=5;
plot(data(t).g_mean_sort',data(t).g_std_sort')
xlim([1 4])
ylim([0 1])
xlabel('$\langle\gamma\rangle\mid\hat{C}=2$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}=2$','interpreter','latex')
legend_and_export('mean_vs_std|Chat=2.pdf',sig,'percent',true)

t=6;
plot(data(t).g_mean_sort',data(t).g_std_sort')
xlim([1 4])
ylim([0 1])
xlabel('$\langle\gamma\rangle\mid$C=1','interpreter','latex')
ylabel('$std(\gamma)\mid$C=1','interpreter','latex')
legend_and_export('mean_vs_std|C=1.pdf',sig,'percent',true)

t=7;
plot(data(t).g_mean_sort',data(t).g_std_sort')
xlim([1 4])
ylim([0 1])
xlabel('$\langle\gamma\rangle\mid$C=2','interpreter','latex')
ylabel('$std(\gamma)\mid$C=2','interpreter','latex')
legend_and_export('mean_vs_std|C=2.pdf',sig,'percent',true)

%% Plot confidence Kepecs-style, one sigma value, correct and incorrect.
t=2;
plot(data(t).o_axis(3,:),data(t).g_mean(3,:),'color',[0 0.5 0]) % Plot <g>|Chat is correct
hold on
t=3;
plot(data(t).o_axis(3,:),data(t).g_mean(3,:),'color',[1 0 0]) % Plot <g>|Chat is correct
xlim([-o_boundary o_boundary])
ylim([1 4])
xlabel('s')
legend('Correct','Error')
title('$\langle\gamma\rangle\mid\sigma=0.4$','interpreter','latex')
export_and_reset('expectedval(confidence)|correct_and_error.pdf')

%% percent correct
% Sorts confidence and std as a function of percent correct. Same as above,
% except with percent correct as the sort index rather than confidence
for t=1:length(data);
    [data(t).percent_correct_sort, data(t).sort_index] = sort(data(t).percent_correct,2);
end

for i=1:length(sig);
    for t=1:length(data);
        data(t).g_std_sort(i,:) = data(t).g_std(i,data(t).sort_index(i,:));
        data(t).g_mean_sort(i,:) = data(t).g_mean(i,data(t).sort_index(i,:));
    end
end

t=1;
plot(data(t).o_axis',data(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('% correct')
legend_and_export('percentcorrect.pdf',sig,'percent',true)

t=4; % note that this converges to p(C=1|s)
plot(data(t).o_axis',data(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid\hat{C}=1$','interpreter','latex')
legend_and_export('percentcorrect|Chat=1.pdf',sig,'percent',true)



t=5; % note that this converges to p(C=2|s)
switch tasktype
    case 'qamar' % skip the center bin, where there are few Chat=2 trials.
        plot(data(t).o_axis(:,1:floor(bins/2))',data(t).percent_correct(:,1:floor(bins/2))')
        hold on
        plot(data(t).o_axis(:,1+ceil(bins/2):end)',data(t).percent_correct(:,1+ceil(bins/2):end)')
    case 'kepecs' % plot normally for kepecs
        plot(data(t).o_axis',data(t).percent_correct')
end

xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid\hat{C}=2$','interpreter','latex')
legend_and_export('percentcorrect|Chat=2.pdf',sig,'percent',true)

t=6;
plot(data(t).o_axis',data(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid$C=1','interpreter','latex')
legend_and_export('percentcorrect|C=1.pdf',sig,'percent',true)

t=7;
plot(data(t).o_axis',data(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid$C=2','interpreter','latex')
legend_and_export('percentcorrect|C=2.pdf',sig,'percent',true)

%% percent correct vs <g>
t=1; % note that this is not useful for qamar
plot(data(t).percent_correct_sort',data(t).g_mean_sort')
xlim([0 1])
ylim([1 4])
xlabel('% correct')
ylabel('$\langle\gamma\rangle$','interpreter','latex')
legend_and_export('percentcorrect_vs_meang.pdf',sig,'percent',true)

t=4; % note that this is not useful for kepecs)
plot(data(t).percent_correct_sort',data(t).g_mean_sort')
xlim([0 1])
ylim([1 4])
xlabel('$\% $ correct$\mid\hat{C}=1$','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid\hat{C}=1$','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|Chat=1.pdf',sig,'percent',true)

t=5; % note that this is not useful for kepecs)
plot(data(t).percent_correct_sort',data(t).g_mean_sort')
xlim([0 1])
ylim([1 4])
xlabel('$\% $ correct$\mid\hat{C}=2$','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid\hat{C}=2$','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|Chat=2.pdf',sig,'percent',true)

t=6; % note that this is not useful for kepecs)
plot(data(t).percent_correct_sort',data(t).g_mean_sort')
xlim([0 1])
ylim([1 4])
xlabel('$\% $ correct$\mid$C=1','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid$C=1','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|C=1.pdf',sig,'percent',true)

t=7; % note that this is not useful for kepecs)
plot(data(t).percent_correct_sort',data(t).g_mean_sort')
xlim([0 1])
ylim([1 4])
xlabel('$\% $ correct$\mid$C=2','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid$C=2','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|C=2.pdf',sig,'percent',true)

%% percent correct vs std(g)
t=1;
plot(data(t).percent_correct_sort',data(t).g_std_sort')
xlim([0 1])
ylim([0 1])
xlabel('% correct')
ylabel('std$(\gamma)$','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect.pdf',sig,'percent',true)

t=4;
plot(data(t).percent_correct_sort',data(t).g_std_sort')
xlim([0 1])
ylim([0 1])
xlabel('$\%$ correct$\mid\hat{C}=1$','interpreter','latex')
ylabel('std$(\gamma)\mid\hat{C}=1$','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|Chat=1.pdf',sig,'percent',true)

t=5;
plot(data(t).percent_correct_sort',data(t).g_std_sort')
xlim([0 1])
ylim([0 1])
xlabel('$\%$ correct$\mid\hat{C}=2$','interpreter','latex')
ylabel('std$(\gamma)\mid\hat{C}=2$','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|Chat=2.pdf',sig,'percent',true)

t=6;
plot(data(t).percent_correct_sort',data(t).g_std_sort')
xlim([0 1])
ylim([0 1])
xlabel('$\%$ correct$\mid$C=1','interpreter','latex')
ylabel('std$(\gamma)\mid$C=1','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|C=1.pdf',sig,'percent',true)

t=7;
plot(data(t).percent_correct_sort',data(t).g_std_sort')
xlim([0 1])
ylim([0 1])
xlabel('$\%$ correct$\mid$C=2','interpreter','latex')
ylabel('std$(\gamma)\mid$C=2','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|C=2.pdf',sig,'percent',true)


%% EXPECTED CHOICE

t=1;
plot(data(t).o_axis',data(t).Chat_mean') % Plot expected Chat
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle$','interpreter','latex')
legend_and_export('expectedval(Chat).pdf',sig,'percent',true)

t=2;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|Chat is correct
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('expectedval(Chat)|Chat is correct.pdf',sig,'percent',true)

t=3;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|Chat is incorrect
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('expectedval(Chat)|Chat is incorrect.pdf',sig,'percent',true)

t=6;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|C = 1
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid$C=1','interpreter','latex')
legend_and_export('expectedval(Chat)|C=1.pdf',sig,'percent',true)

t=7;
plot(data(t).o_axis',data(t).Chat_mean') % Plot <Chat>|C = 2
xlim([-o_boundary o_boundary])
ylim([1 2])
xlabel('s')
title('$\langle\hat{C}\rangle\mid$C=2','interpreter','latex')
legend_and_export('expectedval(Chat)|C=2.pdf',sig,'percent',true)


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

