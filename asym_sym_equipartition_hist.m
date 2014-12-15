% load parameters
cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations')
%load('opt_hpc_dgnoise_2000opts_1000sets.mat')
%p = model.extracted(2).best_params;
%raw = trial_generator(p, 'model', 'optP_d_noise_conf', 'n_samples', 1e5, 'contrasts', exp(-4:.5:-1.5));
close all
clear all

%%
nSubjects = 4;
nBins = 100;
nSamples = 1e6;

conf_levels = 4;
bin_edges = linspace(1/conf_levels, 1 - 1/conf_levels, conf_levels - 1);

streal = compile_data;

for subject = 1:nSubjects
    subplot(5,nSubjects,subject);
    % real resp hist here.
    
    h = hist(streal.data(subject).raw.resp, 1:8);
    bar(5:8, fliplr(h(1:4)), 'b');
    hold on
    bar(1:4, fliplr(h(5:8)), 'r');
    ylabel('responses (real data)')
    set(gca,'xticklabel','', 'yticklabel', '');
    
    subplot(5, nSubjects, nSubjects * 1 + subject)
    
    load('200opts_hpc_real_opt_optconf_d_noise')
    %load('1000opts_hpc_real_all_except_d_noise')
    
    modelno = 1; % 4 is opt_no_partial_lapse_conf, 2 is optP_no_partial_lapse_conf
    p = model(modelno).extracted(subject).best_params;
    %p(8) = 0; % clear lapse
    raw = trial_generator(p, 'model', opt_models{modelno}, 'n_samples', nSamples, 'contrasts', exp(-4:.5:-1.5));
    
    h=hist(raw.resp, 1:8);
    bar(5:8, fliplr(h(1:4)), 'b');
    hold on
    bar(1:4, fliplr(h(5:8)), 'r');
    ylabel('responses (symmetric bounds)')
    set(gca,'xticklabel','', 'yticklabel', '')
    
    
    subplot(5, nSubjects, nSubjects * 2 + subject)
    equipartition_Chatn1 = quantile(raw.d(raw.Chat==-1), bin_edges);
    equipartition_Chat1 = quantile(raw.d(raw.Chat == 1), bin_edges);
    
    xlb = min([equipartition_Chat1 -p(7)]) - .3;
    xub = max([raw.d p(7) equipartition_Chatn1])+.3;
    range = linspace(xlb, xub, nBins);
    
    b=bar(range, histc(raw.d(raw.Chat==-1), range), 'facecolor', 'b', 'barwidth',1);
    ch = get(b,'child');
    set(ch,'facealpha',.5, 'edgealpha', 0)
    hold on
    
    b=bar(range, histc(raw.d(raw.Chat==1), range), 'facecolor', 'r', 'barwidth',1);
    ch = get(b,'child');
    set(ch,'facealpha',.5, 'edgealpha', 0)
    
    xlim([xlb xub])
    ylabel('frequency')
    xlabel('d')
    
    %
    y = get(gca, 'ylim');
    
    plot([0 0], y, 'k--')
    for i = 1 : length(bin_edges)
        plot([equipartition_Chatn1(i) equipartition_Chatn1(i)], y, 'b')
        plot([p(4+i) p(4+i)], y, 'b--')
        
        plot([equipartition_Chat1(i) equipartition_Chat1(i)], y, 'r')
        plot([-p(4+i) -p(4+i)], y, 'r--')
    end
    
    
    subplot(5, nSubjects, nSubjects * 3 + subject)
    
    load('1000opts_asym_partial_lapse_and_no_partial_lapse')
    
    modelno = 2; % 2 is opt_asym_bounds_no_partial_lapse_conf
    p = gen.opt(modelno).extracted(subject).best_params;
    raw = trial_generator(p, 'model', opt_models{modelno}, 'n_samples', nSamples, 'contrasts', exp(-4:.5:-1.5));
    
    h=hist(raw.resp, 1:8);
    bar(5:8, fliplr(h(1:4)), 'b');
    hold on
    bar(1:4, fliplr(h(5:8)), 'r');
    set(gca,'xticklabel','', 'yticklabel', '')
    ylabel('responses (asymmetric bounds)')

    
    subplot(5, nSubjects, nSubjects * 4 + subject)
    
    b=bar(range, histc(raw.d(raw.Chat==-1), range), 'facecolor', 'b', 'barwidth',1);
    ch = get(b,'child');
    set(ch,'facealpha',.5, 'edgealpha', 0)
    hold on
    
    b=bar(range, histc(raw.d(raw.Chat==1), range), 'facecolor', 'r', 'barwidth',1);
    ch = get(b,'child');
    set(ch,'facealpha',.5, 'edgealpha', 0)
    
    
    equipartition_Chatn1 = quantile(raw.d(raw.Chat==-1), bin_edges);
    equipartition_Chat1 = quantile(raw.d(raw.Chat == 1), bin_edges);
    
    xlb = min([equipartition_Chat1 p(4)]) - .3;
    xub = max([raw.d p(10) equipartition_Chatn1])+.3;
    range = linspace(xlb, xub, nBins);
    
    
    xlim([xlb xub])
    ylabel('frequency')
    xlabel('d')
    
    %
    conf_levels = 4;
    bin_edges = linspace(1/conf_levels, 1 - 1/conf_levels, conf_levels - 1);
    
    equipartition_Chatn1 = quantile(raw.d(raw.Chat==-1), bin_edges);
    equipartition_Chat1 = quantile(raw.d(raw.Chat == 1), bin_edges);
    y = get(gca, 'ylim');
    
    plot([p(7) p(7)], y, 'k--')
    for i = 1 : length(bin_edges)
        plot([equipartition_Chatn1(i) equipartition_Chatn1(i)], y, 'b')
        plot([p(3+i) p(3+i)], y, 'r--')
        
        plot([equipartition_Chat1(i) equipartition_Chat1(i)], y, 'r')
        plot([p(7+i) p(7+i)], y, 'b--')
    end
end
