% PAPER 1
AB_choice = load('model_fits/reliability_exp1_ABchoice.mat');
fisher = load('model_fits/reliability_exp1_fisher.mat');
freesigs = load('model_fits/reliability_exp1_free_cats.mat');
lesions = load('model_fits/reliability_exp1_lesions.mat');
% multiprior = load('v3_multiprior_MASTER.mat');
noise_fixed_to_A = load('model_fits/reliability_exp1_noise_tweaks.mat');
tweaks=load('model_fits/reliability_exp1_noise_tweaks.mat');
main = load('model_fits/reliability_exp1.mat');
tworesp = load('model_fits/reliability_exp2.mat');
v1v2 = load('model_fits/reliability_exp3.mat');


% PAPER 1
core = main.modelmaster(1:8);

core_A = main.modelmaster(16:21);
core_A_6freesigs = main.modelmaster(37:42); % this isn't on the map
core_A_choice = main.modelmaster(55:60); % this isn't on the map.

core_AB_choice = AB_choice.modelmaster(7:12);

core_B = main.modelmaster(9:15);
core_B_old = v1v2.modelmaster(1:7);
core_B_choice = main.modelmaster(49:54);
core_B_choice_old = v1v2.modelmaster(21:26);
core_B_choice_no_repeat_lapse = main.modelmaster(43:48); % this isn't on the map
core_B_6freesigs = main.modelmaster(30:36); % this isn't on the map

core_6freesigs = main.modelmaster(22:29);

core_measass_fixed = tweaks.modelmaster(1:8);
core_meas_fixed = tweaks.modelmaster(9:12);
core_measass_free = tweaks.modelmaster(13:16);
core_B_noise_measass_fixed = noise_fixed_to_A.modelmaster(1:7);
core_B_noise_measass_fixed_6freesigs = noise_fixed_to_A.modelmaster(8:14);

free_cat_sigs = freesigs.modelmaster(1:4);

tworesp = tworesp.modelmaster;

core_no_ODN = lesions.modelmaster(1:8);
bayes_no_ODN_no_d_noise = lesions.modelmaster(9:11);
bayes_no_d_noise = lesions.modelmaster(12:14);

fisher = fisher.modelmaster(1);


% PAPER 2
attention = load('model_fits/attention.mat');


%% make tables (latex)
% S1
tables{1} = my_struct_concat(core(8), bayes_no_d_noise, core([1:7]), core_6freesigs, free_cat_sigs, core_measass_free, fisher);
title{1} = 'AB_conf';

% S2
tables{2} = core_A([6 1:5]);
title{2} = 'A_conf';

% S3
tables{3} = core_B([7 1:6]);
title{3} = 'B_conf';

% S4 
tables{4} = core_AB_choice([6 1:5]);
title{4} = 'AB_choice';

% S5
tables{5} = core_B_noise_measass_fixed([7 1:6]);
title{5} = 'B_conf_fixed_to_A';

% S6
tables{6} = tworesp([8 1:7]);
title{6} = 'AB_2resp';

% S7
tables{7} = core_B_old([7 1:6]);
title{7} = 'B_conf_pilots';

% S8
tables{8} = core_B_choice_old([6 1:5]);
title{8} = 'B_choice_pilots';

% 
% 
% %%%% POWER LAW
% powerlaw = my_struct_concat(main.modelmaster(1:21), tweaks.modelmaster);
% 
% % MAIN 8 MODELS
% core = powerlaw([1:8]);
% 
% % TASK B
% B = powerlaw(9:15);
% all_B = my_struct_concat(B, freesigs(9:15));
% for m = 1:length(all_B)
%     all_B(m).extracted = my_struct_concat(all_B(m).extracted, v1v2.modelmaster(m).extracted);
% end
% 
% % TASK A
% A = powerlaw(16:21);
% all_A = my_struct_concat(A, freesigs(end-5:end));
% 
% 
% % MAIN 8 + NOISE TWEAKS
% joint_with_noise_tweaks = powerlaw([1:8 22:end]);
% joint_with_noise_tweaks = joint_with_noise_tweaks([23, 19, 3, 11, 22, 18, 2, 10, 21, 17, 1, 9, 24, 20, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16])
% 
% 
% %%%% 6 FREE SIGS
% freesigs = main.modelmaster(22:42);
% 
% % TASK B
% % B_freesigs = freesigs(9:15);
% 
% % TASK A
% % A_freesigs = freesigs(16:21);
% 
% % MAIN 8 MODELS
% joint_freesigs = freesigs(1:8);
% 
% all_joint = my_struct_concat(joint_with_noise_tweaks, joint_freesigs);
