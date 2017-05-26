function p = parameter_variable_namer(p_in, parameter_names, model, contrasts)
% this gets called by nloglikfcn as well as trial_generator
% parameter_names is an input so that you don't have to run
% parameter_constraints
% exponentiate logs
logparams = strncmpi(parameter_names,'log',3);
p_in(logparams) = exp(p_in(logparams)); % exponentiate the log params
for l = find(logparams)'
    parameter_names{l} = parameter_names{l}(4:end); % chop off the 'log' in the name
end

if ~isfield(model, 'term_params')
    model.term_params = model.termparams; % I changed the name here and it's been messing things up for old data.
end

for t = model.term_params'
    p_in(t)=p_in(t-1)+p_in(t);
    parameter_names{t} = strrep(parameter_names{t}, 'Term', '');
%     parameter_names{t} = parameter_names{t}(1:end-4); % chop off the 'term' in the name
end

% name all the single variables
for i = 1 : length(parameter_names)
    p.(parameter_names{i}) = p_in(i);
end

% do special stuff for variables that go in vectors, such as sigma...
if ~isfield(model, 'nFreesigs') || model.nFreesigs == 0
    if ~exist('contrasts', 'var')
        contrasts = exp(linspace(-5.5,-2,6));
    else
        contrasts = sort(contrasts); % to ensure that it is low to high contrast.
    end
    c_low = min(contrasts);
    c_hi = max(contrasts);
    alpha = (p.sigma_c_low^2-p.sigma_c_hi^2)/(c_low^-p.beta - c_hi^-p.beta);
    p.unique_sigs = fliplr(sqrt(p.sigma_c_low^2 - alpha * c_low^-p.beta + alpha*contrasts.^-p.beta)); % low to high sigma. should line up with contrast id
    p.unique_sigs = max(real(p.unique_sigs), exp(-4)); % prevents problems in nloglik_fcn. this is temporary. won't need this after changes in parameter_constraints.m trickle down.
    
    if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
        alpha_inference = (p.sigma_c_low_inference^2-p.sigma_c_hi_inference^2)/(c_low^-p.beta_inference - c_hi^-p.beta_inference);
        p.unique_sigs_inference = fliplr(sqrt(p.sigma_c_low_inference^2 - alpha_inference * c_low^-p.beta_inference + alpha_inference*contrasts.^-p.beta_inference)); % low to high sigma. should line up with contrast id
%         p.unique_sigs_inference = max(p.unique_sigs_inference, exp(-4)); % prevents problems in nloglik_fcn. this is temporary. won't need this after changes in parameter_constraints.m trickle down. 
    end
else
    p.unique_sigs = [];
    if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
        p.unique_sigs_inference = [];
    end
    for sig_id = 1:model.nFreesigs
        p.unique_sigs = [p.(['sigma_c' num2str(sig_id)]) p.unique_sigs];
        if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
            p.unique_sigs_inference = [p.(['sigma_c' num2str(sig_id) '_inference']) p.unique_sigs_inference];
        end
    end
end

% and boundaries...
if model.choice_only
    if strcmp(model.family, 'opt')
        p.b_i = [-Inf -Inf -Inf -Inf p.b_0_d Inf Inf Inf Inf];
    elseif strcmp(model.family, 'fixed') || strcmp(model.family, 'MAP')
        if ~isfield(model, 'nFreebounds') || model.nFreebounds == 0
            p.b_i = [0 0 0 0 p.b_0_x Inf Inf Inf Inf];
        else
            for b = 1:model.nFreebounds
                p.b_i(b,:) = [0 0 0 0 p.(sprintf('b_0_x_c%i', b)) Inf Inf Inf Inf];
            end
        end 
    elseif strcmp(model.family, 'lin') || strcmp(model.family, 'quad')
        p.b_i = [0 0 0 0 p.b_0_x Inf Inf Inf Inf];
        p.m_i = [0 0 0 0 p.m_0 Inf Inf Inf Inf];
    elseif strcmp(model.family, 'neural1')
        p.b_i = [0 0 0 0 p.b_0_neural1 Inf Inf Inf Inf];
    end
    
else
    if strcmp(model.family, 'opt')
        if ~isfield(model, 'fisher_info') || isempty(model.fisher_info) || ~model.fisher_info
            if model.symmetric
                tmp = [p.b_0_d p.b_1_d p.b_2_d p.b_3_d];
                p.b_i = [-Inf -fliplr(tmp(2:4)-tmp(1))+tmp(1) tmp Inf];
            elseif ~model.symmetric
                p.b_i = [-Inf p.b_n3_d p.b_n2_d p.b_n1_d p.b_0_d p.b_1_d p.b_2_d p.b_3_d Inf];
            end
        else
            p.b_i = [0 p.b_0_d p.b_1_d p.b_2_d Inf];
        end
    elseif strcmp(model.family, 'fixed') || strcmp(model.family, 'MAP')
        if model.symmetric % only applies for task A. no symmetry in these models for task B.
            tmp = [p.b_0_x p.b_1_x p.b_2_x p.b_3_x];
            p.b_i = [-Inf -fliplr(tmp(2:4)-tmp(1))+tmp(1) tmp Inf];
        else
            p.b_i = [0 p.b_n3_x p.b_n2_x p.b_n1_x p.b_0_x p.b_1_x p.b_2_x p.b_3_x Inf];
        end
    elseif strcmp(model.family, 'lin') || strcmp(model.family, 'quad')
        if model.symmetric % only applies for task A. no symmetry in these models for task B.
            tmp = [p.b_0_x p.b_1_x p.b_2_x p.b_3_x];
            p.b_i = [-Inf -fliplr(tmp(2:4)-tmp(1))+tmp(1) tmp Inf];
            tmp = [p.m_0 p.m_1 p.m_2 p.m_3];
            p.m_i = [-Inf -fliplr(tmp(2:4)-tmp(1))+tmp(1) tmp Inf];
        else
            p.b_i = [0 p.b_n3_x p.b_n2_x p.b_n1_x p.b_0_x p.b_1_x p.b_2_x p.b_3_x Inf];
            p.m_i = [0 p.m_n3 p.m_n2 p.m_n1 p.m_0 p.m_1 p.m_2 p.m_3 Inf];
        end
    elseif strcmp(model.family, 'neural1')
        if model.symmetric
            tmp = [p.b_0_neural1 p.b_1_neural1 p.b_2_neural1 p.b_3_neural1];
            p.b_i = [-Inf -fliplr(tmp(2:4)-tmp(1))+tmp(1) tmp Inf];
        else
            p.b_i = [0 p.b_n3_neural1 p.b_n2_neural1 p.b_n1_neural1 p.b_0_neural1 p.b_1_neural1 p.b_2_neural1 p.b_3_neural1 Inf];
        end
    end
end
% 
% if model.diff_mean_same_std % why did i put this in here??
%     p.b_i(p.b_i==0) = -Inf;
% end

% ... and multi-lapse.
if model.multi_lapse
    p.lambda_i = [p.lambda_1 p.lambda_1+(p.lambda_4-p.lambda_1)/3 p.lambda_1+2*(p.lambda_4-p.lambda_1)/3 p.lambda_4];% 8 times faster than: linspace(p.lambda_1, p.lambda_4, 4);
end


