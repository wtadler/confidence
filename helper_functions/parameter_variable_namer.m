function p = parameter_variable_namer(p_in, parameter_names, model)
% this gets called by nloglikfcn as well as trial_generator
% parameter_names is an input so that you don't have to run
% parameter_constraints

% name all the single variables
for i = 1 : length(parameter_names)
    p.(parameter_names{i}) = p_in(i);
end

% do special stuff for variables that go in vectors, such as boundaries...
if model.choice_only
    if strcmp(model.family, 'opt')
        p.b_i = [-Inf -Inf -Inf -Inf p.b_0_d Inf Inf Inf Inf];
    elseif strcmp(model.family, 'fixed')
        p.b_i = [0 0 0 0 p.b_0_x Inf Inf Inf Inf];
    elseif strcmp(model.family, 'lin') || strcmp(model.family, 'quad')
        p.b_i = [0 0 0 0 p.b_0_x Inf Inf Inf Inf];
        p.m_i = [0 0 0 0 p.m_0 Inf Inf Inf Inf];
    end
    
else
    if strcmp(model.family, 'opt')
        if model.symmetric
            tmp = [p.b_0_d p.b_1_d p.b_2_d p.b_3_d];
            p.b_i = [-Inf -fliplr(tmp(2:4)-tmp(1))+tmp(1) tmp Inf];
        elseif ~model.symmetric
            p.b_i = [-Inf p.b_n3_d p.b_n2_d p.b_n1_d p.b_0_d p.b_1_d p.b_2_d p.b_3_d Inf];
        end
    elseif strcmp(model.family, 'fixed')
        p.b_i = [0 p.b_n3_x p.b_n2_x p.b_n1_x p.b_0_x p.b_1_x p.b_2_x p.b_3_x Inf];
    elseif strcmp(model.family, 'lin') || strcmp(model.family, 'quad')
        p.b_i = [0 p.b_n3_x p.b_n2_x p.b_n1_x p.b_0_x p.b_1_x p.b_2_x p.b_3_x Inf];
        p.m_i = [0 p.m_n3 p.m_n2 p.m_n1 p.m_0 p.m_1 p.m_2 p.m_3 Inf];
    end
end

% ... and multi-lapse.
if model.multi_lapse
    p.lambda_i = linspace(p.lambda_1, p.lambda_4, 4);
end


% p.t_str???