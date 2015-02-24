function p = parameter_variable_namer(p_in, parameter_names, model)
% this gets called by nloglikfcn as well as trial_generator
% parameter_names is an input so that you don't have to run
% parameter_constraints
% exponentiate logs
logparams = strncmpi(parameter_names,'log',3);
p_in(logparams) = exp(p_in(logparams)); % exponentiate the log params
for l = find(logparams)'
    parameter_names{l} = parameter_names{l}(4:end);
end

% add terms
% sss='old';
% switch sss
%     case 'new'
        for t = model.termparams'
            p_in(t)=p_in(t-1)+p_in(t);
            parameter_names{t} = parameter_names{t}(1:end-4);
        end
%     case 'old'
%         termparams = ~cellfun(@isempty,strfind(parameter_names,'Term'));
%         for t=find(termparams)'
%             p_in(t)=p_in(t-1)+p_in(t);
%             parameter_names{t} = parameter_names{t}(1:end-4);
%         end
% end

% name all the single variables
for i = 1 : length(parameter_names)
    p.(parameter_names{i}) = p_in(i);
end

% do special stuff for variables that go in vectors, such as boundaries...
if model.choice_only
    if strcmp(model.family, 'opt')
        p.b_i = [-Inf -Inf -Inf -Inf p.b_0_d Inf Inf Inf Inf];
    elseif strcmp(model.family, 'fixed') || strcmp(model.family, 'MAP')
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
    end
end
% 
% if model.diff_mean_same_std % why did i put this in here??
%     p.b_i(p.b_i==0) = -Inf;
% end

% ... and multi-lapse.
if model.multi_lapse
    p.lambda_i = linspace(p.lambda_1, p.lambda_4, 4);
end


