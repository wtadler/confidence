function submodel_struct = prepare_submodels(m)

submodel_struct.model_A = m;
submodel_struct.model_A.name = '';
submodel_struct.model_A.joint_task_fit = 0;
submodel_struct.model_A.joint_d = 0;
submodel_struct.model_A.diff_mean_same_std = 1;

submodel_struct.model_B = m;
submodel_struct.model_B.name = '';
submodel_struct.model_B.joint_task_fit = 0;
submodel_struct.model_B.joint_d = 0;
submodel_struct.model_B.diff_mean_same_std = 0;

% warning('saving pps')
% save pps

if ~m.joint_d
    submodel_struct.nonbound_param_idx = ~cellfun(@isempty, regexp(m.parameter_names,'^((?!(b_|m_)[0-9n]).)*$'));
    submodel_struct.A_bound_param_idx = ~cellfun(@isempty, regexp(m.parameter_names,'^(b_|m_).*TaskA'));
    submodel_struct.B_bound_param_idx = ~cellfun(@isempty, regexp(m.parameter_names,'^(b_|m_)(?!.*TaskA)'));
    
    submodel_struct.A_param_idx = logical(submodel_struct.nonbound_param_idx + submodel_struct.A_bound_param_idx);
    submodel_struct.B_param_idx = logical(submodel_struct.nonbound_param_idx + submodel_struct.B_bound_param_idx);
    
    submodel_struct.model_A = parameter_constraints(submodel_struct.model_A);
    submodel_struct.model_B = parameter_constraints(submodel_struct.model_B);
else
    [submodel_struct.nonbound_param_idx, submodel_struct.A_bound_param_idx, submodel_struct.B_bound_param_idx] = deal(nan); % not necessary for joint model
    
    if ~all(size(submodel_struct.model_A.parameter_names)==size(submodel_struct.model_B.parameter_names))
        error ('uh oh, we have a problem') % sanity check
    end
    
    [submodel_struct.A_param_idx, submodel_struct.B_param_idx] = deal(true(size(submodel_struct.model_A.parameter_names)));
    
    % hacks! normally taken care of by parameter_constraints
    submodel_struct.model_A.lapse_params = find(~cellfun(@isempty, strfind(submodel_struct.model_A.parameter_names,'lambda'))); % this line is normally in parameter_constraints
    submodel_struct.model_B.lapse_params = find(~cellfun(@isempty, strfind(submodel_struct.model_B.parameter_names,'lambda')));
    submodel_struct.model_A.term_params = find(~cellfun(@isempty, strfind(submodel_struct.model_A.parameter_names,'Term'))); % this line is normally in parameter_constraints
    submodel_struct.model_B.term_params = find(~cellfun(@isempty, strfind(submodel_struct.model_B.parameter_names,'Term')));
    submodel_struct.model_A.attention1 = 0;
    submodel_struct.model_B.attention1 = 0;
end

submodel_struct.joint_d = m.joint_d;