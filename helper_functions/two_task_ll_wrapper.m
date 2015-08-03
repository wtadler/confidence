function ll=two_task_ll_wrapper(p_in, rawA, rawB, sm, nDNoiseSets, category_params, new)
if ~exist('new','var') % first run
    new=false;
else
    recompute = [1 1];
end
    
% sm is sub models
if ~sm.joint_d
    persistent old_p_in llA llB
    p_in = reshape(p_in,numel(p_in),1);
    if ~exist('old_p_in','var') || numel(old_p_in)~=numel(p_in) % the second thing happens when you call a new model. if you call a new model with the same number of params, they will all be different
        old_p_in = nan(size(p_in));
    end
    
    different_p = old_p_in~=p_in; % 1 indicates a changed param
    old_p_in = p_in;
    
    if new || any(different_p & sm.nonbound_param_idx) || any(different_p & sm.A_bound_param_idx) && any(different_p & sm.B_bound_param_idx)
        recompute = [1 1]; % recompute both
        p_A = p_in(sm.A_param_idx); % prepare param vectors for the sub models
        p_B = p_in(sm.B_param_idx);
        
    elseif any(different_p & sm.A_bound_param_idx)
        recompute = [1 0]; % recompute only A
        p_A = p_in(sm.A_param_idx);
    elseif any(different_p & sm.B_bound_param_idx)
        recompute = [0 1]; % recompute only B
        p_B = p_in(sm.B_param_idx);
    end
elseif sm.joint_d % always recompute both
    recompute = [1 1];
    [p_A, p_B] = deal(p_in);
end

if recompute(1)
    if ~isfield(sm.model_A, 'separate_measurement_and_inference_noise') % THESE 2 IF STATEMENTS ARE TEMPORARY
        sm.model_A.separate_measurement_and_inference_noise = 0;
    end
    llA = -nloglik_fcn(p_A, rawA, sm.model_A, nDNoiseSets, category_params);
end
if recompute(2)
    if ~isfield(sm.model_B, 'separate_measurement_and_inference_noise')
        sm.model_B.separate_measurement_and_inference_noise = 0;
    end
    llB = -nloglik_fcn(p_B, rawB, sm.model_B, nDNoiseSets, category_params);
end
ll = llA + llB;