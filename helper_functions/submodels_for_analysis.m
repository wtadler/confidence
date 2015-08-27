function [tasks, models, param_indices] = submodels_for_analysis(model)
if model.joint_task_fit
    tasks = {'A', 'B'};
    sm = prepare_submodels(model);
    
    models = [sm.model_A, sm.model_B];
    param_indices = [sm.A_param_idx, sm.B_param_idx]';
else
    if model.diff_mean_same_std
        tasks = {'A'};
    else
        tasks = {'B'};
    end
    models = model;
    param_indices = true(size(model.lb))';
end