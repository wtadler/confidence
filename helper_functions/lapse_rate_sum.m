function lapse_sum = lapse_rate_sum(p, model)

lapse_sum = sum(p(model.lapse_params));

if model.multi_lapse
    % add lambda_1 and lambda_4 again because of the interpolation
    lapse_sum = lapse_sum + p(strcmp(model.parameter_names, 'lambda_1')) + p(strcmp(model.parameter_names, 'lambda_4'));
end

end
