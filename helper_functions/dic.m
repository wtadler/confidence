function score = dic(parameter_samples, loglik_samples, loglik_fcn)
% loglik_fcn needs to be an anonymous function with all the other parameters fixed.
% importantly, it has to have the right sign! (ie it should NOT be the negloglik function)

if iscell(parameter_samples)
    parameter_samples = vertcat(parameter_samples{:});
end

if iscell(loglik_samples)
    loglik_samples = vertcat(loglik_samples{:});
    if mean(loglik_samples) > 0 % if negative log likelihood is passed instead
        loglik_samples = -loglik_samples;
    end
end

if size(parameter_samples,1) ~= size(loglik_samples,1)
    error('problem')
end

mean_params = mean(parameter_samples)';

dbar  = -2 * mean(loglik_samples);
dtbar = -2 * loglik_fcn(mean_params);

score = 2 * dbar - dtbar; %DIC = 2(LL(theta_bar)-2LL_bar)

end