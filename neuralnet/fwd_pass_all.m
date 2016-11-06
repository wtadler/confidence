function [output_p, RMSE, info_loss, perf] = fwd_pass_all(R, W, b, nLayers, hidden_unit_type, optimal_p, C)

nTrials = size(R, 1);
output_p = zeros(nTrials, 1);
for trial = 1:nTrials
    a = fwd_pass(R(trial,:)', W, b, nLayers, hidden_unit_type);
    output_p(trial) = a{end};
end

RMSE = sqrt(mean((optimal_p-output_p).^2));
info_loss = nanmean(optimal_p.*log(optimal_p./output_p) + (1-optimal_p).*log((1-optimal_p)./(1-output_p))) ...
            / nanmean(optimal_p.*log(2*optimal_p) + (1-optimal_p).*log(2*(1-optimal_p)));
perf = mean((output_p > .5) == C);