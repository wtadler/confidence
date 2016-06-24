function [aic, bic, aicc] = aicbic(logL, nParams, varargin)
% aic = AICBIC(logL, nParams)
% [aic, bic, aicc] = AICBIC(logL, nParams, nTrials)
% This is not the official aicbic function.

aic = 2 * nParams - 2 * logL;

if length(varargin) == 2
    % laplace???
end
if length(varargin) == 1
    nTrials = varargin{1};
    bic = -2 * logL + nParams * log(nTrials);
    aicc = aic + 2 * nParams * (nParams + 1) / (nTrials - nParams - 1);
end

