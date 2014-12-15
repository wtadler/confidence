function [score,scorereport] = calcscore(responses,denominator)


score = 100*sum(sum(responses.tf))/(denominator); % non-conf score

scorereport = [num2str(sum(sum(responses.tf))) '/' num2str(denominator) ' trials (' num2str(score,'%.1f') '%) correct.'];
