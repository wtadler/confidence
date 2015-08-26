function p = freesigs_to_powerlaw_params(p_in, contrasts)

p.sigma_c_low = p_in.sigma_c1;
p.sigma_c_hi = p_in.sigma_c6;

c_low = min(contrasts);
c_hi = max(contrasts);
alpha = @(beta) (p.sigma_c_low^2-p.sigma_c_hi^2)/(c_low^-beta - c_hi^-beta);
unique_sigs = @(beta) fliplr(sqrt(p.sigma_c_low^2 - alpha(beta) * c_low^-beta + alpha(beta)*contrasts.^-beta)); % low to high sigma. should line up with contrast id

cost = @(beta) sum((p_in.unique_sigs - unique_sigs(beta)).^2); % minimize sum of squared error

p.beta = fminsearch(cost, 1, optimset('display','iter'));