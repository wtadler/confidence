function contrast_fit_plot(sigma_c_low, sigma_c_hi, beta, varargin)%(alpha, beta, sigma_0, varargin)
% contrasts = exp(-4:.5:-1.5);
contrasts = exp(linspace(-5.5,-2,6));

color = 'k';
yl = [0 40];
assignopts(who,varargin);

contrasts_continuous   = 0:.001:max(contrasts)+.02;

% old sig
% sig = sqrt(sigma_0^2 + alpha * contrasts .^ -beta);

% new sig
c_low = min(contrasts);
c_hi = max(contrasts);
alpha = (sigma_c_low^2-sigma_c_hi^2)/(c_low^-beta - c_hi^-beta);
sig = sqrt(sigma_c_low^2 - alpha * c_low^-beta + alpha*contrasts.^-beta); % low to high sigma. should line up with contrast id
sig_continuous = sqrt(sigma_c_low^2 - alpha * c_low^-beta + alpha*contrasts_continuous.^-beta);

semilogx(contrasts, sig,'o', 'color', color);
hold on
semilogx(contrasts_continuous, sig_continuous, 'color', color);
ylim(yl)
xlim([.0035 .18])
set(gca, 'xtick', contrasts)
set(gca, 'xticklabel', round(contrasts*1000)/10)
xlabel('contrast (%)')
ylabel('\sigma')
end