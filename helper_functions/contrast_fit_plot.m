function contrast_fit_plot(alpha, beta, sigma_0, varargin)
contrasts = exp(-4:.5:-1.5);
color = 'k';
yl = [0 16];
assignopts(who,varargin);

x   = 0:.001:max(contrasts)+.02;
semilogx(contrasts, sqrt(sigma_0^2 + alpha * contrasts .^ -beta),'o', 'color', color);
hold on
semilogx(x, sqrt(sigma_0^2 + alpha * x .^ -beta), 'color', color);
ylim(yl)
xlim([.013 .3])
set(gca, 'xtick', contrasts)
set(gca, 'xticklabel', round(contrasts*1000)/10)
xlabel('contrast (%)')
ylabel('\sigma')
end