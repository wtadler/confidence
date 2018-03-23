function plot_contrast_curves(model, varargin)

contrasts = exp(linspace(-5.5,-2,6)); % THIS IS HARD CODED
true_p_matrix = [];
assignopts(who, varargin);

nModels = length(model);

for m = 1:nModels
    figure
    nSubjects = length(model(1).extracted);
    
    for s = 1:nSubjects
        p = parameter_variable_namer(model.extracted(s).best_params, model.parameter_names, model, exp(linspace(-5.5,-2,6)));
        
        tight_subplot(1, nSubjects, 1, s);
        
        contrast_figure(p.sigma_c_low, p.sigma_c_hi, p.beta, contrasts, 'blue')

        if ~isempty(true_p_matrix)
            true_p = parameter_variable_namer(true_p_matrix(:,s), model.parameter_names, model, exp(linspace(-5.5, -2, 6)));
            
            contrast_figure(true_p.sigma_c_low, p.sigma_c_hi, p.beta, contrasts, 'red')
        end
        
        ylim([0 30])
    end
end
end

function contrast_figure(sigma_c_low, sigma_c_hi, beta, contrasts, color)
c_low = min(contrasts);
c_hi = max(contrasts);
contrasts_disc = contrasts;
contrasts_cont = linspace(c_low, c_hi, 100);
alpha = (sigma_c_low^2-sigma_c_hi^2)/(c_low^-beta - c_hi^-beta);
unique_sigs = @(contrasts) fliplr(sqrt(sigma_c_low^2 - alpha * c_low^-beta + alpha*contrasts.^-beta));
unique_sigs_disc = unique_sigs(contrasts_disc);
unique_sigs_cont = unique_sigs(contrasts_cont);

hold on
plot(fliplr(contrasts_disc), unique_sigs_disc, 'o', 'color', color)
hold on
plot(fliplr(contrasts_cont), unique_sigs_cont, '-', 'color', color)
end

% 
% contrasts = exp(linspace(-5.5,-2,6)); % THIS IS HARD CODED
% c_low = min(contrasts);
% c_hi = max(contrasts);
% for d = 1:5
%     subplot(1,5,d)
%     p.sigma_c_low = exp(noise_p(1,d));
%     p.sigma_c_hi = exp(noise_p(2,d));
%     p.beta = noise_p(3,d);
%     alpha = (p.sigma_c_low^2-p.sigma_c_hi^2)/(c_low^-p.beta - c_hi^-p.beta);
%     unique_sigs = fliplr(sqrt(p.sigma_c_low^2 - alpha * c_low^-p.beta + alpha*contrasts.^-p.beta)); % low to high sigma. should line up with contrast id
%     plot(fliplr(contrasts),unique_sigs,'o-')
%     ylim([0 60])
% end
% 
