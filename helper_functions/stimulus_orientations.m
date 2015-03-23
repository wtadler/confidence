function s = stimulus_orientations(category_params, category, nStimuli)

switch category_params.category_type
    case 'same_mean_diff_std'
        if category==1
            s = category_params.sigma_1 * randn(1,nStimuli);
        elseif category==2
            s = category_params.sigma_2 * randn(1,nStimuli);
        end
    case 'diff_mean_same_std'
        if category==1
            s = category_params.mu_1 + category_params.sigma_s * randn(1,nStimuli);
        elseif category==2
            s = category_params.mu_2 + category_params.sigma_s * randn(1,nStimuli);
        end
    case 'sym_uniform'
        if category==1
            s = -category_params.uniform_range * rand(1,nStimuli) + category_params.overlap;
        elseif category==2
            s = category_params.uniform_range * rand(1,nStimuli) - category_params.overlap;
        end
    case 'half_gaussian'
        if category==1
            s = -abs(category_params.sigma_s * randn(1,nStimuli));
        elseif category==2
            s = abs(category_params.sigma_s * randn(1,nStimuli));
        end
end