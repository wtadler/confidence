function s = stimulus_orientations(category_params, category_type, category, M, N)

switch category_type
    case 'same_mean_diff_std'
        if category==1
            s = category_params.sigma_1 * randn(M,N);
        elseif category==2
            s = category_params.sigma_2 * randn(M,N);
        end
    case 'diff_mean_same_std'
        if category==1
            s = category_params.mu_1 + category_params.sigma_s * randn(M,N);
        elseif category==2
            s = category_params.mu_2 + category_params.sigma_s * randn(M,N);
        end
    case 'sym_uniform'
        if category==1
            s = -category_params.uniform_range * rand(M,N) + category_params.overlap;
        elseif category==2
            s = category_params.uniform_range * rand(M,N) - category_params.overlap;
        end
    case 'half_gaussian'
        if category==1
            s = -abs(category_params.sigma_s * randn(M,N));
        elseif category==2
            s = abs(category_params.sigma_s * randn(M,N));
        end
end