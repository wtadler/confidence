function p = parameter_variable_namer(p_in,model)

p.alpha   = p_in(1);
p.beta    = p_in(2);
p.sigma_0 = p_in(3);

if any(regexp(model, '^optP'))
    p.prior = p_in(4);
elseif any(regexp(model, '^opt'))
    p.prior = .5;
end

if any(regexp(model, '^lin')) || any(regexp(model, '^quad')) || any(regexp(model, '^fixed.*conf'))
    p.b_i     = [0 p_in(4) p_in(5) p_in(6) p_in(7) p_in(8) p_in(9) p_in(10) Inf];
elseif any(regexp(model, '^opt_asym'))
    p.b_i     = [-Inf p_in(4) p_in(5) p_in(6) p_in(7) p_in(8) p_in(9) p_in(10) Inf];
elseif any(regexp(model, 'opt.*conf')) % not asym bounds
    p.b_i     = [0 p_in(5) p_in(6) p_in(7) Inf];
elseif any(regexp(model, '^lin$')) || any(regexp(model, '^quad$')) || any(regexp(model, '^fixed$'))
    p.b_i     = [0 0 0 0 p_in(4) Inf Inf Inf Inf];
    p.lambda  = p_in(5);
end

if any(regexp(model, '^lin2')) || any(regexp(model, '^quad2'))
    p.m_i     = [-5 p_in(11) p_in(12) p_in(13) p_in(14) p_in(15) p_in(16) p_in(17) 30];
end


if any(regexp(model, '^optP$'))
    p.lambda  = p_in(5);
elseif any(regexp(model, '^opt'))
    p.lambda = p_in(4);
end


if any(regexp(model, '^lin2_multilapse')) || any(regexp(model, '^quad2_multilapse'))
    p.lambda_i= linspace(p_in(18), p_in(19), 4);
    if any(regexp(model, '(?<!no_)partial_lapse')) % partial lapse lin2 or quad2
        p.lambda_g= p_in(20);
    end
elseif any(regexp(model, '^lin2_partial')) || any(regexp(model, '^quad2_partial'))
    p.lambda  = p_in(18);
    p.lambda_g= p_in(19);
elseif any(regexp(model, '^opt_asym.*multilapse')) || any(regexp(model, '^fixed.*multilapse'))
    p.lambda_i= linspace(p_in(11),p_in(12),4);
    if any(regexp(model, '(?<!no_)partial_lapse'))
        p.lambda_g = p_in(13);
    end
elseif any(regexp(model, '^opt.*multilapse'))
    p.lambda_i= linspace(p_in(8),p_in(9),4);
    if any(regexp(model, '(?<!no_)partial lapse')) %partial lapse (ignore the 'no partial lapse' models)
        p.lambda_g= p_in(10);
    end
elseif any(regexp(model, '^lin')) || any(regexp(model, '^quad')) || any(regexp(model, '^opt_asym')) || any(regexp(model, '^fixed.*conf'))
    p.lambda  = p_in(11);
    if any(regexp(model, '(?<!no_)partial lapse')) %partial lapse (ignore the 'no partial lapse' models)
        p.lambda_g= p_in(12);
    end
elseif any(regexp(model, '^optP'))
    p.lambda  = p_in(8);
    if any(regexp(model, '(?<!no_)partial lapse'))
        p.lambda_g = p_in(9);
    end
elseif any(regexp(model, '^opt'))
    p.lambda  = p_in(7);
    if any(regexp(model, '(?<!no_)partial lapse'))
        p.lambda_g = p_in(8);
    end
end

if any(regexp(model, '^lin_partial')) || any(regexp(model, '^quad_partial'))
    p.sigma_p = p_in(13);
elseif any(regexp(model, '^lin_no_partial')) || any(regexp(model, '^quad_no_partial'))
    p.sigma_p = p_in(12);
elseif any(regexp(model, '^lin$')) || any(regexp(model, '^quad$'))
    p.sigma_p = p_in(6);
end



switch model
    case 'lin_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f\n\\sigma_d=%.2f', p_in);
    case 'lin_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f\n\\sigma_d=%.2f', p_in);
    case 'quad_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f\n\\sigma_d=%.2f', p_in);
    case 'quad_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f\n\\sigma_d=%.2f', p_in);
    case 'lin2_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\nm=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f', p_in);
    case 'lin2_multilapse_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\nm=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]\n\\lambda_\\gamma=%.2f', p_in);
    case 'lin2_multilapse_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\nm=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]', p_in);
    case 'quad2_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\nm=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f', p_in);
    case 'quad2_multilapse_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\nm=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]\n\\lambda_\\gamma=%.2f', p_in);
    case 'quad2_multilapse_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\nm=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]', p_in);
    case 'opt_asym_bounds_d_noise_multilapse_partial_lapse_conf'
        p.sigma_d = p_in(14);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]\n\\lambda_g=%.2f\n\\sigma_d=%.2f', p_in);
    case 'opt_asym_bounds_d_noise_multilapse_no_partial_lapse_conf'
        p.sigma_d = p_in(13);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]\n\\sigma_d=%.2f', p_in);
    case 'opt_asym_bounds_multilapse_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]\n\\lambda_g=%.2f', p_in);
    case 'opt_asym_bounds_multilapse_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]\n\\lambda_g=%.2f', p_in);
    case 'opt_asym_bounds_d_noise_partial_lapse_conf'
        p.sigma_d = p_in(13);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f\n\\sigma_d=%.2f', p_in);
    case 'opt_asym_bounds_d_noise_conf'
        p.sigma_d = p_in(12);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\sigma_d=%.2f', p_in);
    case 'opt_asym_bounds_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f', p_in);
    case 'opt_asym_bounds_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f', p_in);
    case 'optP_d_noise_conf'
        p.sigma_d = p_in(9);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\sigma_d=%.2f', p_in);
    case 'optP_d_noise_partial_lapse_conf'
        p.sigma_d = p_in(10);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_g=%.2f\n\\sigma_d=%.2f', p_in);
    case 'optP_d_noise_multilapse_partial_lapse_conf'
        p.sigma_d = p_in(11);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda_i=[%.1f; %.1f]\n\\lambda_g=%.2f\n\\sigma_d=%.2f', p_in);
    case 'optP_d_noise_multilapse_no_partial_lapse_conf'
        p.sigma_d = p_in(10);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda_i=[%.1f; %.1f]\n\\sigma_d=%.2f', p_in);
    case 'optP_multilapse_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda_i=[%.1f; %.1f]\n\\lambda_g=%.2f', p_in);
    case 'optP_multilapse_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]', p_in);
    case 'optP_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f', p_in);
    case 'optP_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda=%.2f', p_in);
    case 'opt_d_noise_conf'
        p.sigma_d = p_in(8);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\sigma_d=%.2f', p_in);
    case 'opt_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f', p_in);
    case 'opt_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb[%.1f; %.1f; %.1f]\n\\lambda=%.2f', p_in);
    case 'fixed_multilapse_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]\n\\lambda_\\gamma=%.2f', p_in);
    case 'fixed_multilapse_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda_i=[%.2f; %.2f]', p_in);
    case 'fixed_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f\n\\lambda_\\gamma=%.2f', p_in);
    case 'fixed_no_partial_lapse_conf'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb=[%.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f]\n\\lambda=%.2f', p_in);
    case 'optP_d_noise'
        p.sigma_d = p_in(5);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\n\\sigma_d=%.2f', p_in);
    case 'optP'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nprior=%.2f\n\\lambda=%.2f', p_in);
    case 'opt_d_noise'
        p.sigma_d = p_in(4);
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\n\\sigma_d=%.2f', p_in);
    case 'opt'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\n\\lambda=%.2f', p_in);
    case 'fixed'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb_0=%.2f\n\\lambda=%.2f', p_in);
    case 'quad'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb_0=%.2f\n\\lambda=%.2f\n\\sigma_p=%.2f', p_in);
    case 'lin'
        p.t_str = sprintf('\\alpha=%.1e\n\\beta=%.2f\n\\sigma_0=%.2f\nb_0=%.2f\n\\lambda=%.2f\n\\sigma_p=%.2f', p_in);
end