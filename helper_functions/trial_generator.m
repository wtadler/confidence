function raw = trial_generator(p_in, model, varargin)

% define defaults
n_samples = 3240;
dist_type = 'same_mean_diff_std'; % 'same_mean_diff_std' (Qamar) or 'diff_mean_same_std' or 'sym_uniform' or 'half_gaussian' (Kepecs)
contrasts  = exp(-4:.5:-1.5);%[.125 .25 .5 1 2 4];
model_fitting_data = [];
conf_levels = 4;

category_params.sigma_1 = 3;
category_params.sigma_2 = 12;
category_params.sigma_s = 5; % for 'diff_mean_same_std' and 'half_gaussian'
category_params.a = 0; % overlap for sym_uniform
category_params.mu_1 = -4; % mean for 'diff_mean_same_std'
category_params.mu_2 = 4;
category_params.uniform_range = 1;

assignopts(who,varargin);

nContrasts = length(contrasts);

p = parameter_variable_namer(p_in, model.parameter_names, model);

if model.free_cats
    category_params.sigma_1 = p.category_params.sigma_1;
    category_params.sigma_2 = p.category_params.sigma_2;
%else
%    category_params.sigma_1 = 3; % defaults for qamar distributions
%    category_params.sigma_2 = 12;
end

% category_params.sigma_1 = category_params.sigma_1;
% category_params.sigma_2 = category_params.sigma_2;
% category_params.overlap = a;
% category_params.uniform_range = uniform_range;
% category_params.sigma_s = sigma_s;
% category_params.mu_1 = mu_1;
% category_params.mu_2 = mu_2;

[raw.C, raw.s, raw.sig, raw.Chat] = deal(zeros(1, n_samples));
if ~model.choice_only
    raw.g = zeros(1, n_samples);
end

if isempty(model_fitting_data)
    raw.C         = randsample([-1 1], n_samples, 'true');
    raw.contrast  = randsample(contrasts, n_samples, 'true'); % if no p, contrasts == sig
    raw.s(raw.C == -1) = stimulus_orientations(category_params, dist_type, 1, 1, sum(raw.C ==-1));
    raw.s(raw.C ==  1) = stimulus_orientations(category_params, dist_type, 2, 1, sum(raw.C == 1));

else % take real data
    raw.C           = model_fitting_data.C;
    raw.contrast    = model_fitting_data.contrast;
    raw.s           = model_fitting_data.s;
end

[raw.contrast_values, raw.contrast_id] = unique_contrasts(raw.contrast);
sigs = sqrt(p.sigma_0^2 + p.alpha .* raw.contrast_values .^ - p.beta);
raw.sig = sqrt(p.sigma_0^2 + p.alpha .* raw.contrast .^ - p.beta);
raw.sig = reshape(raw.sig,1,length(raw.sig));

if model.ori_dep_noise
    pre_sig = raw.sig;
    raw.sig = pre_sig + abs(sin(raw.s*pi/90))*p.sig_amplitude;
end

raw.x = raw.s + randn(size(raw.sig)) .* raw.sig; % add noise to s. this line is the same in both tasks

% calculate d(x)
switch dist_type
    case 'same_mean_diff_std'
        if strcmp(model.family,'opt')
            if model.non_overlap
                raw.d = zeros(1, n_samples);
                for c = 1 : nContrasts; % for each sigma level, generate d from the separate function
                    cursig = sqrt(p.sigma_0^2 + p.alpha .* contrasts(c) .^ - p.beta);
                    s=trun_sigstruct(cursig,category_params.sigma_1,category_params.sigma_2);
                    raw.d(raw.sig==cursig) = trun_da(raw.x(raw.sig==cursig), s);
                end
            elseif model.ori_dep_noise
                ds = 1;
                sVec = -90:ds:90;
                s_mat = repmat(sVec',1,length(raw.x));
                x_mat = repmat(raw.x,length(sVec),1);
                sig_mat=repmat(pre_sig, length(sVec), 1);
                raw.d = log((1/category_params.sigma_1 * sum((1./(sig_mat+abs(sin(s_mat*pi/90))*p.sig_amplitude)).*exp(-((x_mat-s_mat).^2)./(2*(sig_mat+abs(sin(s_mat*pi/90))*p.sig_amplitude).^2) - s_mat.^2 ./ (2*category_params.sigma_1^2)))) ./ ...
                            (1/category_params.sigma_2 * sum((1./(sig_mat+abs(sin(s_mat*pi/90))*p.sig_amplitude)).*exp(-((x_mat-s_mat).^2)./(2*(sig_mat+abs(sin(s_mat*pi/90))*p.sig_amplitude).^2) - s_mat.^2 ./ (2*category_params.sigma_2^2)))));
%                 raw.d = log((1/category_params.sigma_1 * sum((1./sig_mat).*exp(-((x_mat-s_mat).^2)./(2*sig_mat.^2) - s_mat.^2 ./ (2*category_params.sigma_1^2)))) ./ ...
%                             (1/category_params.sigma_2 * sum((1./sig_mat).*exp(-((x_mat-s_mat).^2)./(2*sig_mat.^2) - s_mat.^2 ./ (2*category_params.sigma_2^2)))));
            else
                raw.k1 = .5 * log( (raw.sig.^2 + category_params.sigma_2^2) ./ (raw.sig.^2 + category_params.sigma_1^2));% + p.b_i(5);
                raw.k2 = (category_params.sigma_2^2 - category_params.sigma_1^2) ./ (2 .* (raw.sig.^2 + category_params.sigma_1^2) .* (raw.sig.^2 + category_params.sigma_2^2));
                raw.d = raw.k1 - raw.k2 .* raw.x.^2;
            end
            %raw.posterior = 1 ./ (1 + exp(-raw.d));
        end
        
    case 'half_gaussian'        
        mu = (raw.x.* category_params.sigma_s^2)./(raw.sig.^2 + category_params.sigma_s^2);
        k = raw.sig .* category_params.sigma_s ./ sqrt(raw.sig.^2 + category_params.sigma_s^2);
        raw.d = log(normcdf(0,mu,k)./normcdf(0,-mu,k));
        
    case 'sym_uniform'
        denom = raw.sig * sqrt(2);
        raw.d = log( (erf((raw.x-a)./denom) - erf((raw.x+1-a)./denom)) ./ (erf((raw.x-1+a)./denom) - erf((raw.x+a)./denom)));
        
    case 'diff_mean_same_std'
        raw.d = (2*raw.x * (category_params.mu_1 - category_params.mu_2) - category_params.mu_1^2 + category_params.mu_2^2) ./ ...
            (2*(raw.sig.^2 + category_params.sigma_s^2));
        
    otherwise
        error('DIST_TYPE is not valid.')
        
end



confidences = [linspace(conf_levels,1,conf_levels) linspace(1,conf_levels,conf_levels)];

if strcmp(model.family,'opt') % for all opt family models
    if model.d_noise% add D noise
        raw.d = raw.d + p.sigma_d * randn(size(raw.d));
    end
    raw.d(raw.d==Inf)  =  1e6;
    raw.d(raw.d==-Inf) = -1e6;

    raw.Chat(raw.d >= p.b_i(5)) = -1;
    raw.Chat(raw.d < p.b_i(5)) = 1;

    if ~model.choice_only
        for g = 1 : conf_levels * 2;
            raw.g( p.b_i(g) <= raw.d ...
                 & raw.d    <= p.b_i(g+1)) = confidences(g);
        end
    end
    
elseif strcmp(model.family, 'MAP')
    raw.shat = zeros(1,3240);
    for i = 1:nContrasts
        sig = sigs(i);
        idx = find(raw.contrast_id==i);

        switch dist_type
            case 'same_mean_diff_std'
                k1 = sqrt(1/(sig^-2 + category_params.sigma_1^-2));
                mu1 = raw.x(idx)*sig^-2 * k1^2;
                k2 = sqrt(1/(sig^-2 + category_params.sigma_2^-2));
                mu2 = raw.x(idx)*sig^-2 * k2^2;
                
                w1 = normpdf(raw.x(idx),0,sqrt(category_params.sigma_1^2 + sig^2));
                w2 = normpdf(raw.x(idx),0,sqrt(category_params.sigma_2^2 + sig^2));
                
                raw.shat(idx) = gmm1max_n2_fast([w1' w2'], [mu1' mu2'], repmat([k1 k2],length(idx),1));

            case 'diff_mean_same_std'
                k = sqrt(1/(sig^-2 + category_params.sigma_s^-2));
                mu1 = (raw.x(idx)*sig^-2 + category_params.mu_1*category_params.sigma_s^-2) * k^2;
                mu2 = (raw.x(idx)*sig^-2 + category_params.mu_2*category_params.sigma_s^-2) * k^2;
                
                w1 = exp(raw.x(idx)*category_params.mu_1./(category_params.sigma_s^2 + sig^2));
                w2 = exp(raw.x(idx)*category_params.mu_2./(category_params.sigma_s^2 + sig^2)); % i got this order by fiddling. it might be better to indicate this as -mu_2 rather than +mu_1

                raw.shat(idx) = gmm1max_n2_fast([w1' w2'], [mu1' mu2'], repmat([k k],length(idx),1));
% %%                
%                 psx=normpdf(s,x,sig).*(normpdf(s,category_params.mu_1,category_params.sigma_s)+normpdf(s,category_params.mu_2,category_params.sigma_s));
%                 plot(s,psx./sum(psx))
%                 B = (x.^2*sig^-2 + category_params.mu_1^2 * category_params.sigma_s^-2) * k^2;
%                 psx2=exp(-(s-mu1).^2./(2*k^2)).*exp(-(B-mu1^2)./(2*k^2)) + exp(-(s-mu2).^2./(2*k^2)).*exp(-(B-mu2^2)./(2*k^2))
%                 hold on
%                 plot(s,psx2./sum(psx2))
%                 
%                 psx3=normpdf(s,mu1,k).*exp(-(B-mu1^2)./(2*k^2)) + normpdf(s,mu2,k).*exp(-(B-mu2^2)./(2*k^2));
%                 
%                 psx4=normpdf(s,mu1,k).*exp(x*category_params.mu_1./(category_params.sigma_s^2 + sig^2)) + normpdf(s,mu2,k).*exp(x*category_params.mu_2./(category_params.sigma_s^2 + sig^2));
%                 
%                 
                
        end
    end
    save tgtest
    plot(raw.x,raw.shat,'.');
    
    b = p.b_i(5);
    raw.Chat(abs(raw.shat) <= b) = -1;
    raw.Chat(abs(raw.shat) >  b) =  1;
    
    if ~model.choice_only
        for g = 1 : conf_levels * 2
            raw.g( p.b_i(g)   <  abs(raw.shat) ...
                &  p.b_i(g+1) >= abs(raw.shat)) = confidences(g);
        end
    end
    
else % all measurement models
    if strcmp(model.family, 'lin')
        b = p.b_i(5) + p.m_i(5) * raw.sig;
    elseif strcmp(model.family, 'quad')
        b = p.b_i(5) + p.m_i(5) * raw.sig.^2;
    else
        b = p.b_i(5);
    end
    
    if strcmp(dist_type, 'same_mean_diff_std')
        x_tmp=abs(raw.x);
    elseif strcmp(dist_type, 'diff_mean_same_std')
        x_tmp=raw.x;
    end
        
    raw.Chat(x_tmp <= b)   = -1;
    raw.Chat(x_tmp >  b)   =  1;
%     if strcmp(dist_type, 'diff_mean_same_std')
%        raw.Chat = -raw.Chat;
%     end
    
    if ~model.choice_only % all non-optimal confidence models
        for g = 1 : conf_levels * 2
            if strcmp(model.family, 'lin')
                raw.g( p.b_i(g) + p.m_i(g) * raw.sig < x_tmp ...
                    &  p.b_i(g+1) + p.m_i(g+1) * raw.sig >= x_tmp) = confidences(g);
            elseif strcmp(model.family, 'quad')
                raw.g( p.b_i(g) + p.m_i(g) * raw.sig.^2 < x_tmp ...
                    &  p.b_i(g+1) + p.m_i(g+1) * raw.sig.^2 >= x_tmp) = confidences(g);
            else
                raw.g( p.b_i(g)   <  x_tmp ...
                    &  p.b_i(g+1) >= x_tmp) = confidences(g);
            end
        end
    end
end


% LAPSE TRIALS %%%%%%%%%%%%%%
randvals = rand(1, n_samples);

if model.multi_lapse
    cuml=[0 cumsum(p.lambda_i)]; % cumulative confidence lapse rate
    Chat_lapse_rate = cuml(end); 
    
    for l = 1 : conf_levels
        lapse_trials = randvals > cuml(l)...
                     & randvals < cuml(l+1);
        raw.g(lapse_trials) = l;
    end
    
else % models with full lapse
    Chat_lapse_rate = p.lambda;
end
Chat_lapse_trials = randvals < Chat_lapse_rate; % lapse Chat at each conf level
n_Chat_lapse_trials = sum(Chat_lapse_trials);
raw.Chat(Chat_lapse_trials) = randsample([-1 1], n_Chat_lapse_trials, 'true');
if ~model.choice_only && ~model.multi_lapse
    raw.g(Chat_lapse_trials) = randsample(conf_levels, n_Chat_lapse_trials, 'true');
end

if model.partial_lapse
    partial_lapse_rate = p.lambda_g;
    partial_lapse_trials = randvals > Chat_lapse_rate...
                         & randvals < Chat_lapse_rate + p.lambda_g;
    n_partial_lapse_trials = sum(partial_lapse_trials);
    raw.g(partial_lapse_trials) = randsample(conf_levels, n_partial_lapse_trials, 'true');
else
    partial_lapse_rate = 0;
end

if model.repeat_lapse
    repeat_lapse_rate = p.lambda_r;
    repeat_lapse_trials = find(randvals > Chat_lapse_rate + partial_lapse_rate & randvals < Chat_lapse_rate + partial_lapse_rate + repeat_lapse_rate);
    if ~model.choice_only
        raw.g(repeat_lapse_trials) = raw.g(max(1,repeat_lapse_trials-1)); % max(1,etc) is to avoid issues when the first trial is chosen to be a repeat lapse (impossible)
    end
    raw.Chat(repeat_lapse_trials) = raw.Chat(max(1,repeat_lapse_trials-1));
else
    repeat_lapse_rate = 0; % this is in case we come up with another kind of lapsing.
end


if ~model.choice_only
    raw.resp  = raw.g + conf_levels + ... % combine conf and class to give resp on 8 point scale
        (raw.Chat * .5 -.5) .* (2 * raw.g - 1);
end

raw.tf = raw.Chat == raw.C;