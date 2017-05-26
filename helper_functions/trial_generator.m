function raw = trial_generator(p_in, model, varargin)

% define defaults
n_samples = 2160; % might be overwritten by number of trials in model_fitting_data
% contrasts  = exp(-4:.5:-1.5);%[.125 .25 .5 1 2 4];

model_fitting_data = [];
conf_levels = 4;

category_params.sigma_1 = 3;
category_params.sigma_2 = 12;
category_params.sigma_s = 5; % for 'diff_mean_same_std' and 'half_gaussian'
category_params.a = 0; % overlap for sym_uniform
category_params.mu_1 = -4; % mean for 'diff_mean_same_std'
category_params.mu_2 = 4;
category_params.uniform_range = 1;

category_type = 'same_mean_diff_std'; % 'same_mean_diff_std' (Qamar) or 'diff_mean_same_std' or 'sym_uniform' or 'half_gaussian' (Kepecs)

attention_manipulation = false;

multi_prior = false;
contrasts = [];

nn_d = false; % generate d from spikes instead of from x
nn_baseline = 0;
assignopts(who,varargin);

% updating category_type according to the model. not sure why i wasn''t doing this before
if model.diff_mean_same_std
    category_type = 'diff_mean_same_std';
elseif ~model.diff_mean_same_std
    category_type = 'same_mean_diff_std';
end

if isempty(model_fitting_data)
    if isempty(contrasts)
        if ~attention_manipulation
            contrasts = exp(linspace(-5.5,-2,6));
        else
            contrasts = .08;
        end
    end
    raw.C         = randsample([-1 1], n_samples, 'true');
    raw.contrast  = randsample(contrasts, n_samples, 'true'); % if no p, contrasts == sig
    raw.s(raw.C == -1) = stimulus_orientations(category_params, 1, sum(raw.C ==-1), category_type);
    raw.s(raw.C ==  1) = stimulus_orientations(category_params, 2, sum(raw.C == 1), category_type);
    
    
    if attention_manipulation
        if model.nFreesigs==3
            v = .8;
            prop_neutral_trials = 1/6;
            cue_validities = [(1-v)/3 .25 v];
            freq = [(1-prop_neutral_trials)*(1-v) prop_neutral_trials (1-prop_neutral_trials)*v];
        elseif model.nFreesigs==5
            v = .9;
            v2= .45;
            cue_validities = [(1-v)/3 (1-2*v2)/2 .25 v2 v];
            freq = [(1/3)*(1-v) (1/3)*(1-2*v2) 1/3 (1/3)*2*v2 (1/3)*v];
        end
        raw.cue_validity = rand(1, n_samples);
        temp_freq = [0 cumsum(freq)];
        for i = 1:model.nFreesigs
            raw.cue_validity(raw.cue_validity>temp_freq(i) & raw.cue_validity<temp_freq(i+1)) = i;
        end
        % do we need probe and cue?
        
%         cue_validity = .7;
%         
%         raw.cue = randsample([-1 0 1], n_samples, 'true');
%         raw.probe = raw.cue;
%         
%         % make 30% of cues invalid
%         flip_idx = rand(1, n_samples) > cue_validity;
%         raw.cue(flip_idx) = -raw.cue(flip_idx);
%         
%         neutral_cue_idx = raw.cue == 0;
%         raw.probe(neutral_cue_idx) = randsample([-1 1], nnz(neutral_cue_idx), true);
%         
%         raw.cue_validity(raw.probe == raw.cue)                  =  1;  % valid cues
%         raw.cue_validity(raw.cue == 0)                          =  0;  % neutral cues
%         raw.cue_validity(raw.probe ~= raw.cue & raw.cue ~= 0)   = -1; % invalid cues
    elseif multi_prior
        % DO THIS
    end
    
else % take real data
    contrasts = model_fitting_data.contrast_values;
    
    raw.C           = model_fitting_data.C;
    raw.contrast    = model_fitting_data.contrast;
    raw.s           = model_fitting_data.s;
    
    if attention_manipulation
        raw.probe        = model_fitting_data.probe;
        raw.cue          = model_fitting_data.cue;
        raw.cue_validity = model_fitting_data.cue_validity;
    end
end

p = parameter_variable_namer(p_in, model.parameter_names, model, contrasts);


nContrasts = length(contrasts);
n_samples = length(raw.C);

if ~model.free_cats
    p.sig1 = category_params.sigma_1;
    p.sig2 = category_params.sigma_2;
end

[raw.sig, raw.Chat] = deal(zeros(1, n_samples));
if ~model.choice_only
    raw.g = zeros(1, n_samples);
end

if ~attention_manipulation
    [raw.contrast_values, raw.contrast_id] = unique_contrasts(raw.contrast, 'flipsig', true); % contrast_values is in descending order. so a high contrast_id indicates a lower contrast value, and a higher sigma value.
    raw.sig = p.unique_sigs(raw.contrast_id);
    if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
        raw.sig_inference = p.unique_sigs_inference(raw.contrast_id);
    end
    %     c_low = min(raw.contrast_values);
    %     c_hi = max(raw.contrast_values);
    %     alpha = (p.sigma_c_low^2-p.sigma_c_hi^2)/(c_low^-p.beta - c_hi^-p.beta);
    %     sigs =    sqrt(p.sigma_c_low^2 - alpha * c_low^-p.beta + alpha * raw.contrast_values .^ -p.beta); % the list of possible sigma values
    %     raw.sig = sqrt(p.sigma_c_low^2 - alpha * c_low^-p.beta + alpha * raw.contrast        .^ -p.beta); % sigma values on every trial
elseif attention_manipulation
    [raw.contrast_values, raw.contrast_id] = unique_contrasts(raw.contrast);
    [raw.cue_validity_values, raw.cue_validity_id] = unique_contrasts(raw.cue_validity);
    raw.sig = p.unique_sigs(raw.cue_validity_id);
    if model.separate_measurement_and_inference_noise
        raw.sig_inference = p.unique_sigs_inference(raw.cue_validity_id);
    end
    %     raw.sig(raw.cue_validity  == -1) = p.sigma_c_low; % invalid cue -> high sigma (c_low means low "contrast")
    %     raw.sig(raw.cue_validity  ==  0) = p.sigma_c_mid;
    %     raw.sig(raw.cue_validity ==  1) = p.sigma_c_hi;
end
raw.sig = reshape(raw.sig,1,length(raw.sig)); % make sure it's a row.
if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
    raw.sig_inference = reshape(raw.sig_inference, 1, length(raw.sig_inference));
end

if model.ori_dep_noise
    ODN = @(s, sig_amplitude) abs(sin(s * pi / 90)) * sig_amplitude;
%     pre_sig = raw.sig;
%     raw.sig = pre_sig + ODN(raw.s, p.sig_amplitude);
%     if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
%         pre_sig_inference = raw.sig_inference;
%         raw.sig_inference = pre_sig_inference + ODN(raw.s, p.sig_amplitude_inference);
%     end
    if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
        pre_sig = raw.sig_inference;
        raw.sig_inference = pre_sig + ODN(raw.s, p.sig_amplitude_inference);
    else
        pre_sig = raw.sig;
        raw.sig = pre_sig + ODN(raw.s, p.sig_amplitude);
    end
end

if strcmp(model.family, 'neural1')
    %%
    %     figure
    %     nTrials = 10000; %%%%%
    %     p.sigma_tc = exp(2.5); %%%%%
    %     raw.s = randn(1,nTrials)*12; %%%%%
    %     raw.sig = 1*ones(1, nTrials); %%%%%
    g = 1./(raw.sig.^2);
    neural_mu = g .* raw.s .* sqrt(2*pi*p.sigma_tc^2);
    neural_sig = sqrt(g .* (p.sigma_tc^2 + raw.s.^2) * sqrt(2*pi*p.sigma_tc^2));
    raw.x = neural_mu + neural_sig .* randn(size(raw.sig));
    %     plot(raw.s, raw.x, '.') %%%%%
elseif nn_d
    raw.x = nan(size(raw.sig));
else
    raw.x = raw.s + randn(size(raw.sig)) .* raw.sig; % add noise to s. this line is the same in both tasks
end

if model.ori_dep_noise && strcmp(model.family, 'opt')
    ds = 1;
    sVec = -90:ds:90;
    s_mat = repmat(sVec',1, n_samples);
    x_mat = repmat(raw.x,length(sVec),1);
    
    %     if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
    %         sig_mat = repmat(pre_sig_inference, length(sVec), 1);
    %         sig_plusODN_mat = sig_mat + ODN(s_mat, p.sig_amplitude_inference);
    %     else
    sig_mat=repmat(pre_sig, length(sVec), 1); % nTrials vector of sigma levels repeated some number of rows defined by ds
    if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
        sig_plusODN_mat = sig_mat + ODN(s_mat, p.sig_amplitude_inference);
    else
        sig_plusODN_mat = sig_mat + ODN(s_mat, p.sig_amplitude);
    end
    
    % p(x|C). see conf data likelihood my task.pages>orientation dependent noise
    likelihood = @(sigma_cat, mu_cat) 1/sigma_cat * sum(1 ./sig_plusODN_mat .*exp(-(x_mat-s_mat).^2 ./ (2*sig_plusODN_mat.^2) - (s_mat - mu_cat).^2 ./ (2*sigma_cat^2)));
end



% calculate d(x)
if strcmp(model.family,'opt')
    if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
        assumed_sig = raw.sig_inference; % assumed sig is not the same as the sig that generated the data
    else
        assumed_sig = raw.sig; % assumed sig is accurate, and the same as the generative sig
    end
    
    switch category_type
        case 'same_mean_diff_std'
            if model.non_overlap
                raw.d = zeros(1, n_samples);
                for c = 1 : nContrasts; % for each sigma level, generate d from the separate function
                    cursig = sqrt(p.sigma_0^2 + p.alpha .* contrasts(c) .^ - p.beta);
                    s=trun_sigstruct(cursig,category_params.sigma_1,category_params.sigma_2);
                    raw.d(assumed_sig==cursig) = trun_da(raw.x(assumed_sig==cursig), s);
                end
            elseif model.ori_dep_noise
                raw.d = log(likelihood(p.sig1, 0) ./ likelihood(p.sig2, 0));
            elseif nn_d
                [raw.spikes, ~, raw.d] = generate_popcode(raw.C', raw.s', raw.sig',...
                    'sig1_sq', category_params.sigma_1^2, ...
                    'sig2_sq', category_params.sigma_2^2, ...
                    'baseline', nn_baseline);
                raw.spikes = raw.spikes';
                raw.d = raw.d';
            else
                raw.k1 = .5 * log( (assumed_sig.^2 + p.sig2^2) ./ (assumed_sig.^2 + p.sig1^2));% + p.b_i(5);
                raw.k2 = (p.sig2^2 - p.sig1^2) ./ (2 .* (assumed_sig.^2 + p.sig1^2) .* (assumed_sig.^2 + p.sig2^2));
                raw.d = raw.k1 - raw.k2 .* raw.x.^2;
            end
            %raw.posterior = 1 ./ (1 + exp(-raw.d));
            
        case 'half_gaussian'
            mu = (raw.x.* category_params.sigma_s^2)./(assumed_sig.^2 + category_params.sigma_s^2);
            k = assumed_sig .* category_params.sigma_s ./ sqrt(assumed_sig.^2 + category_params.sigma_s^2);
            raw.d = log(normcdf(0,mu,k)./normcdf(0,-mu,k));
            
        case 'sym_uniform'
            denom = assumed_sig * sqrt(2);
            raw.d = log( (erf((raw.x-a)./denom) - erf((raw.x+1-a)./denom)) ./ (erf((raw.x-1+a)./denom) - erf((raw.x+a)./denom)));
            
        case 'diff_mean_same_std'
            
            if model.ori_dep_noise
                raw.d = log(likelihood(category_params.sigma_s, category_params.mu_1) ./ likelihood(category_params.sigma_s, category_params.mu_2));
            else
                raw.d = (2*raw.x * (category_params.mu_1 - category_params.mu_2) - category_params.mu_1^2 + category_params.mu_2^2) ./ ...
                    (2*(assumed_sig.^2 + category_params.sigma_s^2));
            end
        otherwise
            error('DIST_TYPE is not valid.')
    end
    
end


confidences = [linspace(conf_levels,1,conf_levels) linspace(1,conf_levels,conf_levels)];

if strcmp(model.family,'opt') % for all opt family models
    if model.d_noise% add D noise
        raw.d = raw.d + p.sigma_d * randn(size(raw.d));
    end
    raw.d(raw.d==Inf)  =  1e6;
    raw.d(raw.d==-Inf) = -1e6;
    
    if ~isfield(model,'fisher_info') || ~model.fisher_info
        raw.Chat(raw.d >= p.b_i(5)) = -1;
        raw.Chat(raw.d < p.b_i(5)) = 1;
        
        if ~model.choice_only
            for g = 1 : conf_levels * 2
                raw.g( p.b_i(g) <= raw.d ...
                    & raw.d    <= p.b_i(g+1)) = confidences(g);
            end
        end

    else
        raw.d = raw.d + p.fisher_prior;
        raw.Chat(raw.d >= 0) = -1;
        raw.Chat(raw.d < 0) = 1;
        
        raw.d = 1./(1+exp(raw.Chat.*raw.d)) + p.fisher_weight.*assumed_sig.^-2;
        
        if ~model.choice_only
            for g = 1 : conf_levels
                raw.g( p.b_i(g) <= raw.d ...
                    & raw.d    <= p.b_i(g+1)) = g;
            end
        end
    end
    
        
    
elseif strcmp(model.family, 'MAP')
    if isfield(model, 'separate_measurement_and_inference_noise') && model.separate_measurement_and_inference_noise
        assumed_sig = p.unique_sigs_inference; % assumed sig is not the same as the sig that generated the data
    else
        assumed_sig = p.unique_sigs; % assumed sig is accurate, and the same as the generative sig
    end
    
    if ~model.ori_dep_noise
        raw.shat = zeros(1,n_samples);
        
        switch category_type
            case 'same_mean_diff_std' % task B
                k1sq = 1./(assumed_sig.^-2 + p.sig1^-2);
                k2sq = 1./(assumed_sig.^-2 + p.sig2^-2);
                k1 = sqrt(k1sq);
                k2 = sqrt(k2sq);
                
            case 'diff_mean_same_std' % task A
                ksq = 1./(assumed_sig.^-2 + category_params.sigma_s^-2);
                k = sqrt(ksq);
        end
        
        for i = 1:nContrasts
            cur_sig = assumed_sig(i);
            idx = find(raw.contrast_id==i);
            
            switch category_type
                case 'same_mean_diff_std'
                    %                 k1 = sqrt(1/(sig^-2 + p.sig1^-2));
                    mu1 = raw.x(idx)*cur_sig^-2 * k1sq(i);
                    %                 k2 = sqrt(1/(sig^-2 + p.sig2^-2));
                    mu2 = raw.x(idx)*cur_sig^-2 * k2sq(i);
                    
                    w1 = exp(raw.x(idx)./(p.sig1^2 + cur_sig^2));
                    w2 = exp(raw.x(idx)./(p.sig2^2 + cur_sig^2));
                    
                    raw.shat(idx) = gmm1max_n2_fast([w1' w2'], [mu1' mu2'], repmat([k1(i) k2(i)],length(idx),1));
                    
                case 'diff_mean_same_std'
                    %                 k = sqrt(1/(sig^-2 + category_params.sigma_s^-2));
                    mu1 = (raw.x(idx)*cur_sig^-2 + category_params.mu_1*category_params.sigma_s^-2) * ksq(i);
                    mu2 = (raw.x(idx)*cur_sig^-2 + category_params.mu_2*category_params.sigma_s^-2) * ksq(i);
                    
                    w1 = exp(raw.x(idx)*category_params.mu_1./(category_params.sigma_s^2 + cur_sig^2));
                    w2 = exp(raw.x(idx)*category_params.mu_2./(category_params.sigma_s^2 + cur_sig^2));
                    
                    raw.shat(idx) = gmm1max_n2_fast([w1' w2'], [mu1' mu2'], repmat([k(i) k(i)],length(idx),1));
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
    elseif model.ori_dep_noise
        sSteps = 600;
        sVec = reshape(linspace(-60,60,sSteps), 1, 1, sSteps);
        
        if ~model.diff_mean_same_std % task B
            logprior = log(1/(2*sqrt(2*pi)) * (p.sig1^-1 * exp(-sVec.^2 / (2*p.sig1^2)) + p.sig2^-1 * exp(-sVec.^2 / (2*p.sig2^2))));
        elseif model.diff_mean_same_std % task A
            logprior = log(1/(2*category_params.sigma_s*sqrt(2*pi)) * (exp(-(sVec-category_params.mu_1).^2 / (2*category_params.sigma_s^2)) + exp(-(sVec-category_params.mu_2).^2 / (2*category_params.sigma_s^2))));
        end
        
        loglikelihood = bsxfun_normlogpdf(raw.x, sVec, raw.sig);
        
        logposterior = bsxfun(@plus, loglikelihood, logprior);
        
        raw.shat = qargmax1(sVec, logposterior, 3);
        
    end
    
    b = p.b_i(5);
    
    if strcmp(category_type, 'same_mean_diff_std')
        shat_tmp = abs(raw.shat);
    elseif strcmp(category_type, 'diff_mean_same_std')
        shat_tmp = raw.shat;
    end
    raw.Chat(shat_tmp <= b) = -1;
    raw.Chat(shat_tmp >  b) =  1;
    
    if ~model.choice_only
        for g = 1 : conf_levels * 2
            raw.g( p.b_i(g)   <  shat_tmp ...
                &  p.b_i(g+1) >= shat_tmp) = confidences(g);
        end
    end
    
else % all non-Bayesian models
    if strcmp(model.family, 'lin')
        b = p.b_i(5) + p.m_i(5) * raw.sig;
    elseif strcmp(model.family, 'quad')
        b = p.b_i(5) + p.m_i(5) * raw.sig.^2;
    else % fixed and neural
        if ~isfield(model, 'nFreebounds') || model.nFreebounds == 0
            b = p.b_i(5);
        else
            % choose the choice bound column, row determined by cue_validity
            b = p.b_i(sub2ind(size(p.b_i), model.nFreebounds + 1 - raw.cue_validity_id, 5*ones(size(raw.cue_validity_id))));
        end
    end
    
    if strcmp(category_type, 'same_mean_diff_std')
        x_tmp=abs(raw.x);
    elseif strcmp(category_type, 'diff_mean_same_std')
        x_tmp=raw.x;
    end
    
    raw.Chat(x_tmp <= b)   = -1;
    raw.Chat(x_tmp >  b)   =  1;
    %     if strcmp(category_type, 'diff_mean_same_std')
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
            else % fixed and neural
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

if ~isfield(model, 'biased_lapse') || isempty(model.biased_lapse) || ~model.biased_lapse
    p.lambda_bias = .5; % p_lapse(Chat = -1)
end

lapse_Chat = rand(1, n_Chat_lapse_trials);
lapse_Chat(lapse_Chat < p.lambda_bias) = -1;
lapse_Chat(lapse_Chat >= p.lambda_bias) = 1;
raw.Chat(Chat_lapse_trials) = lapse_Chat;

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
    % combine conf and class to give resp on 8 point scale
    raw.resp  = raw.Chat .* raw.g - .5 * (raw.Chat+1) + conf_levels + 1;
    %
    %     raw.g + conf_levels + ...
    %         (raw.Chat * .5 -.5) .* (2 * raw.g - 1);
end

raw.tf = raw.Chat == raw.C;