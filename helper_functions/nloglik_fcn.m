function [nll, ll_trials] = nloglik_fcn(p_in, raw, model, nDNoiseSets, varargin)

lapse_sum = lapse_rate_sum(p_in, model);

if lapse_sum > 1
    nll = Inf;
    ll_trials = -inf(size(raw.Chat));
    %     warning('lapse_sum > 1')
    return
end

if length(varargin) == 1;
    category_params = varargin{1};
end

if ~isfield(model, 'separate_measurement_and_inference_noise') % this should be temporary! take it out after the 7/2015 jobs.
    model.separate_measurement_and_inference_noise = 0;
end


% % constraints for optimization algorithms that don't have constraints as inputs. this should be a wrapper for the function
% if length(varargin) == 2 | length(varargin) == 3
%     alg = varargin{2};
%     if strcmp(alg,'snobfit') | strcmp(alg,'mcs') % opt algorithms that don't have linear constraints built in
%         c = varargin{1};
%         p_in = reshape(p_in,length(p_in),1);
%         if any(c.A * p_in > c.b) || any(p_in < c.lb') || any(p_in > c.ub')
%             nloglik = 1e8;
%             %disp('violation!')
%             return
%         end
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~model.nFreesigs
    p = parameter_variable_namer(p_in, model.parameter_names, model, raw.contrast_values);
else
    p = parameter_variable_namer(p_in, model.parameter_names, model);
end

if isfield(raw, 'cue_validity_id')
    raw.contrast_id = raw.cue_validity_id;
    if isfield(raw, 'cue_validity_values')
        raw.contrast_values = raw.cue_validity_values;
    end
end
nContrasts = length(raw.contrast_values);

nTrials = length(raw.s);

nPriors = length(raw.prior_values);
logprior = log(raw.prior_values./(1-raw.prior_values));

if ~model.d_noise
    nDNoiseSets = 1;
    normalized_weights = 1;
    d_noise_draws = 0;
elseif model.d_noise
    if ~exist('nDNoiseSets','var')
        nDNoiseSets=101;
    end
    nSTDs = 5;
    weights = my_normpdf(linspace(-nSTDs, nSTDs, nDNoiseSets), 0, 1);
    normalized_weights = weights ./ sum(weights);
    
    d_noise_draws = linspace(-p.sigma_d*nSTDs, p.sigma_d*nSTDs, nDNoiseSets);
end
d_noise = repmat(d_noise_draws',1,nContrasts);
d_noise_big = repmat(d_noise_draws',1,nTrials);% this is for non_overlap. too hard to figure out the indexing. going to be too many computations, because there's redundancy in the a,b,k vectors. but the bulk of computation has to be on an individual trial basis anyway.

if isfield(p,'b_i')
    conf_levels = (length(p.b_i) - 1)/2;
    nBounds = conf_levels*2-1;
else
    conf_levels = 0;
end

if model.free_cats
    sig1 = p.sig1;
    sig2 = p.sig2;
else
    sig1 = 3; % defaults for qamar distributions
    sig2 = 12;
end

xSteps = 201; %200 and 20 take about the same amount of time
if ~model.diff_mean_same_std
    xVec = linspace(0,90,xSteps)';
else
    xVec = linspace(-45,45,xSteps)';
end

% clean this up!
if model.ori_dep_noise
    ODN = @(s, sig_amplitude) abs(sin(s * pi / 90)) * sig_amplitude;
    
    if strcmp(model.family, 'opt')
        d_lookup_table = zeros(nContrasts,length(xVec));
        x_dec_bound = zeros(nDNoiseSets,nContrasts);

        sSteps = 199;
        sVec = linspace(-100,100,sSteps)';
%         s_mat = repmat(sVec,1, xSteps);

%         x_mat = repmat(xVec', sSteps,1);
        
        
%         likelihood = @(sigma, sigma_cat, mu_cat) 1/sigma_cat * sum(1 ./ sigma .*exp(-(x_mat-s_mat).^2 ./ (2*sigma.^2) - (s_mat - mu_cat) .^2 ./ (2*sigma_cat^2))); % 7.3 secs
%         % equivalently (after normalization or ratio), more readable, but slower. doesn't require computation of x_mat, s_mat, or ODN_s_mat.
%         likelihood2 = @(sigma, sigma_cat, mu_cat) sum(normpdf(x_mat,s_mat,sigma) .* normpdf(s_mat, mu_cat, sigma_cat));
        likelihood = @(sigma, sigma_cat, mu_cat) sum(bsxfun(@times, bsxfun_normpdf(xVec', sVec, sigma), normpdf(sVec, mu_cat, sigma_cat)));

        
        % this is reversed from trial_generator. there, we generate x first, with regular sigs. here, we get the boundaries first, with sigs_inference
        if model.separate_measurement_and_inference_noise
            sig = p.unique_sigs_inference;
           	ODN_s_mat = ODN(sVec, p.sig_amplitude_inference);
        else
            sig = p.unique_sigs;
            ODN_s_mat = ODN(sVec, p.sig_amplitude);
        end
        
    else
        if model.separate_measurement_and_inference_noise
            sig = p.unique_sigs_inference(raw.contrast_id);
            sig = sig + ODN(raw.s, p.sig_amplitude_inference);
        else
            sig = p.unique_sigs(raw.contrast_id); % 1 x nTrials vector of sigmas
            sig = sig + ODN(raw.s, p.sig_amplitude); % add orientation dependent noise to each sigma.
        end
    end
else
    % this is reversed from trial_generator.m. there, we generate x first, with regular sigs. here, we get the boundaries first, with sigs_inference
    if model.separate_measurement_and_inference_noise
        sig = p.unique_sigs_inference;
    else
        sig = p.unique_sigs;
    end
end

    function [d_lookup_table, k] = d_table_and_choice_bound(sig_cat1, sig_cat2, mu_cat1, mu_cat2)
        sig_plusODN = reshape(bsxfun(@plus, ODN_s_mat, p.unique_sigs), [sSteps 1 nContrasts]);
        LLR = log(likelihood(sig_plusODN, sig_cat1, mu_cat1) ./ likelihood(sig_plusODN, sig_cat2, mu_cat2));
        d_lookup_table = permute(bsxfun(@plus, LLR, logprior'), [3 2 1]);
        
        d_plus_noise = bsxfun(@plus, fliplr(d_lookup_table), permute(d_noise_draws, [4 3 1 2]));
        k = zeros(nDNoiseSets, nContrasts, nPriors);
        for contrast = 1:nContrasts
            for prior = 1:nPriors
                k(:, contrast, prior) = lininterp1_multiple(squeeze(d_plus_noise(contrast, :, prior, :))', fliplr(xVec'), bf(0));
            end
        end
    end

    function [x_lb, x_ub] = x_bounds_by_trial()
        % NEED TO MAKE THIS WORK FOR PRIORS
        x_bounds = zeros(nContrasts, nBounds, nDNoiseSets);
        
        for contrast = 1:nContrasts
            for response = -(conf_levels-1):(conf_levels-1)%1:7 problem at r = 1:3
                x_bounds(nContrasts+1-contrast,conf_levels-response,:) = lininterp1_multiple(bsxfun(@plus, fliplr(d_lookup_table(contrast,:)), d_noise_draws'), fliplr(xVec'), bf(response));
            end
        end
        if ~model.diff_mean_same_std
            x_bounds = [zeros(nContrasts,1,nDNoiseSets) x_bounds inf(nContrasts,1,nDNoiseSets)];
        elseif model.diff_mean_same_std
            x_bounds = [-inf(nContrasts,1,nDNoiseSets) x_bounds inf(nContrasts,1,nDNoiseSets)];
        end
        %cols of x_bounds are for each choice and confidence
        cols_lb = 4 + raw.Chat .* (raw.g - 1); % used to have 5 up here and -1 below
        cols_ub = 4 + raw.Chat .*  raw.g;
        %rows are for each contrast level
        rows = nContrasts - raw.contrast_id; % used to have + 1 up here and - 1 below
        
        % flatten the whole cube into a vector
        x_bounds_squash = reshape(permute(x_bounds,[3 2 1]), numel(x_bounds), 1);
        % each trial is a nDNoiseSets-vector from the x_bounds vector. This defines the first index of each vector
        start_pts_lb = (nDNoiseSets*((nBounds+2)*(rows)+cols_lb)+1)';
        start_pts_ub = (nDNoiseSets*((nBounds+2)*(rows)+cols_ub)+1)';
        
        % this takes those vectors, and turns them back into a matrix.
        if ~model.diff_mean_same_std
%             x_lb = bsxfun(@times, raw.Chat', x_bounds_squash(bsxfun(@plus, start_pts_lb, (0:nDNoiseSets-1))))';
%             x_ub = bsxfun(@times, raw.Chat', x_bounds_squash(bsxfun(@plus, start_pts_ub, (0:nDNoiseSets-1))))';
            
            a = x_bounds_squash(bsxfun(@plus, start_pts_lb, (0:nDNoiseSets-1)))';
            b = x_bounds_squash(bsxfun(@plus, start_pts_ub, (0:nDNoiseSets-1)))';

        elseif model.diff_mean_same_std
            a = x_bounds_squash(bsxfun(@plus, start_pts_lb, (0:nDNoiseSets-1)))';
            b = x_bounds_squash(bsxfun(@plus, start_pts_ub, (0:nDNoiseSets-1)))';
            
        end
        
        ab_cat = cat(3,a,b);
        x_lb = min(ab_cat, [], 3); % this is a hack to get everything in the right order so that x_lb < x_ub
        x_ub = max(ab_cat, [], 3);

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHOICE PROBABILITY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% k is the category decision boundary in measurement space for each trial
% f computes the probability mass on the chosen side of k.
if strcmp(model.family, 'opt') && ~model.diff_mean_same_std% for normal bayesian family, task B
    
    if model.non_overlap
        x_bounds = find_intersect_truncated_cats(p, sig1, sig2, contrasts, d_noise_big, raw);
        
        if ~model.d_noise
            x_dec_bound = fliplr(x_bounds(:,4)');
            x_bounds = [zeros(nContrasts,1) x_bounds inf(nContrasts,1)]; % this is for confidence
        elseif model.d_noise
            % for d noise, need long noisesets x trials matrix
            x_dec_bound = permute(x_bounds(3,:,:),[3 2 1]);
        end
        
    elseif model.ori_dep_noise
        [d_lookup_table, x_dec_bound] = d_table_and_choice_bound(sig1, sig2, 0, 0);
    else
        k1_pre = .5 * log( (sig.^2 + sig2^2) ./ (sig.^2 + sig1^2));
        k1 = repmat(k1_pre, 1, 1, nPriors) + repmat(permute(logprior, [1 3 2]), 1, nContrasts, 1);
        k2 = (sig2^2 - sig1^2) ./ (2 .* (sig.^2 + sig1^2) .* (sig.^2 + sig2^2));
        x_dec_bound = real(sqrt((repmat(k1, nDNoiseSets, 1) + repmat(d_noise,1,1,nPriors) - bf(0)) ./ repmat(k2, nDNoiseSets, 1, nPriors)));
        % using bsxfun instead of repmat takes about 1.3x as much time
    end
elseif strcmp(model.family, 'opt') && model.diff_mean_same_std
    
    if model.ori_dep_noise
        [d_lookup_table, x_dec_bound] = d_table_and_choice_bound(category_params.sigma_s, category_params.sigma_s, category_params.mu_1, category_params.mu_2);        
    else
        x_dec_bound = (2*(bsxfun(@minus, d_noise, permute(logprior, [1 3 2])) + bf(0)).*(repmat(sig, nDNoiseSets, 1, nPriors).^2 + category_params.sigma_s^2) - category_params.mu_2^2 + category_params.mu_1^2)...
            / (2*(category_params.mu_1 - category_params.mu_2));
    end
elseif strcmp(model.family, 'lin') && ~model.diff_mean_same_std
    x_dec_bound = max(bf(0) + mf(0) * sig, 0);
elseif strcmp(model.family, 'lin') && model.diff_mean_same_std
    x_dec_bound = bf(0) + mf(0) * sig;
elseif strcmp(model.family, 'quad') && ~model.diff_mean_same_std
    x_dec_bound = max(bf(0) + mf(0) * sig.^2, 0);
elseif strcmp(model.family, 'quad') && model.diff_mean_same_std
    x_dec_bound = bf(0) + mf(0) * sig.^2;
    
elseif strcmp(model.family, 'fixed') || strcmp(model.family, 'neural1')
    if model.nPriors == 1
        x_dec_bound = bf(0)*ones(1,nContrasts);
    else
        x_dec_bound = bf_prior(0, raw.prior_id)';
    end
    
elseif strcmp(model.family, 'MAP')
    if ~model.ori_dep_noise
        shat_lookup_table = zeros(nContrasts, xSteps);
        niter = 4;
        
        if ~model.diff_mean_same_std % task B
            k1sq = 1./(sig.^-2 + sig1^-2); % like k1 in trial_generator
            k2sq = 1./(sig.^-2 + sig2^-2); % like k2 in trial_generator
            k1 = sqrt(k1sq);
            k2 = sqrt(k2sq);
            
        elseif model.diff_mean_same_std % task A
            ksq = 1./(sig.^-2 + category_params.sigma_s^-2);
            k = sqrt(ksq);
        end
        
        for i = 1:nContrasts
            cur_sig = sig(i);
            
            if ~model.diff_mean_same_std
                mu1 = xVec*cur_sig^-2 * k1sq(i);
                mu2 = xVec*cur_sig^-2 * k2sq(i);
                w1 = my_normpdf(xVec,0,sqrt(sig1^2 + cur_sig^2));
                w2 = my_normpdf(xVec,0,sqrt(sig2^2 + cur_sig^2));
                shat_lookup_table(i,:) = gmm1max_n2_fast([w1 w2], [mu1 mu2], repmat([k1(i) k2(i)],xSteps,1),niter);

            elseif model.diff_mean_same_std
                mu1 = (xVec*cur_sig^-2 + category_params.mu_1*category_params.sigma_s^-2) * ksq(i);
                mu2 = (xVec*cur_sig^-2 + category_params.mu_2*category_params.sigma_s^-2) * ksq(i);
                w1 = exp(xVec*category_params.mu_1./(category_params.sigma_s^2+cur_sig^2));
                w2 = exp(xVec*category_params.mu_2./(category_params.sigma_s^2+cur_sig^2));
                shat_lookup_table(i,:) = gmm1max_n2_fast([w1 w2], [mu1 mu2], repmat([k(i) k(i)],xSteps,1),niter);
            end
        end
    elseif model.ori_dep_noise
        sVec(1,1,:) = xVec(:);
        
        if ~model.diff_mean_same_std % task B
            logprior_s = log(1/(2*sqrt(2*pi)) * (sig1^-1 * exp(-sVec.^2 / (2*sig1^2)) + sig2^-1 * exp(-sVec.^2 / (2*sig2^2))));
        elseif model.diff_mean_same_std % task A
            logprior_s = log(1/(2*category_params.sigma_s*sqrt(2*pi)) * (exp(-(sVec-category_params.mu_1).^2 / (2*category_params.sigma_s^2)) + exp(-(sVec-category_params.mu_2).^2 / (2*category_params.sigma_s^2))));
        end
        
        if model.separate_measurement_and_inference_noise
            noise = bsxfun(@plus, reshape(p.unique_sigs_inference, 6, 1), ODN(sVec, p.sig_amplitude_inference)); % 2d slice dependent on c and s;
        else
            noise = bsxfun(@plus, reshape(p.unique_sigs, 6, 1), ODN(sVec, p.sig_amplitude)); % 2d slice dependent on c and s;
        end
        loglikelihood = bsxfun_normlogpdf(xVec', sVec, noise);
        
        logposterior = bsxfun(@plus, loglikelihood, logprior_s); % 3d
        
        shat_lookup_table = qargmax1(sVec, logposterior, 3);
    end

    x_dec_bound = lininterp1_multiple(shat_lookup_table, xVec, bf(0)*ones(1,nContrasts)); % find the x values corresponding to the MAP criterion, for each contrast level (or, if doing ODN where each trial has a different sigma, for every trial/sigma)
end

%if ~(model.non_overlap && model.d_noise)
% do this for all models except nonoverlap+d noise, where k is already in this form.
if any(size(x_dec_bound) == nContrasts) % x_dec_bound needs to be expanded for each trial
    % index x_dec_bound by trial contrast and prior
    x_dec_bound = reshape(x_dec_bound, [nDNoiseSets, nContrasts*nPriors]);
    x_dec_bound = x_dec_bound(:, (raw.prior_id-1)*nContrasts+raw.contrast_id);
    %x_dec_bound = x_dec_bound(:, raw.contrast_id);
end

% the following two if statements could be joined in some nice way eventually

if numel(sig) == nContrasts % equivalently, if ~(model.ori_dep_noise && ~strcmp(model.family, 'opt'))
    sig = sig(raw.contrast_id); % re-arrange sigs if it hasn't been done yet
    if model.ori_dep_noise
        if model.separate_measurement_and_inference_noise
            sig = sig + ODN(raw.s, p.sig_amplitude_inference);
        else
            sig = sig + ODN(raw.s, p.sig_amplitude);
        end
    end
end

% define generative mu and sigma. get the cdf of this distribution that falls between bounds
if ~strcmp(model.family, 'neural1')
    mu = raw.s;
    if ~model.separate_measurement_and_inference_noise
        measurement_sigma = sig;
    elseif model.separate_measurement_and_inference_noise
        measurement_sigma = p.unique_sigs(raw.contrast_id);
        if model.ori_dep_noise
            measurement_sigma = measurement_sigma + ODN(raw.s, p.sig_amplitude);
        end
    end
else
    g = 1./(sig.^2);
    mu = g .* raw.s .* sqrt(2*pi) * p.sigma_tc;
    measurement_sigma = sqrt(g .* (p.sigma_tc^2 + raw.s.^2) .* sqrt(2*pi) * p.sigma_tc);
end

if ~model.diff_mean_same_std
    % 0.5 + 0.5 * Chat - Chat *symmetric_normcdf(x_dec_bound, s, measurement_sigma)
%     p_choice = 0.5 + 0.5 * repmat(raw.Chat, nDNoiseSets, 1) - repmat(raw.Chat, nDNoiseSets, 1) .* symmetric_normcdf(x_dec_bound, repmat(mu, nDNoiseSets, 1), repmat(measurement_sigma, nDNoiseSets, 1));
%     p_choice = bsxfun(@minus, 0.5 + 0.5 * raw.Chat, ...
%                               bsxfun(@times, raw.Chat,...
%                                              symmetric_normcdf_old(x_dec_bound,...
%                                                                repmat(mu, nDNoiseSets, 1),...
%                                                                repmat(measurement_sigma, nDNoiseSets, 1))));
    p_choice = bsxfun(@minus, 0.5 + 0.5 * raw.Chat, ...
                              bsxfun(@times, raw.Chat,...
                                             symmetric_normcdf(x_dec_bound,...
                                                               mu,...
                                                               measurement_sigma)));

elseif model.diff_mean_same_std
    % 0.5 + Chat * (0.5 - normcdf(x_dec_bound, s, measurement_sigma)
%     p_choice = 0.5 + repmat(raw.Chat, nDNoiseSets, 1) .* (0.5 - my_normcdf(x_dec_bound, repmat(mu, nDNoiseSets, 1), repmat(measurement_sigma, nDNoiseSets, 1)));
    p_choice = 0.5 + bsxfun(@times, raw.Chat,...
                                    0.5 - bsxfun_normcdf(x_dec_bound, mu, measurement_sigma));
end

p_choice = normalized_weights*p_choice;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONFIDENCE PROBABILITY %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a and b are the confidence/category lower and upper decision boundaries in measurement space around the confidence/category response on each trial.
% f computes the prob. mass that falls between a and b
if ~model.choice_only
    if ~isfield(raw, 'g')
        error('You are trying to fit confidence responses in a dataset that has no confidence trials.')
    end
    % there are redundant calculations that go into a and b but i think it's okay, and that getting rid of them wouldn't result in a huge speedup. see note above.
    if strcmp(model.family,'opt') && ~model.diff_mean_same_std
        
        if model.non_overlap
            %contrast id is more of a sig id. higher means lower contrast. so we need to reverse it.
            % this indexing stuff is a bit of a hack to make sure that term1 and term2 for each trial specify the correct
            % upper and lower bounds on the measurement, from the x_bounds (contrasts X decision boundaries) matrix made above.
            if ~model.d_noise
                x_lb = raw.Chat .* max(x_bounds((5 + raw.Chat .* raw.g) * nContrasts + 1 - raw.contrast_id),0);
                x_ub = raw.Chat .* max(x_bounds((5 + raw.Chat .* (raw.g - 1)) * nContrasts + 1 - raw.contrast_id),0);
            else
                x_lb = permute(x_bounds(1,:,:),[3 2 1]); % reshape top half of x_bounds
                x_ub = permute(x_bounds(2,:,:),[3 2 1]); % reshape bottom half
            end
        elseif model.ori_dep_noise
            [x_lb, x_ub] = x_bounds_by_trial();
            
        else
            ind = sub2ind(size(k1), ones(1,nTrials), raw.contrast_id, raw.prior_id);
            k1 = k1(ind);
            k2 = k2(ind);

            d_lb=p.b_i(2*conf_levels+1-raw.resp);
            d_ub=p.b_i(2*conf_levels+2-raw.resp);
            
            x_lb = real(sqrt(bsxfun(@rdivide,bsxfun(@minus, bsxfun(@plus, k1, d_noise_draws'), d_ub), k2)));
            x_ub = real(sqrt(bsxfun(@rdivide,bsxfun(@minus, bsxfun(@plus, k1, d_noise_draws'), d_lb), k2)));

        end
    elseif strcmp(model.family, 'opt') && model.diff_mean_same_std
        if model.ori_dep_noise
            [x_lb, x_ub] = x_bounds_by_trial();
        else
            d_lb = p.b_i(2*conf_levels+1-raw.resp) + logprior(raw.prior_id);  % should the sign of this be flipped?
            d_ub = p.b_i(2*conf_levels+2-raw.resp) + logprior(raw.prior_id); % this is the same as doing fliplr on p.b_i first
            % INTEGRATE PRIOR HERE
            x_lb = bsxfun(@rdivide, bsxfun(@times, bsxfun(@minus, d_ub, d_noise_draws'), sig.^2 + category_params.sigma_s^2), 2*category_params.mu_1);
            x_ub = bsxfun(@rdivide, bsxfun(@times, bsxfun(@minus, d_lb, d_noise_draws'), sig.^2 + category_params.sigma_s^2), 2*category_params.mu_1);
        end
        
    elseif strcmp(model.family, 'lin')
        x_lb = p.b_i(raw.resp)   + sig    .* p.m_i(raw.resp);
        x_ub = p.b_i(raw.resp+1) + sig    .* p.m_i(raw.resp+1);
        
        if ~model.diff_mean_same_std
            x_lb = max(x_lb, 0);
            x_ub = max(x_ub, 0);
        end
    elseif strcmp(model.family, 'quad')
        x_lb = p.b_i(raw.resp)   + sig.^2 .* p.m_i(raw.resp);
        x_ub = p.b_i(raw.resp+1) + sig.^2 .* p.m_i(raw.resp+1);
        
        if ~model.diff_mean_same_std
            x_lb = max(x_lb, 0);
            x_ub = max(x_ub, 0);
        end
    elseif (strcmp(model.family, 'fixed') || strcmp(model.family, 'neural1'))% && ~model.diff_mean_same_std
        x_lb = p.b_i(raw.resp); 
        x_ub = p.b_i(raw.resp+1);
        
    elseif strcmp(model.family, 'MAP')
        
        x_bounds = zeros(nContrasts, nBounds);
        
        for c = 1:nContrasts
            %cur_sig = p.unique_sigs(c);
            for r = -(conf_levels-1):(conf_levels-1)
                x_bounds(c,r+conf_levels) = lininterp1(shat_lookup_table(c,:), xVec, bf(r));
                %                     if zoomgrid
                %                         x_fine = (x_bounds(c,r)-dx : dx_fine : x_bounds(c,r)+dx)';
                %                         mu1 = x_fine*cur_sig^-2 * ksq1(c)^2;
                %                         mu2 = x_fine*cur_sig^-2 * ksq2(c)^2;
                %                         fine_lookup_table = gmm1max_n2_fast([normpdf(x_fine,0,sqrt(sig1^2 + cur_sig^2)) normpdf(x_fine,0,sqrt(sig2^2 + cur_sig^2))],...
                %                             [mu1 mu2], repmat([ksq1(c) ksq2(c)],fine_length,1));
                %                         x_bounds(c,r) = lininterp1(fine_lookup_table, x_fine, p.b_i(1+r));
                %                     end
            end
        end
        
        rows = nContrasts + 1 - raw.contrast_id;
        
        if ~model.diff_mean_same_std % task B
            x_bounds = [zeros(nContrasts,1) flipud(x_bounds) inf(nContrasts,1)];
            
            lb_cols = raw.resp;
            lb_cols(lb_cols >= 5) = lb_cols(lb_cols >= 5) + 1;
            lb_index = my_sub2ind(nContrasts, rows, lb_cols);
            a = x_bounds(lb_index);
            
            ub_cols = raw.resp;
            ub_cols(ub_cols <= 4) = ub_cols(ub_cols <= 4) + 1;
            ub_index = my_sub2ind(nContrasts, rows, ub_cols);
            b = x_bounds(ub_index);
            
            x_lb = min(cat(3,a,b), [], 3); % this is a hack. try to fix the above part
            x_ub = max(cat(3,a,b), [], 3);
            
        else % task A
            x_bounds = [-inf(nContrasts,1) flipud(x_bounds) inf(nContrasts,1)]; % -inf instead of zero, because Task A is asymmetric
            
            lb_cols = raw.resp;
            lb_index = my_sub2ind(nContrasts, rows, lb_cols);
            x_lb = x_bounds(lb_index);
            
            ub_cols = raw.resp + 1;
            ub_index = my_sub2ind(nContrasts, rows, ub_cols);
            x_ub = x_bounds(ub_index);
        end
    end
        
    if ~model.diff_mean_same_std
%         cum_prob_lb = symmetric_normcdf_old(x_lb, repmat(mu, nDNoiseSets, 1), repmat(measurement_sigma, nDNoiseSets, 1));
%         cum_prob_ub = symmetric_normcdf_old(x_ub, repmat(mu, nDNoiseSets, 1), repmat(measurement_sigma, nDNoiseSets, 1));
        cum_prob_lb = symmetric_normcdf(x_lb, mu, measurement_sigma);
        cum_prob_ub = symmetric_normcdf(x_ub, mu, measurement_sigma);
    else
        %         cum_prob_lb = my_normcdf(x_lb, repmat(mu, nDNoiseSets, 1), repmat(measurement_sigma, nDNoiseSets, 1));
        %         cum_prob_ub = my_normcdf(x_ub, repmat(mu, nDNoiseSets, 1), repmat(measurement_sigma, nDNoiseSets, 1));
        cum_prob_lb = my_normcdf(x_lb, mu, measurement_sigma);
        cum_prob_ub = my_normcdf(x_ub, mu, measurement_sigma);
    end
    
    p_conf_choice = cum_prob_ub - cum_prob_lb;
    
    p_conf_choice = normalized_weights*p_conf_choice;
    %p_conf_choice = max(0,p_conf_choice); % this max is a hack. it covers for non overlap x_bounds being weird.
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAPSES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(p, 'lambda_i')
    p_full_lapse = p.lambda_i(raw.g)/2;
    p.lambda = sum(p.lambda_i);
else
    if ~isfield(p, 'lambda')
        p.lambda=0;
    end
    if model.choice_only
        if ~isfield(p, 'lambda_bias')
            p.lambda_bias = .5;
        end
        p_full_lapse = nan(1, nTrials);
        p_full_lapse(raw.Chat == -1) = p.lambda_bias;
        p_full_lapse(raw.Chat ==  1) = 1 - p.lambda_bias;
        
        p_full_lapse = p.lambda * p_full_lapse;
        
    else
        p_full_lapse = p.lambda / (conf_levels*2);
    end
end

if ~isfield(p, 'lambda_g') % partial lapse
    p.lambda_g = 0;
end

if ~isfield(p, 'lambda_r') % repeat lapse
    p.lambda_r = 0;
    p_repeat = 0;
else
    % small problem with repeat lapse: the way I've dealt with the data puts all trials in one stream. So it could capture repeats between blocks, sections, trials. These are about 2% of trials.
    if ~model.choice_only
        p_repeat = [0 diff(raw.resp)==0];
    else
        p_repeat = [0 diff(raw.Chat)==0];
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE LOG LIKELIHOOD %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~model.choice_only
    ll_trials = log (p_full_lapse + ...
        (p.lambda_g / 4) * p_choice + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_g - p.lambda_r) * p_conf_choice); % can also do 1 - lapse_sum instead of (1 - stuff - stuff).
else % choice models
    ll_trials = log(p_full_lapse + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_r) * p_choice); % can also do 1 - lapse_sum instead of (1 - stuff - stuff).
end
% Set all -Inf logliks to a very negative number. These are usually trials where
% the subject reports something very strange. Usually lapse rate accounts for this.
ll_trials(ll_trials < -1e5) = -1e5;
nll = -sum(ll_trials);

if ~isreal(nll)
    % in case things go wrong. this shouldn't execute.
    warning('imaginary nloglik')
    %     save nltest
    nll = real(nll) + 1e3; % is this an okay way to avoid "undefined at initial point" errors? it's a hack.
end

    function bval = bf(name)
        bval = p.b_i(name + conf_levels + 1);
    end

    function bval = bf_prior(name, prior)
        bval = p.b_i(prior, name + conf_levels + 1);
    end

    function mval = mf(name)
        mval = p.m_i(name + conf_levels + 1);
    end
% 
%     function aval = af(name)
%         aval = p.a_i(name + conf_levels + 1);
%     end
% 
%     function d_boundsval = d_boundsf(name)
%         d_boundstmp = [Inf d_bounds 0];
%         d_boundsval = d_boundstmp(name + conf_levels + 1);
%     end
end

function retval = symmetric_normcdf(k, mu, sigma)
% this returns the probability mass from -k to k of N(x; mu, sig).
% symmetric_normcdf(k_big) - symmetric_normcdf(k_small) will give you the
% sum of the two probability bands, which are symmetric across x=0

% mu is the mean of the measurement distribution, usually the stimulus
% sigma is the width of the measurement distribution.
% y is the x_lb or x_ub


retval = my_normcdf(k, mu, sigma) - my_normcdf(-k, mu, sigma);
% retval(k<=0) = 0; % don't need this if we put a real() around every sqrt()?

end
 
% function retval = symmetric_normcdf_old(k, mu, sigma)
% % this returns the probability mass from -k to k of N(x; mu, sig).
% % symmetric_normcdf(k_big) - symmetric_normcdf(k_small) will give you the
% % sum of the two probability bands, which are symmetric across x=0
% 
% % mu is the mean of the measurement distribution, usually the stimulus
% % sigma is the width of the measurement distribution.
% % y is the x_lb or x_ub
% 
% 
% 
% retval              = zeros(size(mu)); % length of all trials
% idx           = find(k>0);      % find all trials where k is greater than 0. k is either positive or imaginary. so a non-positive k would indicate negative bounds, which can be dropped?
% mu                   = mu(idx);
% sigma               = sigma(idx);
% k                   = k(idx);
% 
% retval(idx)   = my_normcdf(k, mu, sigma) - my_normcdf(-k, mu, sigma);
% 
% end