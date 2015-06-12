function [nloglik, loglik_vec] = nloglik_fcn(p_in, raw, model, nDNoiseSets, varargin)

lapse_sum = lapse_rate_sum(p_in, model);

if lapse_sum > 1
    nloglik = Inf;
    loglik_vec = -inf(size(raw.Chat));
    %     warning('lapse_sum > 1')
    return
end

if length(varargin) == 1;
    category_params = varargin{1};
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
if ~model.d_noise
    nDNoiseSets = 1;
    normalized_weights = 1;
    d_noise_draws = 0;
    d_noise = 0;
    d_noise_big = 0;
elseif model.d_noise
    if ~exist('nDNoiseSets','var')
        nDNoiseSets=101;
    end
    nSTDs = 5;
    weights = normpdf(linspace(-nSTDs, nSTDs, nDNoiseSets), 0, 1);
    normalized_weights = weights ./ sum(weights);
    
    d_noise_draws = linspace(-p.sigma_d*nSTDs, p.sigma_d*nSTDs, nDNoiseSets);
    
    d_noise = repmat(d_noise_draws',1,nContrasts);
    d_noise_big = repmat(d_noise_draws',1,nTrials);% this is for non_overlap. too hard to figure out the indexing. going to be too many computations, because there's redundancy in the a,b,k vectors. but the bulk of computation has to be on an individual trial basis anyway.
end

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

xSteps = 90;
if ~model.diff_mean_same_std
    xVec = linspace(0,90,xSteps)';
else
    xVec = linspace(-45,45,xSteps)';
end


if model.ori_dep_noise
    ODN = @(s) abs(sin(s * pi / 90)) * p.sig_amplitude;
    if ~strcmp(model.family, 'opt')
        sig = p.unique_sigs(raw.contrast_id); % 1 x nTrials vector of sigmas
        sig = sig + ODN(raw.s); % add orientation dependent noise to each sigma.
    else
        sig = p.unique_sigs;
        
        sSteps = 200;
        sVec = linspace(-100,100,sSteps)';
        
        d_lookup_table = zeros(nContrasts,length(xVec));
        
        % faster than meshgrid
        s_mat = repmat(sVec,1, xSteps);
        ODN_s_mat = ODN(s_mat);
        x_mat = repmat(xVec', sSteps,1);
        x_dec_bound = zeros(nDNoiseSets,nContrasts);
        
        likelihood = @(sigma, sigma_cat, mu_cat) 1/sigma_cat * sum(1 ./ sigma .*exp(-(x_mat-s_mat).^2 ./ (2*sigma.^2) - (s_mat - mu_cat) .^2 ./ (2*sigma_cat^2)));        
        
    end
else
    sig = p.unique_sigs;
end

    function [d_lookup_table, k] = d_table_and_choice_bound(sig_cat1, sig_cat2, mu_cat1, mu_cat2)
        for c = 1:nContrasts
            cur_sig = p.unique_sigs(c);
            sig_plusODN = cur_sig + ODN_s_mat;
            d_lookup_table(c,:) = log(likelihood(sig_plusODN, sig_cat1, mu_cat1) ./ likelihood(sig_plusODN, sig_cat2, mu_cat2));
            
            %k(:, c) = lininterp1m(repmat(fliplr(d_lookup_table(c,:)),nDNoiseSets,1)+repmat(d_noise_draws',1,xSteps), fliplr(xVec'), p.b_i(5))'; % take this out of the loop?
            % would be faster to take this out of the loop for the models without d_noise. but we're not really using models w/o d noise
            k(:, c) = lininterp1_multiple(bsxfun(@plus, fliplr(d_lookup_table(c,:)), d_noise_draws'), fliplr(xVec'), bf(0)); % take this out of the loop?
        end
    end

    function [x_lb, x_ub] = x_bounds_by_trial()
        x_bounds = zeros(nContrasts, nBounds, nDNoiseSets);
        
        for c = 1:nContrasts
            for r = -(conf_levels-1):(conf_levels-1)%1:7 problem at r = 1:3
                x_bounds(nContrasts+1-c,conf_levels-r,:) = lininterp1_multiple(bsxfun(@plus, fliplr(d_lookup_table(c,:)), d_noise_draws'), fliplr(xVec'), bf(r));
            end
        end
        if ~model.diff_mean_same_std
            x_bounds = [zeros(nContrasts,1,nDNoiseSets) x_bounds inf(nContrasts,1,nDNoiseSets)];
        elseif model.diff_mean_same_std
            x_bounds = [-inf(nContrasts,1,nDNoiseSets) x_bounds inf(nContrasts,1,nDNoiseSets)];
        end
        %cols of x_bounds are for each choice and confidence
        cols_lb = 5 + raw.Chat.*(raw.g - 1);
        cols_ub = 5 + raw.Chat.* raw.g;
        %rows are for each contrast level
        rows = nContrasts + 1 - raw.contrast_id;
        
        % flatten the whole cube into a vector
        x_bounds_squash = reshape(permute(x_bounds,[3 2 1]), numel(x_bounds), 1);
        % each trial is a nDNoiseSets-vector from the x_bounds vector. This defines the first index of each vector
        start_pts_lb = (nDNoiseSets*(size(x_bounds,2)*(rows-1)+cols_lb-1)+1)';
        start_pts_ub = (nDNoiseSets*(size(x_bounds,2)*(rows-1)+cols_ub-1)+1)';
        
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
        
        x_lb = min(cat(3,a,b), [], 3); % this is a hack. try to fix the above part
        x_ub = max(cat(3,a,b), [], 3);

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHOICE PROBABILITY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% k is the category decision boundary in measurement space for each trial
% f computes the probability mass on the chosen side of k.
if strcmp(model.family, 'opt') && ~model.diff_mean_same_std% for normal bayesian family, qamar
    
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
%         sq_flag = 1; % this is because f gets passed a square root that can be negative. causes f to ignore the resulting imaginaries
        k1 = .5 * log( (sig.^2 + sig2^2) ./ (sig.^2 + sig1^2)); %log(prior / (1 - prior));
        k2 = (sig2^2 - sig1^2) ./ (2 .* (sig.^2 + sig1^2) .* (sig.^2 + sig2^2));
        x_dec_bound  = sqrt((repmat(k1, nDNoiseSets, 1) + d_noise - bf(0)) ./ repmat(k2, nDNoiseSets, 1));
        % equivalently: k= sqrt(bsxfun(@rdivide,bsxfun(@minus,bsxfun(@plus,k1,d_noise_draws'),bf(0)),k2));
    end
elseif strcmp(model.family, 'opt') && model.diff_mean_same_std
    
    if model.ori_dep_noise
        [d_lookup_table, x_dec_bound] = d_table_and_choice_bound(category_params.sigma_s, category_params.sigma_s, category_params.mu_1, category_params.mu_2);        
    else
        x_dec_bound = (2*(bf(0)+d_noise).*(repmat(sig, nDNoiseSets, 1).^2 + category_params.sigma_s^2) - category_params.mu_2^2 + category_params.mu_1^2)...
            / (2*(category_params.mu_1 - category_params.mu_2)); % ADAPT FOR D NOISE? does this d noise work?
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
    x_dec_bound = bf(0)*ones(1,nContrasts);
    
elseif strcmp(model.family, 'MAP')
    %     zoomgrid = false;
    %     if zoomgrid
    %         dx_fine = .5;
    %         fine_length = 2*dx / dx_fine + 1;
    %     end
    
    if model.ori_dep_noise
        len = nTrials;
    elseif ~model.ori_dep_noise
        len = nContrasts;
    end
    shat_lookup_table = zeros(len,xSteps);
    niter = 4;
    
    if ~model.diff_mean_same_std
        ksq1 = sqrt(1./(sig.^-2 + sig1^-2)); % like k1 in trial_generator
        ksq2 = sqrt(1./(sig.^-2 + sig2^-2)); % like k2 in trial_generator
        for i = 1:len
            cur_sig = sig(i);
            mu1 = xVec*cur_sig^-2 * ksq1(i)^2;
            mu2 = xVec*cur_sig^-2 * ksq2(i)^2;
            w1 = normpdf(xVec,0,sqrt(sig1^2 + cur_sig^2));
            w2 = normpdf(xVec,0,sqrt(sig2^2 + cur_sig^2));
            shat_lookup_table(i,:) = gmm1max_n2_fast([w1 w2], [mu1 mu2], repmat([ksq1(i) ksq2(i)],xSteps,1),niter);
        end
    elseif model.diff_mean_same_std
        ksq = sqrt(1./(sig.^-2 + category_params.sigma_s^-2));
        for i = 1:len
            cur_sig = sig(i);
            mu1 = (xVec*cur_sig^-2 + category_params.mu_1*category_params.sigma_s^-2) * ksq(i)^2; % not sure about the minus sign here.
            mu2 = (xVec*cur_sig^-2 + category_params.mu_2*category_params.sigma_s^-2) * ksq(i)^2;
            w1 = exp(xVec*category_params.mu_1./(category_params.sigma_s^2+cur_sig^2)); % not sure about minus sign here either.
            w2 = exp(xVec*category_params.mu_2./(category_params.sigma_s^2+cur_sig^2));
            shat_lookup_table(i,:) = gmm1max_n2_fast([w1 w2], [mu1 mu2], repmat([ksq(i) ksq(i)],xSteps,1),niter);
        end
    end
    %k(i) = lininterp1(shat_lookup_table(i,:), x, bf(0));
    
    %         if zoomgrid
    %             x_fine = (k(i)-dx : dx_fine : k(i)+dx)';
    %             mu1 = x_fine*cur_sig^-2 * ksq1(i)^2;
    %             mu2 = x_fine*cur_sig^-2 * ksq2(i)^2;
    %             fine_lookup_table = gmm1max_n2_fast([normpdf(x_fine,0,sqrt(sig1^2 + cur_sig^2)) normpdf(x_fine,0,sqrt(sig2^2 + cur_sig^2))],...
    %                 [mu1 mu2], repmat([ksq1(i) ksq2(i)],fine_length,1));
    %
    %             k(i) = lininterp1(fine_lookup_table, x_fine, bf(0));
    %         end
    x_dec_bound = lininterp1_multiple(shat_lookup_table, xVec, bf(0)*ones(1,len)); % find the x values corresponding to the MAP criterion, for each contrast level (or, if doing ODN where each trial has a different sigma, for every trial/sigma)
end

if numel(sig) == nContrasts
    %if ~(model.ori_dep_noise && ~strcmp(model.family, 'opt'))
    sig = p.unique_sigs(raw.contrast_id); % re-arrange sigs if it hasn't been done yet
    if model.ori_dep_noise
        sig = sig + p.sig_amplitude*abs(sin(raw.s*pi/90));
    end
end

%if ~(model.non_overlap && model.d_noise)
% do this for all models except nonoverlap+d noise, where k is already in this form.
if any(size(x_dec_bound) == nContrasts) % k needs to be expanded for each trial
    x_dec_bound = x_dec_bound(:,raw.contrast_id);
end

if ~strcmp(model.family, 'neural1')
    mu = raw.s;
    sigma = sig;
else
    g = 1./(sig.^2);
    mu = g .* raw.s .* sqrt(2*pi) * p.sigma_tc;
    sigma = sqrt(g .* (p.sigma_tc^2 + raw.s.^2) .* sqrt(2*pi) * p.sigma_tc);
end

if ~model.diff_mean_same_std
    p_choice = 0.5 + 0.5 * repmat(raw.Chat, nDNoiseSets, 1) - repmat(raw.Chat, nDNoiseSets, 1) .* symmetric_normcdf(x_dec_bound, repmat(mu, nDNoiseSets, 1), repmat(sigma, nDNoiseSets, 1));
elseif model.diff_mean_same_std
    p_choice = 0.5 + repmat(raw.Chat, nDNoiseSets, 1) .* (0.5 - my_normcdf(x_dec_bound, repmat(mu, nDNoiseSets, 1), repmat(sigma, nDNoiseSets, 1)));
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
            k1 = k1(raw.contrast_id);
            k2 = k2(raw.contrast_id);

            d_lb=p.b_i(2*conf_levels+1-raw.resp);
            d_ub=p.b_i(2*conf_levels+2-raw.resp);
            
            x_lb = sqrt(bsxfun(@rdivide,bsxfun(@minus, bsxfun(@plus, k1, d_noise_draws'), d_ub), k2));
            x_ub = sqrt(bsxfun(@rdivide,bsxfun(@minus, bsxfun(@plus, k1, d_noise_draws'), d_lb), k2));

        end
    elseif strcmp(model.family, 'opt') && model.diff_mean_same_std
        if model.ori_dep_noise
            [x_lb, x_ub] = x_bounds_by_trial();
        else
            d_lb = p.b_i(2*conf_levels+1-raw.resp);
            d_ub = p.b_i(2*conf_levels+2-raw.resp); % this is the same as doing fliplr on p.b_i first
                        
            x_lb = bsxfun(@rdivide, bsxfun(@times, bsxfun(@minus, d_ub, d_noise_draws'), sig.^2 + category_params.sigma_s^2), 2*category_params.mu_1);
            x_ub = bsxfun(@rdivide, bsxfun(@times, bsxfun(@minus, d_lb, d_noise_draws'), sig.^2 + category_params.sigma_s^2), 2*category_params.mu_1);
        end
        
    elseif strcmp(model.family, 'lin')
        x_lb = p.b_i(raw.resp)   + sig    .* p.m_i(raw.resp);
        x_ub = p.b_i(raw.resp+1) + sig    .* p.m_i(raw.resp+1);
        
    elseif strcmp(model.family, 'quad')
        x_lb = p.b_i(raw.resp)   + sig.^2 .* p.m_i(raw.resp);
        x_ub = p.b_i(raw.resp+1) + sig.^2 .* p.m_i(raw.resp+1);
        
    elseif (strcmp(model.family, 'fixed') || strcmp(model.family, 'neural1'))% && ~model.diff_mean_same_std
        x_lb = p.b_i(raw.resp); 
        x_ub = p.b_i(raw.resp+1);
        
    elseif strcmp(model.family, 'MAP')
        
        if ~model.ori_dep_noise
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
            if ~model.diff_mean_same_std
                x_bounds = [zeros(nContrasts,1) flipud(x_bounds) inf(nContrasts,1)];
                x_lb = raw.Chat .* x_bounds((5 + raw.Chat .* raw.g) * nContrasts + 1 - raw.contrast_id);
                x_ub = raw.Chat .* x_bounds((5 + raw.Chat .* (raw.g - 1)) * nContrasts + 1 - raw.contrast_id);
                %a = raw.Chat .* x_bounds(sub2ind([nContrasts conf_levels*2+1], nContrasts + 1 - raw.contrast_id, 5 + raw.Chat .* raw.g)); % equivalent, but 3x slower
                %b = raw.Chat .* x_bounds(sub2ind([nContrasts conf_levels*2+1], nContrasts + 1 - raw.contrast_id, 5 + raw.Chat .* (raw.g - 1)));
            else
                x_bounds = [-inf(nContrasts,1) flipud(x_bounds) inf(nContrasts,1)]; % -inf instead of zero, because Task A is asymmetric
                %a = x_bounds((5 + raw.Chat .* (raw.g - 1)) * nContrasts + 1 - raw.contrast_id);
                %                 b = x_bounds((5 + raw.Chat .* raw.g) * nContrasts + 1 - raw.contrast_id);
                x_lb = x_bounds((5 + raw.Chat .* raw.g - .5*(raw.Chat+1)) * nContrasts + 1 - raw.contrast_id);
                x_ub = x_bounds((5 + raw.Chat .* raw.g - .5*(raw.Chat-1)) * nContrasts + 1 - raw.contrast_id);
            end
            
            % The following is cleaner than the above approach, and it matches more with the below ori_dep_noise version, but it's 2x slower:
            %             a = zeros(1,nTrials);
            %             b = zeros(1,nTrials);
            %             for i = 1:len
            %                 a(raw.contrast_id==i) = lininterp1_multiple(shat_lookup_table(i,:), x, bf(raw.Chat(raw.contrast_id==i).*raw.g(raw.contrast_id==i)));
            %                 b(raw.contrast_id==i) = lininterp1_multiple(shat_lookup_table(i,:), x, bf(raw.Chat(raw.contrast_id==i).*(raw.g(raw.contrast_id==i)-1)));
            %             end
            %             a(raw.Chat==1 & raw.g ==4) = Inf;
            %             a = raw.Chat .* a;
            %             b = raw.Chat .* b;
            
        elseif model.ori_dep_noise
            x_lb = raw.Chat .* lininterp1_multiple(shat_lookup_table, xVec, bf(raw.Chat .* raw.g));
            x_ub = raw.Chat .* lininterp1_multiple(shat_lookup_table, xVec, bf(raw.Chat .*(raw.g - 1)));
            x_lb(raw.Chat==1 & raw.g==4) = Inf;
        end
        
    end
    
    if ~strcmp(model.family, 'neural1')
        mu = raw.s;
        sigma = sig;
    else
        mu = sig.^-2 .* raw.s .* sqrt(2*pi) * p.sigma_tc;
        sigma = sqrt(sig.^-2 .* (p.sigma_tc^2 + raw.s.^2) .* sqrt(2*pi) * p.sigma_tc);
    end
    
    if ~model.diff_mean_same_std
        cum_prob_lb = symmetric_normcdf(x_lb, repmat(mu, nDNoiseSets, 1), repmat(sigma, nDNoiseSets, 1));
        cum_prob_ub = symmetric_normcdf(x_ub, repmat(mu, nDNoiseSets, 1), repmat(sigma, nDNoiseSets, 1));
    else
        cum_prob_lb = my_normcdf(x_lb, repmat(mu, nDNoiseSets, 1), repmat(sigma, nDNoiseSets, 1));
        cum_prob_ub = my_normcdf(x_ub, repmat(mu, nDNoiseSets, 1), repmat(sigma, nDNoiseSets, 1));
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
    if ~isfield(p, 'lambda') % this is only for a few d_noise models that are probably deprecated
        p.lambda=0;
    end
    p_full_lapse = p.lambda/(conf_levels*2); % only goes into conf models below
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
    loglik_vec = log (p_full_lapse + ...
        (p.lambda_g / 4) * p_choice + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_g - p.lambda_r) * p_conf_choice); % can also do 1 - lapse_sum instead of (1 - stuff - stuff).
else % choice models
    loglik_vec = log(p.lambda / 2 + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_r) * p_choice); % can also do 1 - lapse_sum instead of (1 - stuff - stuff).
end
% Set all -Inf logliks to a very negative number. These are usually trials where
% the subject reports something very strange. Usually lapse rate accounts for this.
loglik_vec(loglik_vec < -1e5) = -1e5;
nloglik = - sum(loglik_vec);

if ~isreal(nloglik)
    % in case things go wrong. this shouldn't execute.
    warning('imaginary nloglik')
    %     save nltest
    nloglik = real(nloglik) + 1e3; % is this an okay way to avoid "undefined at initial point" errors? it's a hack.
end

    function bval = bf(name)
        bval = p.b_i(name + conf_levels + 1);
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

function y = my_normcdf(k, mu, sigma)
y = 0.5 * (1 + erf((k-mu)./(sigma*sqrt(2))));
end

function retval = symmetric_normcdf(k, mu, sigma)
% this returns the probability mass from -k to k of N(x; mu, sig).
% symmetric_normcdf(k_big) - symmetric_normcdf(k_small) will give you the
% sum of the two probability bands, which are symmetric across x=0

% mu is the mean of the measurement distribution, usually the stimulus
% sigma is the width of the measurement distribution.
% y is the x_lb or x_ub

retval              = zeros(size(mu)); % length of all trials
idx           = find(k>0);      % find all trials where k is greater than 0. k is either positive or imaginary. so a non-positive k would indicate negative bounds, which can be dropped?
mu                   = mu(idx);
sigma               = sigma(idx);
k                   = k(idx);

retval(idx)   = my_normcdf(k, mu, sigma) - my_normcdf(-k, mu, sigma);
end