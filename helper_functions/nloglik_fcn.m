function [nloglik, loglik_vec] = nloglik_fcn(p_in, raw, model, nDNoiseSets, varargin)
% try
if length(varargin) == 1;
    category_params = varargin{1};
end

% % this is now going to break non-fmincon optimization, because i took out the constraints
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

p = parameter_variable_namer(p_in, model.parameter_names, model);
contrasts = exp(linspace(-5.5,-2,6)); % THIS IS HARD CODED
%contrasts = exp(-4:.5:-1.5);
nContrasts = length(contrasts);
nTrials = length(raw.s);
if ~model.d_noise
    nDNoiseSets = 1;
    normalized_weights = 1;
    d_noise_draws = 0;
    d_noise = 0;
    d_noise_big = 0;
elseif model.d_noise
    if ~exist('nDNoiseSets')
        nDNoiseSets=101;
    end
    nSTDs = 5;
    weights = normpdf(linspace(-nSTDs, nSTDs, nDNoiseSets), 0, 1);
    normalized_weights = weights ./ sum(weights);
    
    d_noise_draws = linspace(-p.sigma_d*nSTDs, p.sigma_d*nSTDs, nDNoiseSets);
    
    d_noise = repmat(d_noise_draws',1,nContrasts);
    d_noise_big = repmat(d_noise_draws',1,nTrials);% this is for confidence. too hard to figure out the indexing. going to be too many computations, because there's redundancy in the a,b,k vectors. but the bulk of computation has to be on an individual trial basis anyway.
end

if isfield(p,'b_i')
    conf_levels = (length(p.b_i) - 1)/2;
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


contrast_type = 'new';
switch contrast_type
    case 'old'
        unique_sigs = fliplr(sqrt(max(0,p.sigma_0^2 + p.alpha .* contrasts .^ -p.beta))); % low to high sigma. should line up with contrast id
    case 'new'
        c_low = min(contrasts);
        c_hi = max(contrasts);
        alpha = (p.sigma_c_low^2-p.sigma_c_hi^2)/(c_low^-p.beta - c_hi^-p.beta);
        unique_sigs = fliplr(sqrt(p.sigma_c_low^2 - alpha * c_low^-p.beta + alpha*contrasts.^-p.beta)); % low to high sigma. should line up with contrast id
end

% now k will only be 6 cols, rather than 3240.
sq_flag = 0;

if model.ori_dep_noise && ~strcmp(model.family, 'opt')
    sig = unique_sigs(raw.contrast_id);
    sig = sig + p.sig_amplitude*abs(sin(raw.s*pi/90));
else
    sig = unique_sigs;
end

xSteps = 90;
xVec = linspace(0,90,xSteps)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHOICE PROBABILITY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% k is the category decision boundary in measurement space for each trial
% f computes the probability mass on the chosen side of k.
if strcmp(model.family, 'opt') && ~model.diff_mean_same_std% for normal bayesian family, qamar
    
    if model.non_overlap
        x_bounds = find_intersect_truncated_cats(p, sig1, sig2, contrasts, d_noise_big, raw);
        
        if ~model.d_noise
            k = fliplr(x_bounds(:,4)');
            x_bounds = [zeros(6,1) x_bounds inf(6,1)]; % this is for confidence
        elseif model.d_noise
            % for d noise, need long noisesets x trials matrix
            k = permute(x_bounds(3,:,:),[3 2 1]);
        end
        
    elseif model.ori_dep_noise
        sSteps = 200;
        sVec = linspace(-100,100,sSteps)';
        
        d_lookup_table = zeros(nContrasts,length(xVec));
        % faster than meshgrid
        s_mat = repmat(sVec,1, xSteps);
        x_mat = repmat(xVec', sSteps,1);
        k = zeros(nDNoiseSets,nContrasts);
        for c = 1:nContrasts
            cur_sig = unique_sigs(c);
            
            term1=cur_sig+abs(sin(s_mat*pi/90))*p.sig_amplitude;
            term2=-((x_mat-s_mat).^2)./(2*(term1).^2);
            term3= 0.5*s_mat.^2;
            d_lookup_table(c,:) = log(sig2/sig1) +...
                log(sum(1./term1.*exp(term2 - term3 ./ (sig1^2))) ./ ...
                (sum(1./term1.*exp(term2 - term3 ./ (sig2^2)))));
            
            %k(:, c) = lininterp1m(repmat(fliplr(d_lookup_table(c,:)),nDNoiseSets,1)+repmat(d_noise_draws',1,xSteps), fliplr(xVec'), p.b_i(5))'; % take this out of the loop?
            % would be faster to take this out of the loop for the models without d_noise. but we're not really using models w/o d noise
            k(:, c) = lininterp1_multiple(bsxfun(@plus, fliplr(d_lookup_table(c,:)), d_noise_draws'), fliplr(xVec'), bf(0)); % take this out of the loop?
        end
        
    else
        sq_flag = 1; % this is because f gets passed a square root that can be negative. causes it to ignore the negatives
        k1 = .5 * log( (sig.^2 + sig2^2) ./ (sig.^2 + sig1^2)); %log(prior / (1 - prior));
        k2 = (sig2^2 - sig1^2) ./ (2 .* (sig.^2 + sig1^2) .* (sig.^2 + sig2^2));
        k  = sqrt((repmat(k1, nDNoiseSets, 1) + d_noise - bf(0)) ./ repmat(k2, nDNoiseSets, 1));
        % equivalently: k= sqrt(bsxfun(@rdivide,bsxfun(@minus,bsxfun(@plus,k1,d_noise_draws'),bf(0)),k2));
    end
elseif strcmp(model.family, 'opt') && model.diff_mean_same_std
    k = (2*(bf(0)+d_noise).*(repmat(sig, nDNoiseSets, 1).^2 + category_params.sigma_s^2) - category_params.mu_2^2 + category_params.mu_1^2)...
        / (2*(category_params.mu_1 - category_params.mu_2)); % ADAPT FOR D NOISE? does this d noise work?
    
elseif strcmp(model.family, 'lin') && ~model.diff_mean_same_std
    k = max(bf(0) + mf(0) * sig, 0);
elseif strcmp(model.family, 'lin') && model.diff_mean_same_std
    k = bf(0) + mf(0) * sig;
    
elseif strcmp(model.family, 'quad') && ~model.diff_mean_same_std
    k = max(bf(0) + mf(0) * sig.^2, 0);
elseif strcmp(model.family, 'quad') && model.diff_mean_same_std
    k = bf(0) + mf(0) * sig.^2;
    
elseif strcmp(model.family, 'fixed')
    k = bf(0)*ones(1,nContrasts);
    
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
        ksq1 = sqrt(1./(sig.^-2 + sig1^-2));
        ksq2 = sqrt(1./(sig.^-2 + sig2^-2));
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
    k = lininterp1_multiple(shat_lookup_table, xVec, bf(0)*ones(1,len));
end

if numel(sig) == nContrasts
    %if ~(model.ori_dep_noise && ~strcmp(model.family, 'opt'))
    sig = unique_sigs(raw.contrast_id); % re-arrange sigs if it hasn't been done yet
    if model.ori_dep_noise
        sig = sig + p.sig_amplitude*abs(sin(raw.s*pi/90));
    end
end

%if ~(model.non_overlap && model.d_noise)
% do this for all models except nonoverlap+d noise, where k is already in this form.
if any(size(k) == nContrasts) % k needs to be expanded for each trial
    k = k(:,raw.contrast_id);
end

if ~model.diff_mean_same_std
    p_choice = 0.5 + 0.5 * repmat(raw.Chat, nDNoiseSets, 1) -repmat(raw.Chat, nDNoiseSets, 1) .* f(k, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1), sq_flag);
elseif model.diff_mean_same_std
    p_choice = 0.5 - repmat(-raw.Chat,nDNoiseSets, 1) .* sym_f(-k, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1)); % 2/14 put - in front of chat
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
                a = raw.Chat .* max(x_bounds((5 + raw.Chat .* raw.g) * nContrasts + 1 - raw.contrast_id),0);
                b = raw.Chat .* max(x_bounds((5 + raw.Chat .* (raw.g - 1)) * nContrasts + 1 - raw.contrast_id),0);
            else
                a = permute(x_bounds(1,:,:),[3 2 1]); % reshape top half of x_bounds
                b = permute(x_bounds(2,:,:),[3 2 1]); % reshape bottom half
            end
        elseif model.ori_dep_noise
            % this is kinda weird, but it works.
            % this should be the model for what to do in bayesian models with non-analytical decision variables.
            % 6 x 9 x 101 or 1 cube of x cat/conf bounds
            x_bounds = zeros(nContrasts, conf_levels*2-1, nDNoiseSets);
            
            for c = 1:nContrasts
                for r = -3:3%1:7
                    %                     x_bounds(nContrasts+1-c,conf_levels*2-r,:) = lininterp1m(repmat(fliplr(d_lookup_table(c,:)), nDNoiseSets, 1) + repmat(d_noise_draws',1,xSteps), -flipud(xVec), p.b_i(1+r))';
                    x_bounds(nContrasts+1-c,conf_levels-r,:) = lininterp1_multiple(bsxfun(@plus, fliplr(d_lookup_table(c,:)), d_noise_draws'), fliplr(xVec'), bf(r));
                end
            end
            %                         x_bounds = [zeros(6,1,nDNoiseSets) -x_bounds inf(6,1,nDNoiseSets)];
            x_bounds = [zeros(6,1,nDNoiseSets) x_bounds inf(6,1,nDNoiseSets)];
            
            %cols of x_bounds are for each choice and confidence
            cols_a = 5 + raw.Chat.* raw.g;
            cols_b = 5 + raw.Chat.*(raw.g - 1);
            %rows are for each contrast level
            rows = nContrasts + 1 - raw.contrast_id;
            
            % flatten the whole cube into a vector
            x_bounds_squash = reshape(permute(x_bounds,[3 2 1]), numel(x_bounds), 1);
            % each trial is a nDNoiseSets-vector from the x_bounds vector. This defines the first index of each vector
            start_pts_a = (nDNoiseSets*(size(x_bounds,2)*(rows-1)+cols_a-1)+1)';
            start_pts_b = (nDNoiseSets*(size(x_bounds,2)*(rows-1)+cols_b-1)+1)';
            
            % this takes those vectors, and turns them back into a matrix.
            a = bsxfun(@times, raw.Chat', x_bounds_squash(bsxfun(@plus, start_pts_a, (0:nDNoiseSets-1))))';
            b = bsxfun(@times, raw.Chat', x_bounds_squash(bsxfun(@plus, start_pts_b, (0:nDNoiseSets-1))))';
            
        else
            k1 = k1(raw.contrast_id);
            k2 = k2(raw.contrast_id);
            bound1=bf((raw.Chat - 1)./2 - raw.Chat .* raw.g);
            bound2=bf((raw.Chat + 1)./2 - raw.Chat .* raw.g);
            
            %                 a = sqrt(repmat(k1 - bf((raw.Chat - 1)./2 - raw.Chat .* raw.g), nDNoiseSets, 1) + d_noise_big) ./ repmat(sqrt(k2), nDNoiseSets, 1);
            a = sqrt(bsxfun(@rdivide,bsxfun(@minus, bsxfun(@plus, k1, d_noise_draws'), bound1), k2));
            %                 b = sqrt(repmat(k1 - bf((raw.Chat + 1)./2 - raw.Chat .* raw.g), nDNoiseSets, 1) + d_noise_big) ./ repmat(sqrt(k2), nDNoiseSets, 1);
            b = sqrt(bsxfun(@rdivide,bsxfun(@minus, bsxfun(@plus, k1, d_noise_draws'), bound2), k2));
            
        end
    elseif strcmp(model.family, 'opt') && model.diff_mean_same_std
        % these might be right, and they are more flexible. but i'm gonna do the simpler version where the categories are symmetric.
        %              a = (2*(repmat(bf(0.5*(raw.Chat - 1) - raw.Chat .* raw.g), nDNoiseSets, 1)+d_noise_big).*(repmat(sig, nDNoiseSets, 1).^2 + category_params.sigma_s^2) - category_params.mu_2^2 + category_params.mu_1^2) ...
        %                  ./ (2*(category_params.mu_1 - category_params.mu_2));
        %             b = (2*(repmat(bf(0.5*(raw.Chat + 1) - raw.Chat .* raw.g), nDNoiseSets, 1)+d_noise_big).*(repmat(sig, nDNoiseSets, 1).^2 + category_params.sigma_s^2) - category_params.mu_2^2 + category_params.mu_1^2) ...
        %                 ./ (2*(category_params.mu_1 - category_params.mu_2));
        
        
        bound1 = bf(-raw.Chat .* raw.g + .5*(raw.Chat+1));
        bound2 = bf(-raw.Chat .* raw.g + .5*(raw.Chat-1));
        
        a = bsxfun(@rdivide, bsxfun(@times, bsxfun(@minus, bound1, d_noise_draws'), sig.^2 + category_params.sigma_s^2), 2*category_params.mu_1);
        b = bsxfun(@rdivide, bsxfun(@times, bsxfun(@minus, bound2, d_noise_draws'), sig.^2 + category_params.sigma_s^2), 2*category_params.mu_1);
        
    elseif strcmp(model.family, 'lin') && ~model.diff_mean_same_std
        a = raw.Chat .* max(bf(raw.Chat .* (raw.g    )) + sig .* mf(raw.Chat .* (raw.g    )), 0); %Chat multiplier here is to make sure that a is less than b, i think
        b = raw.Chat .* max(bf(raw.Chat .* (raw.g - 1)) + sig .* mf(raw.Chat .* (raw.g - 1)), 0);
    elseif strcmp(model.family, 'lin') && model.diff_mean_same_std
        %             a = raw.Chat .* bf(raw.Chat .* (raw.g    )) + sig .* mf(raw.Chat .* (raw.g    ));
        %             b = raw.Chat .* bf(raw.Chat .* (raw.g - 1)) + sig .* mf(raw.Chat .* (raw.g - 1));
        a = bf(raw.Chat .* raw.g - .5*(raw.Chat+1)) + sig .* mf(raw.Chat .* raw.g - .5*(raw.Chat+1));
        b = bf(raw.Chat .* raw.g - .5*(raw.Chat-1)) + sig .* mf(raw.Chat .* raw.g - .5*(raw.Chat-1));
        
    elseif strcmp(model.family, 'quad') && ~model.diff_mean_same_std
        a = raw.Chat .* max(bf(raw.Chat .* (raw.g    )) + sig.^2 .* mf(raw.Chat .* (raw.g    )), 0);
        b = raw.Chat .* max(bf(raw.Chat .* (raw.g - 1)) + sig.^2 .* mf(raw.Chat .* (raw.g - 1)), 0);
    elseif strcmp(model.family, 'quad') && model.diff_mean_same_std
        %             a = raw.Chat .* bf(raw.Chat .* (raw.g    )) + sig.^2 .* mf(raw.Chat .* (raw.g    ));
        %             b = raw.Chat .* bf(raw.Chat .* (raw.g - 1)) + sig.^2 .* mf(raw.Chat .* (raw.g - 1));
        a = bf(raw.Chat .* raw.g - .5*(raw.Chat+1)) + sig.^2 .* mf(raw.Chat .* raw.g - .5*(raw.Chat+1));
        b = bf(raw.Chat .* raw.g - .5*(raw.Chat-1)) + sig.^2 .* mf(raw.Chat .* raw.g - .5*(raw.Chat-1));
        
    elseif strcmp(model.family, 'fixed') && ~model.diff_mean_same_std
        a = raw.Chat .* bf(raw.Chat .* (raw.g)    );
        b = raw.Chat .* bf(raw.Chat .* (raw.g - 1));
    elseif strcmp(model.family, 'fixed') && model.diff_mean_same_std
        a = bf(raw.Chat .* raw.g - .5*(raw.Chat+1));
        b = bf(raw.Chat .* raw.g - .5*(raw.Chat-1));
        
    elseif strcmp(model.family, 'MAP')
        
        if ~model.ori_dep_noise
            x_bounds = zeros(nContrasts, conf_levels*2-1);
            
            for c = 1:nContrasts
                %cur_sig = unique_sigs(c);
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
            x_bounds = [zeros(6,1) flipud(x_bounds) inf(6,1)];
            a = raw.Chat .* x_bounds((5 + raw.Chat .* raw.g) * nContrasts + 1 - raw.contrast_id);
            b = raw.Chat .* x_bounds((5 + raw.Chat .* (raw.g - 1)) * nContrasts + 1 - raw.contrast_id);
            
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
            a = raw.Chat .* lininterp1_multiple(shat_lookup_table, xVec, bf(raw.Chat .* raw.g));
            b = raw.Chat .* lininterp1_multiple(shat_lookup_table, xVec, bf(raw.Chat .*(raw.g - 1)));
            a(raw.Chat==1 & raw.g==4) = Inf;
        end
        
    end
    
    if ~model.diff_mean_same_std
        % if model.d_noise
        fa = f(a, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1), sq_flag);
        fb = f(b, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1), sq_flag);
        p_conf_choice = fa - fb;
        
        % else
        %     p_conf_choice = f(a,raw.s,sig,sq_flag) - f(b,raw.s,sig,sq_flag);
        %%if sum(p_conf_choice<0)~=0
        %%fprintf('%g trials where f(b)>f(a)\n',sum(p_conf_choice<0))
        %%end
        %end
    elseif model.diff_mean_same_std
        %             fb = sym_f(b, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1));
        %             fa = sym_f(a, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1));
        %             p_conf_choice = fa - fb; % equivalent to 0.5 * (erf((a + raw.s)./(sqrt(2)*sig)) - erf((b + raw.s)./(sqrt(2)*sig))); % just switched fa and fb
        %p_conf_choice = .5 * (erf((b-raw.s)./(sqrt(2)*sig)) + erf((a - raw.s)./(sqrt(2)*sig)));
        %                           save nltest
        %
        fa = sym_f(a, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1));
        fb = sym_f(b, repmat(raw.s, nDNoiseSets, 1), repmat(sig, nDNoiseSets, 1));
        p_conf_choice = fa - fb;
        %                           p_conf_choice = .5 * (erf((raw.s-a)./(sqrt(2)*sig))-erf((raw.s-b)./(sqrt(2)*sig)));
    end
    
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

if ~isfield(p, 'lambda_g')
    p.lambda_g = 0;
end

if ~model.choice_only
    p_repeat = [0 diff(raw.resp)==0];
else
    p_repeat = [0 diff(raw.Chat)==0];
end

if ~isfield(p, 'lambda_r')
    p.lambda_r = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE LOG LIKELIHOOD %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~model.choice_only
    loglik_vec = log (p_full_lapse + ...
        (p.lambda_g / 4) * p_choice + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_g - p.lambda_r) * p_conf_choice);
    
else % choice models
    loglik_vec = log(p.lambda / 2 + ...
        p.lambda_r * p_repeat + ...
        (1 - p.lambda - p.lambda_r) * p_choice);
    
end
% Set all -Inf logliks to an arbitrarily small number. It looks like these
% trials are all ones in which abs(s) was very large, and the subject
% didn't respond with full confidence. This occurs about .3% of trials.
% Shouldn't happen with lapse rate
loglik_vec(loglik_vec < -1e5) = -1e5;
nloglik = - sum(loglik_vec);

if ~isreal(nloglik)
    % this is a big problem for truncated cats.
    warning('imaginary nloglik')
    nloglik = real(nloglik) + 1e3; % is this an okay way to avoid "undefined at initial point" errors? it's a hack.
end

    function bval = bf(name)
        bval = p.b_i(name + conf_levels + 1);
    end

    function mval = mf(name)
        mval = p.m_i(name + conf_levels + 1);
    end

    function aval = af(name)
        aval = p.a_i(name + conf_levels + 1);
    end

    function d_boundsval = d_boundsf(name)
        d_boundstmp = [Inf d_bounds 0];
        d_boundsval = d_boundstmp(name + conf_levels + 1);
    end


end

function retval = f(y,s,sigma,sq_flag)
retval              = zeros(size(s)); % length of all trials
if sq_flag
    idx           = find(y>0);      % find all trials where y is greater than 0. y is either positive or imaginary. so a non-positive y would indicate negative a or b
    s                   = s(idx);
    sigma               = sigma(idx);
    y                   = y(idx);
else
    idx = true(size(s));
end
retval(idx)   = 0.5 * (erf((s+y)./(sigma*sqrt(2))) - erf((s-y)./(sigma*sqrt(2)))); % erf is faster than normcdf.
end



function retval = sym_f(y,s,sigma)
retval = 0.5 * erf((s-y)./(sigma*sqrt(2)));
end

