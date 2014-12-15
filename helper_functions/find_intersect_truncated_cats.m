function x_bounds = find_intersect_truncated_cats(p, sig1, sig2, contrasts, d_noise, varargin)
% could simplify this input scheme. d_noise always come with raw

% this takes about .02-.04 seconds with no d_noise. with 500 samples of d_noise, it takes about 26 seconds, or about 700 times slower.

tol = 1e-2; % for newton iteration, d bound target
maxiter=6;
sigs = sqrt(max(0, p.sigma_0^2 + p.alpha .* contrasts .^ -p.beta));
nDNoiseSets = size(d_noise,1);

d_bounds = fliplr(p.b_i(2:end-1));

% choice_boundary = p.b_i(5);
% d_choice = d_noise + choice_boundary; % this could be one number, or a nDNoiseSets x nTrials matrix.
% 
% x_choice = .8*ones(size(d_chocie)); % PUSH THIS TOWARDS D_CHOICE WITH NEWTON
% 

if nDNoiseSets > 1
    % if using d_noise, make sure that raw data is input, and that their sizes are compatible
    if isempty(varargin)
        error('need to include raw data if using d_noise')
    end
    raw=varargin{1};
    nTrials = length(raw.g);
    if nTrials ~= size(d_noise,2);
        error('length of d_noise is not equal to the number of trials')
    end

    % for k, choice boundary
    
    
    
    % for a and b, confidence bounds
    d_boundstmp = [Inf d_bounds -Inf];
    d_id(1,:) = raw.Chat.*(raw.g-0.5)+5.5;
    d_id(2,:) = raw.Chat.*(raw.g-0.5)+4.5;
    %d_id2 = repmat(d_id,1,1,nDNoiseSets);
    d_bounds2 = d_boundstmp(d_id);
    d_bounds2 = cat(1,d_bounds2,repmat(p.b_i(5),1,nTrials));
    
    
    % this makes a 3 x nTrials x nDNoiseSets cube of noisy d values, where the rows are a, b, k. find x equivalent for each.
    d_bounds = repmat(d_bounds2,1,1,nDNoiseSets) + repmat(permute(d_noise,[3 2 1]),3,1);

    x_bounds = zeros(size(d_bounds));

else
    x_bounds = zeros(length(contrasts), length(d_bounds));
end

x=6*ones(size(d_bounds)); %starting x value

% this exclusion is not contrast-dependent, so it can be outside the loop    
high = d_bounds==-Inf; % find trials where the d_bounds are -Inf. Here, the equivalent measurement will be positive infinity for all contrasts.
x(high) = Inf;

for level = 1:length(contrasts)
    % for speed, these are inline versions of the external functions trun_sigstruct and trun_da used in trial_generator. 
    % this is redundant and annoying

    f=trun_sigstruct(sigs(level), sig1, sig2);
    
    % d(x)
    f.d = @(x) f.term1 ...
        + (.5*x.^2)*(1/f.sigsq2-1/f.sigsq1) ...
        + log(erf(f.tk_1sqrt2 + x*f.k1s2) + erf(f.tk_1sqrt2 - x*f.k1s2)) ...
        - log(2 - erf(f.tk_2sqrt2 + x*f.k2s2) - erf(f.tk_2sqrt2 - x*f.k2s2));
    
    % approximation for large pos/neg x
    f.da = @(x) f.term1 ...
        + f.term2 ...
        + (.5*x.^2)*(1/f.sigsq2-1/f.sigsq1) ...
        - x.^2 * f.k1sq / (2*f.sig^4) ...
        + sign(x)*f.t.*x/f.sigsq ... % sign change here
        -log(-f.tk_1sqrt2 +sign(x).*x*f.k1s2); % and here
    
    % approximation for x->0. this is a bit off.
    f.da0 = @(x) f.term1 ...
        + (.5*x.^2)*(1/f.sigsq2-1/f.sigsq1) ...
        + log(erf(f.tk_1sqrt2 + x*f.k1s2) + erf(f.tk_1sqrt2 - x*f.k1s2)) ...
        + f.term3 ...
        + x.^2 * f.k2sq ./ (2*f.sig^4) ...
        - log(exp(-f.t*x/f.sigsq)./(f.t*f.sigsq - x.*f.k2sq)+exp(f.t.*x./f.sigsq)./(f.t*f.sigsq + x.*f.k2sq));
    
    % d/dx d(x)
    f.dd = @(x) f.term4 * ( ...
        f.k_1*(exp(-(f.tk_1sqrt2 + x*f.k1s2).^2) - exp(-(f.tk_1sqrt2 - x*f.k1s2).^2)) ./ (erf(f.tk_1sqrt2 + x*f.k1s2) + erf(f.tk_1sqrt2 - x*f.k1s2)) ...
        + f.k_2*(exp(-(f.tk_2sqrt2 + x*f.k2s2).^2) - exp(-(f.tk_2sqrt2 - x*f.k2s2).^2)) ./ (2 - erf(f.tk_2sqrt2 + x*f.k2s2) - erf(f.tk_2sqrt2 - x*f.k2s2))) ...
        + x*f.term5;
    
    % approximation for large pos/neg x
    f.dda= @(x) x*f.term5 ...
        - x*f.k1sq/f.sig^4 ...
        + sign(x) * f.t/f.sigsq ...
        - f.k1sq ./ (-f.t*f.sigsq + x*f.k1sq);
    
    % approximation for x->0. this is a bit off.
    f.dda0 = @(x) f.term4 * ( ...
        f.k_1*(exp(-(f.tk_1sqrt2 + x*f.k1s2).^2) - exp(-(f.tk_1sqrt2 - x*f.k1s2).^2)) ./ (erf(f.tk_1sqrt2 + x*f.k1s2) + erf(f.tk_1sqrt2 - x*f.k1s2)) ...
        + sqrt(pi/2)*(exp(-(f.tk_2sqrt2 + x*f.k2s2).^2) - exp(-(f.tk_2sqrt2 - x*f.k2s2).^2)) ...
        ./ (f.sigsq.*exp(-(f.t^2./(2*f.k2sq)) - x.^2 .* f.k2sq ./(2.*f.sig^4)) .* (exp(-f.t*x./f.sigsq)./(f.t.*f.sigsq + x.*f.k2sq) + exp(f.t*x./f.sigsq)./(f.t.*f.sigsq - x.*f.k2sq)))) ...
        + x*f.term5;
    o=-40:.01:40;
    
    %test = trun_inline(o,f);
    % find d(0), the maximum of d(x) at the current loop's contrast. any d_bounds greater than that will be set to a x_bound very close to zero.
    low = trun_inline(0,f)<d_bounds;
    
    if nDNoiseSets > 1 % if using d noise
        % index trials from contrasts that aren't in the current loop. don't run these trials in the newton iteration
        other_contrasts = repmat(raw.contrast_id ~= length(contrasts)+1-level, 3, 1, nDNoiseSets);
        
        % set the x_bounds that are low and of the current contrast to basically zero.
        x(low & ~other_contrasts) = 0;%d_id2(low & ~other_contrasts)*1e-5;
        on = ~high & ~low & ~other_contrasts;     % only fit the x_bounds that aren't Inf, 0, or relevant but some other contrast.

    else
        x(low) = 0;%linspace(1e-5,1e-5*sum(low), sum(low));
        on = ~high & ~low;
        
    end

    count=0;
    y = inf(size(d_bounds)); % set initial d val to infinity to get the loop started
    dy = inf(size(d_bounds));
    while any(abs(y(on) - d_bounds(on)) > tol) && count < maxiter % newtonian iteration
        count = count+1;
        [y(on),dy(on)]=trun_inline(x(on),f);
        dy(dy==0)=.001; % this is a hack.
        x(on) = abs(x(on) - (y(on)-d_bounds(on))./dy(on));
        on = on & (abs(y - d_bounds) > tol);
    end
    if nDNoiseSets == 1
        x_bounds(level,:) = x;
        x_bounds(diff(x_bounds,1,2)<0)=0; % this is a hack.
    end

end
    
if nDNoiseSets > 1
    x_bounds=x;
end
%save fitc_nan
end

function f = trun_sigstruct(sig,sig1,sig2)
f.sig = sig;
f.t=sqrt(log(sig2/sig1)/(.5*(1/sig1^2 - 1/sig2^2)));
f.sigsq = sig^2;
f.sigsqrt2=f.sigsq*sqrt(2);
f.sigsq1 = f.sigsq + sig1^2;
f.sigsq2 = f.sigsq + sig2^2;
f.sig1sqrt2 = sig1*sqrt(2);
f.sig2sqrt2 = sig2*sqrt(2);
f.k_1 = sig.*sig1./sqrt(f.sigsq1);
f.k_2 = sig.*sig2./sqrt(f.sigsq2);
f.k1sq = f.k_1.^2;
f.k2sq = f.k_2.^2;
f.tk_1sqrt2=f.t/(f.k_1*sqrt(2));
f.tk_2sqrt2=f.t/(f.k_2*sqrt(2));
f.k1s2 = f.k_1./f.sigsqrt2;
f.k2s2 = f.k_2./f.sigsqrt2;
f.term1= log(sqrt(f.sigsq2/f.sigsq1)) ...
    + log(1-erf(f.t/f.sig2sqrt2)) ...
    - log(erf(f.t./f.sig1sqrt2));
f.term2= - f.t^2/(2*f.k1sq) ...
    -log(2*sqrt(pi));
f.term3= .5 * (log(pi)-log(2)) ...
    - log(f.k_2) ...
    - 2*log(f.sig) ...
    + f.t^2 / (2*f.k2sq);
f.term4= (2/(f.sigsq*sqrt(2*pi)));
f.term5= (1/f.sigsq2-1/f.sigsq1);
end


function [y,dy] = trun_inline(x,f)
y = zeros(size(x));

erfperf=erf(f.tk_1sqrt2 + x*f.k1s2) + erf(f.tk_1sqrt2 - x*f.k1s2);
terferf=2 - erf(f.tk_2sqrt2 + x*f.k2s2) - erf(f.tk_2sqrt2 - x*f.k2s2);
tol=1e-10;%1e-10;

idx_in=find(erfperf > tol & terferf > tol);
x_in = x(idx_in);
y(idx_in) = f.d(x_in);

idx_approx_large=find(erfperf <= tol);
x_out_large = x(idx_approx_large);
y(idx_approx_large) = f.da(x_out_large);

idx_approx_small=find(terferf <= tol);
x_out_small = x(idx_approx_small);
y(idx_approx_small) = f.da0(x_out_small);

if nargout == 2
dy= zeros(size(x));
dy(idx_in)= f.dd(x_in);
dy(idx_approx_large)= f.dda(x_out_large);
dy(idx_approx_small)= f.dda0(x_out_small);
end

end