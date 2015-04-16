function p = random_param_generator(sets, model, varargin);
% this gets called by optimize_fcn. any others?

fixed_params = [];
generating_flag = false;
assignopts(who,varargin);

if generating_flag
    lb=model.lb_gen;
    ub=model.ub_gen;
else
    lb=model.lb;
    ub=model.ub;
end

contrast_type = 'new';
switch contrast_type
    case 'old'
        contrasts=exp(-4:.5:-1.5);
        sig_lim = [20 3 .2]; % [max for highest sigma, max for lowest sigma, minimum difference between the two]
    case 'new'
        sig_lim = [exp(ub(1)) exp(ub(2)) .2]; % bounds are handled directly by ub in this case
end

%A=model.A;
%b=model.b;
%Aeq=model.Aeq;
beq=model.beq;
%monotonic_params = model.monotonic_params;

if min(size(lb)~=1) || min(size(ub)~=1)
    error('lb and ub must be vectors.')
elseif length(lb) ~= length(ub)
    error('lb and ub must be the same size.')
end

% if isempty(A)
%     A = zeros(length(lb), length(lb));
% end
% if isempty(b)
%     b = zeros(length(lb), 1);
% end

nParams = length(lb);

% reshape lb and ub into column vectors
lb = reshape(lb,nParams,1);
ub = reshape(ub,nParams,1);
%%
p = zeros(nParams,sets);


% this can be parfor if necessary. shouldn't have to be though.
for i = 1 : sets;
    maxsig = Inf;
    minsig = Inf;
    %x = zeros(nParams,1);
    while maxsig > sig_lim(1) || minsig > sig_lim(2) || maxsig - minsig < sig_lim(3)% || ~all(A * x <= b)
        x = lb + rand(nParams,1) .* (ub - lb);
        %         if iscell(monotonic_params) % for multiple sets of monotonic params (so far, only lin2 and quad2)
        %             for m = 1:length(monotonic_params)
        %                 mp = monotonic_params{m};
        %                 x(mp) = sort(x(mp));
        %             end
        %         else
        %             x(monotonic_params) = sort(x(monotonic_params));
        %         end

        switch contrast_type
            case 'old'
                maxsig = sqrt(x(3).^2 + x(1) .* contrasts(1) .^ - x(2));
                minsig = sqrt(x(3).^2 + x(1) .* contrasts(6) .^ - x(2));
            case 'new'
                maxsig = exp(x(1));
                minsig = exp(x(2));
                
        end
    end
    
    p(:,i) = x;
end

if ~isempty(fixed_params)
    p(fixed_params,:) = repmat(beq(fixed_params),1,sets);
end