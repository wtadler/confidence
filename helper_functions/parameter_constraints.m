function models = parameter_constraints(models)

nModels = length(models);

for m_id = 1 : nModels
    c = models(m_id);
    
    options = {'multi_lapse','partial_lapse','repeat_lapse','choice_only','symmetric','d_noise','free_cats','non_overlap'};
    for o = 1:length(options)
        if ~isfield(c,options{o}) || isempty(c.(options{o}))
            c.(options{o}) = 0;
        end
    end
    
    
    if c.choice_only
        % the below options are only for confidence models. they don't make sense if choice_only has been specified
        c.partial_lapse = 0;
        c.multi_lapse = 0;
        c.symmetric = 0;
    end
    
    if ~strcmp(c.family, 'opt')
        % the below options are only for opt models. they don't make sense if opt has not been specified
        c.symmetric = 0;
        c.d_noise = 0;
    end
    
    % give model its name based on the settings. in future, might be better to report the alternative rather than just leave it out when it's zero.
    fields = fieldnames(c);
    str='';
    for f = 1 : length(fields)
        if ischar(c.(fields{f}))
            str = [str c.(fields{f})];
        elseif c.(fields{f})
            str = [str ' : ' fields{f}];
        end
    end
    c.name = str; % model name
    
    c.parameter_names = {
        'alpha'
        'beta'
        'sigma_0'
        'b_n3_d'
        'b_n2_d'
        'b_n1_d'
        'b_0_d'
        'b_1_d'
        'b_2_d'
        'b_3_d'
        'b_n3_x'
        'b_n2_x'
        'b_n1_x'
        'b_0_x'
        'b_1_x'
        'b_2_x'
        'b_3_x'
        'm_n3'
        'm_n2'
        'm_n1'
        'm_0'
        'm_1'
        'm_2'
        'm_3'
        'sigma_d'
        'lambda'
        'lambda_1'
        'lambda_4'
        'lambda_g'
        'lambda_r'
        'sig1'
        'sig2'
        };
    
    c.lb       = [  0   0   0   -40 -40 -40 -40 -40 -40 -40 0   0   0   0   0   0   0   -30 -30 -30 -30 -30 -30 -30 0   0   0   0   0   0   0   0];
    c.lb_gen   = [  0   0   0   -2  -2  -2  -1  -1  -1  -1  0   0   0   0   0   0   0   -2  -2  -2  -.1 0   0   0   0   0   0   0   0   0   2   10];
    c.ub       = [  50  8   30  40  40  40  40  40  40  40  40  40  40  40  40  40  40  30  30  30  30  30  30  30  20  .5  .25 .25 .5  .5  25  25];
    c.ub_gen   = [  50  8   30  1.2 1.2 1.2 1.2 1.2 1.2 1.2 20  20  20  20  20  20  20  0   0   0   .1  3   3   3   3   .4  .1  .1  .2  .1  4   14];
    c.beq      = [  1   1   1   -2  -1.5 -.5 0  .5  1.5 2   2   4   5   5.5 6   7   8   -2  -1  -.5 0   .5  1   2   0   0   0   0   0   0   3   12]';
    
    nParams = length(c.lb);
    %%
    c.monotonic_params = {4:10,11:17,18:24,31:32};
    A = [];
    for mpunit=1:length(c.monotonic_params)
        l = length(c.monotonic_params{mpunit})-1;
        unit = [eye(l) zeros(l,1)] + [zeros(l,1) -eye(l)];
        low = min(c.monotonic_params{mpunit});
        high= max(c.monotonic_params{mpunit});
        row = [zeros(l, low-1) unit zeros(l, nParams - high)];
        A = [A;row];
    end
    c.A = [A;
        zeros(1, 25) 1 2 2 1 1 0 0];
    c.b = [zeros(size(A,1),1); 1];
    
    c.monotonic_params = {};
    
    % then strip out the ones we don't need.
    c = familyizer(c);
    
    if strcmp(c.family, 'quad') || strcmp(c.family, 'lin')
        b = find(~cellfun(@isempty,regexp(c.parameter_names, 'b_')));
        m = find(~cellfun(@isempty,regexp(c.parameter_names, 'm_')));
        c.monotonic_params = {b,m};
    elseif strcmp(c.family, 'fixed') || strcmp(c.family, 'MAP')
        b = find(~cellfun(@isempty,regexp(c.parameter_names, 'b_')));
        c.monotonic_params = {b};
    elseif strcmp(c.family, 'opt')
        b = find(~cellfun(@isempty,regexp(c.parameter_names, 'b_')));
        c.monotonic_params = {b};
    end
    
    c = symmetricizer(c);
    c = choiceizer(c);
    c = d_noiseizer(c);
    c = partial_lapseizer(c);
    c = multi_lapseizer(c);
    c = repeat_lapseizer(c);
    c = free_catsizer(c);

    
    % calculate uniform param prior
    c.param_prior = prod(1 ./ (c.ub - c.lb));

    % put back into model
    fields = fieldnames(c);
    for f = 1:length(fields)
        models(m_id).(fields{f}) = c.(fields{f});
    end
end
end






function c = familyizer(c)
% strip out bound parameters from families other than the specified one
optbounds = find(~cellfun(@isempty, regexp(c.parameter_names, '^b.*d$')));
xbounds   = find(~cellfun(@isempty, regexp(c.parameter_names,'^b.*x$')));
slopebounds=find(~cellfun(@isempty, regexp(c.parameter_names, '^m_')));

if strcmp(c.family, 'quad') || strcmp(c.family, 'lin')
    %c.monotonic_params = [c.monotonic_params xbounds slopebounds];
    otherbounds = optbounds;
elseif strcmp(c.family, 'fixed') || strcmp(c.family, 'MAP')
    %c.monotonic_params = [c.monotonic_params xbounds];
    otherbounds = union(optbounds, slopebounds);
elseif strcmp(c.family, 'opt')
    %c.monotonic_params = [c.monotonic_params optbounds];
    otherbounds = union(xbounds, slopebounds);
end

c = p_stripper(c,otherbounds);
end

function c = symmetricizer(c)
if c.symmetric
    % if doing symmetric (opt) model, strip out negative D parameters
    c.monotonic_params{1} = c.monotonic_params{1}(1:4);
    negDPs = find(~cellfun(@isempty,regexp(c.parameter_names,'b_n')));
    c = p_stripper(c,negDPs);
    
    % trash top row of A and b (hacky)
    c.A = c.A(2:end,:);
    c.b = c.b(2:end,:);
end
end

function c = choiceizer(c)
if c.choice_only
    c.monotonic_params = {}; % no monotonic bounds in choice only
    
    % keep only rows of A that don't have negative values (this is a hacky way to get rid of the monotonic bounds in there)
    r = ~any(c.A'==-1);
    c.A = c.A(r,:);
    c.b = c.b(r);

    % if only doing choice, strip out extra bound parameters
    extra_bounds = find(~cellfun(@isempty, regexp(c.parameter_names, '^[bm]_(?!0)')));
    c = p_stripper(c,extra_bounds);
end
end

function c = d_noiseizer(c)
% if not doing d noise, strip out d noise parameter
if ~c.d_noise
    d_noiseP = find(~cellfun(@isempty, regexp(c.parameter_names, 'sigma_d')));
    c = p_stripper(c,d_noiseP);
end
end

function c = partial_lapseizer(c)
% if not doing partial lapse, strip out partial lapse parameter
if ~c.partial_lapse
    partial_lapseP = find(~cellfun(@isempty, regexp(c.parameter_names, 'lambda_g')));
    c = p_stripper(c,partial_lapseP);
end
end

function c = multi_lapseizer(c)
% if doing multilapse, strip out the single full lapse and leave the multilapse
% if not doing multilapse, strip out the multilapse and leave the full lapse
if c.multi_lapse
    unwantedLapseP = find(~cellfun(@isempty, regexp(c.parameter_names, 'lambda$')));
elseif ~c.multi_lapse % strike the two multilapses
    unwantedLapseP = find(~cellfun(@isempty, regexp(c.parameter_names, 'lambda_[14]')));
end
c = p_stripper(c,unwantedLapseP);
end

function c = repeat_lapseizer(c)
% if not doing repeat lapse, strip out repeat lapse parameter
if ~c.repeat_lapse
    repeat_lapseP = find(~cellfun(@isempty, regexp(c.parameter_names, 'lambda_r')));
    c = p_stripper(c,repeat_lapseP);
end
end

function c = free_catsizer(c)
% if not doing free cats, strip out sig1 and sig2
free_catsP = find(~cellfun(@isempty, regexp(c.parameter_names, 'sig[12]')));
if ~c.free_cats
    c = p_stripper(c,free_catsP);
else
    c.monotonic_params = [c.monotonic_params free_catsP];
end

end


function c = p_stripper(c, p_to_remove)
% this is called by the above functions to strip out specified parameter from the named fields.
    p = setdiff(1:length(c.lb), p_to_remove); % p is the parameters to keep
    fields = {'parameter_names','lb','lb_gen','ub','ub_gen','beq'};
    for f = 1:length(fields);
        c.(fields{f}) = c.(fields{f})(p);
    end
    
%     mpunit_to_delete = [];
%     for mpunit = 1 : length(c.monotonic_params)
%             save pctest.mat
% 
%         tmp = intersect(c.monotonic_params{mpunit}, p);
%         if isempty(tmp) || length(tmp)==1 % if tmp is empty or only has one param, that means we are tossing the unit
%             mpunit_to_delete = [mpunit_to_delete mpunit];% c.monotonic_params=c.monotonic_params(setdiff(1:length(c.monotonic_params), mpunit));
%         else % if the unit is reduced or kept in entirety
%             c.monotonic_params{mpunit} = tmp;
%         end
%     end
%     c.monotonic_params = c.monotonic_params(setdiff(1:length(c.monotonic_params), mpunit_to_delete));
            
    tmpA = c.A(:,p);
    nonzeroRows = find(any(tmpA'));
    c.A = tmpA(nonzeroRows, :);
    c.b = c.b(nonzeroRows);
end