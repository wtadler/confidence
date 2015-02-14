function models = parameter_constraints(models)
nModels = length(models);

for m_id = 1 : nModels
    c = models(m_id);
    
    options = {'multi_lapse','partial_lapse','repeat_lapse','choice_only','symmetric','d_noise','free_cats','non_overlap','ori_dep_noise','diff_mean_same_std'};
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
        if isa(c.(fields{f}), 'char')
            str = [str c.(fields{f})];
        elseif isa(c.(fields{f}), 'double') && all(size(c.(fields{f})) == [1 1]) && c.(fields{f})==1
            str = [str ' : ' fields{f}];
        end
    end
    c.name = str; % model name
    %%
    c.parameter_names = {
        'alpha'
        'beta'
        'logsigma_0'
        'b_n3_d'
        'b_n2_dTerm'
        'b_n1_dTerm'
        'b_0_dTerm'
        'b_1_dTerm'
        'b_2_dTerm'
        'b_3_dTerm'
        'b_n3_x'
        'b_n2_xTerm'
        'b_n1_xTerm'
        'b_0_xTerm'
        'b_1_xTerm'
        'b_2_xTerm'
        'b_3_xTerm'
        'm_n3'
        'm_n2Term'
        'm_n1Term'
        'm_0Term'
        'm_1Term'
        'm_2Term'
        'm_3Term'
        'logsigma_d'
        'lambda'
        'lambda_1'
        'lambda_4'
        'lambda_g'
        'lambda_r'
        'logsig1'
        'logsig2Term'
        'sig_amplitude'
        'b_0_dChoice'
        'b_0_xChoice'
        'm_0Choice'
        };
    log_params = strncmpi(c.parameter_names,'log',3);
    
    %               alp bet sig0bn3dbn2dbn1dbn0db1d b2d b3d bn3xbn2xbn1xb0x b1x b2x b3x mn3 mn2 mn1 m0  m1  m2  m3  sigdlm  lm1 lm4 lmg lmr s1  s2  sa  b0dcb0xcm0xc
    c.lb       = [  0   0   0   -60 0   0   0   0   0   0   0   0   0   0   0   0   0   -30 0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   -60 0   -30   ];
    c.lb_gen   = [  0   0   0   -2  .1  .1  .1  .1  .1  .1  0   2   2   2   2   2   2   -2  .2  .2  .2  .2  .2  .2  0   0   0   0   0   0   2   8   2   -2  3   0];
    c.ub       = [  50  3   30  60  10  10  10  10  10  10  90  30  30  30  30  30  30  30  10  10  10  10  10  10  20  .25 .25 .25 .25 .5  25  25  25  60  40  30 ];
    c.ub_gen   = [  50  8   30  1.2 1.2 1.2 1.2 1.2 1.2 1.2 3   5   5   5   5   5   5   1   1   1   1   1   1   1   3   .1  .1  .1  .2  .1  4   10  10  1.2 8   2];
    
    c.beq      = [  1   1   1   -2  .7  .7  .7  .7  .7  .7  2   2   2   2   2   2   2  -2   .7  .7  .7  .7  .7  .7  0   0   0   0   0   0   3   9   0   0   5   .5]';
    
    fields = {'lb','ub','lb_gen','ub_gen'}; % convert log param bounds
    c.lb(log_params) = -5;
    c.lb_gen(log_params) = -1.5;
    c.ub(log_params) = 4;
    c.ub_gen(log_params) = 1.2; % bring back to 4?
    
%     for field = 1:length(fields)
%         c.(fields{field})(log_params) = log(c.(fields{field})(log_params));
%         c.(fields{field})(c.(fields{field})==-Inf) = -100;
%         c.(fields{field})(c.(fields{field})==Inf) = 100;
%     end
    
    c = familyizer(c);
    c = symmetricizer(c);
    c = choiceizer(c);
    c = d_noiseizer(c);
    c = partial_lapseizer(c);
    c = multi_lapseizer(c);
    c = repeat_lapseizer(c);
    c = free_catsizer(c);
    c = sig_ampizer(c);
    
    % calculate uniform param prior
    c.param_prior = prod(1 ./ (c.ub - c.lb)); % this is not great. lapse param should be a beta dist, not uniform. but only applies when doing hessian, which we've moved on from.
    
    % put back into model
    fields = fieldnames(c);
    for f = 1:length(fields)
        models(m_id).(fields{f}) = c.(fields{f});
    end
end
end


function c = familyizer(c)
% strip out bound parameters from families other than the specified one
optbounds = find(~cellfun(@isempty, regexp(c.parameter_names, 'b_.*d')));
xbounds   = find(~cellfun(@isempty, regexp(c.parameter_names,'b_.*x')));
slopebounds=find(~cellfun(@isempty, regexp(c.parameter_names, 'm_')));

if strcmp(c.family, 'quad') || strcmp(c.family, 'lin')
    otherbounds = optbounds;
elseif strcmp(c.family, 'fixed') || strcmp(c.family, 'MAP')
    otherbounds = union(optbounds, slopebounds);
elseif strcmp(c.family, 'opt')
    otherbounds = union(xbounds, slopebounds);
end

% for symmetric task, need to shift the range of the x bound to allow it to be negative.
% (COME BACK TO THIS). might want to just add a new set of params. 1/23)
if c.diff_mean_same_std && (strcmp(c.family, 'quad') || strcmp(c.family, 'lin') || strcmp(c.family, 'fixed'))
    gen_range = c.ub_gen(xbounds(1)) - c.lb_gen(xbounds(1));
    range = c.ub(xbounds(1)) - c.lb(xbounds(1));
    gen_fields = {'lb_gen','ub_gen'};
    fields = {'lb','ub'};
    
    for f = 1:2
        c.(gen_fields{f})(xbounds) = c.(gen_fields{f})(xbounds)-gen_range/2;
        c.(fields{f})(xbounds) = c.(fields{f})(xbounds)-range/2;
    end
end

c = p_stripper(c,otherbounds);
end

function c = symmetricizer(c)
if c.symmetric
    % if doing symmetric (opt) model, strip out negative D parameters
    b0term = find(~cellfun(@isempty, regexp(c.parameter_names,'b_0_dTerm')));
    c.parameter_names{b0term} = 'b_0_d';
    fields = {'lb','ub','lb_gen','ub_gen'};
    for f = 1:length(fields)
        c.(fields{f})(b0term)=c.(fields{f})(b0term-3);
    end
    
    negDPs = find(~cellfun(@isempty,regexp(c.parameter_names,'b_n')));
    
    c = p_stripper(c,negDPs);
end
end

function c = choiceizer(c)
choice_bounds = find(~cellfun(@isempty,regexp(c.parameter_names, 'Choice$')));
if c.choice_only
    % if only doing choice, strip out extra bound parameters
    all_bounds = find(~cellfun(@isempty, regexp(c.parameter_names, '[bm]_')));
    extra_bounds = setdiff(all_bounds, choice_bounds);
    c = p_stripper(c,extra_bounds);
    % chop Choice off the end
    choice_bounds = find(~cellfun(@isempty,regexp(c.parameter_names, 'Choice$'))); % find params again
    for cb = choice_bounds'
        c.parameter_names{cb} = c.parameter_names{cb}(1:end-6);
    end
    
elseif ~c.choice_only
    c = p_stripper(c,choice_bounds);
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
end

end

function c = sig_ampizer(c)
% if not doing ori dep noise, strip out sig_amplitude
if ~c.ori_dep_noise
    sig_ampP = find(~cellfun(@isempty, regexp(c.parameter_names, 'sig_amplitude')));
    c = p_stripper(c, sig_ampP);
end
end



function c = p_stripper(c, p_to_remove)
% this is called by the above functions to strip out specified parameter from the named fields.
p = setdiff(1:length(c.lb), p_to_remove); % p is the parameters to keep
fields = {'parameter_names','lb','lb_gen','ub','ub_gen','beq'};
for f = 1:length(fields);
    c.(fields{f}) = c.(fields{f})(p);
end
end