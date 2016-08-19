function names_out = rename_models(names_in, varargin)
% eventually, this would be cleaner if it took in a model struct rather than the strings
short = false;
latex = false;
assignopts(who, varargin);

if iscell(names_in)
    names_out = cell(size(names_in));
    for n = 1:length(names_in)
        names_out{n} = namify(names_in{n}, short, latex);
    end
elseif isstr(names_in)
    names_out = namify(names_in, short, latex);
end

end

function name_out = namify(name_in, short, latex)
if isstrprop(name_in(1), 'upper') && ~strcmp(name_in(1:3), 'MAP') % if names have already been namified, the first character will be upper case
    name_out = name_in;
    return
end

% FAMILIES
if regexp(name_in, '^opt')
    if regexp(name_in, 'joint_d')
        if latex
            name_out = '$\\text{Bayes}_\\text{S}$';
        else
            name_out = 'Bayes_{Strong}';
        end
    elseif regexp(name_in, 'choice_only')
        name_out = 'Bayes';
    elseif regexp(name_in, 'symmetric')
        if latex
            name_out = '$\\text{Bayes}_\\text{W}$';
        else
            name_out = 'Bayes_{Weak}';
        end
    else
        if latex
            name_out = '$\\text{Bayes}_\\text{U}$';
        else
            name_out = 'Bayes_{Ultraweak}';
        end
    end    
elseif regexp(name_in, '^lin')
    name_out = 'Lin';
elseif regexp(name_in, '^quad')
    name_out = 'Quad';
elseif regexp(name_in, '^neural1')
    name_out = 'Linear Neural';
    if short
        name_out = 'Lin. Neur.';
    end
elseif regexp(name_in, '^fixed')
    name_out = 'Fixed';
elseif regexp(name_in, '^MAP')
    name_out = 'Orientation Estimation';
    if short
        name_out = 'Ori. Est.';
    end
end

if short
    return
end

if regexp(name_in, 'd_noise')
    name_out = [name_out, ' + {\itd} noise'];
end


if isempty(regexp(name_in, 'joint_task_fit'))
    if regexp(name_in, 'diff_mean_same_std')
        name_out = [name_out ', A'];
    else
        name_out = [name_out ', B'];
    end
end

if regexp(name_in, 'nFreesigs')
    name_out = [name_out, ' + non-param. \sigma'];
end

if regexp(name_in, 'choice_only')
    name_out = [name_out, ', choice only'];
end

if regexp(name_in, 'separate_measurement_and_inference_noise')
    name_out = [name_out, ' + sep. meas. and inf. noise'];
end
if regexp(name_in, 'noise_fixed')
    name_out = [name_out, ', noise fixed'];
end

if regexp(name_in, 'measurement_fixed')
    name_out = [name_out, ', measurement noise fixed'];
end

if regexp(name_in, 'free_cats')
    name_out = [name_out, ' + free \sigma_{\itC}'];
end

end