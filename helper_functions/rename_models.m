function names_out = rename_models(names_in, varargin)
% eventually, this would be cleaner if it took in a model struct rather than the strings
short = false;
abbrev = false;
latex = false;
task = true;
choice = true;
assignopts(who, varargin);

if isstruct(names_in)
    names_out = cell(size(names_in));
    for n = 1:length(names_in)
        names_out{n} = namify(names_in(n).name, short, abbrev, latex, task, choice);
        fprintf('%i: %s\n', n, names_out{n})
    end
    
elseif iscell(names_in)
    names_out = cell(size(names_in));
    for n = 1:length(names_in)
        names_out{n} = namify(names_in{n}, short, abbrev, latex, task, choice);
    end
elseif isstr(names_in)
    names_out = namify(names_in, short, abbrev, latex, task, choice);
end

end

function name_out = namify(name_in, short, abbrev, latex, task, choice)
if isstrprop(name_in(1), 'upper') && ~strcmp(name_in(1:3), 'MAP') % if names have already been namified, the first character will be upper case
    name_out = name_in;
    return
end

% FAMILIES
if regexp(name_in, '^opt')
    if regexp(name_in, 'joint_d')
        if latex
            name_out = '$\\text{Bayes}_\\text{U}$';
        else
            if abbrev
                name_out = 'Bayes_U';
            else
                name_out = 'Bayes_{Ultrastrong}';
            end
        end
    elseif any(regexp(name_in, 'choice_only')) || (any(regexp(name_in, 'diff_mean_same_std')) && ~any(regexp(name_in, 'joint_task_fit')))
        name_out = 'Bayes';
    elseif regexp(name_in, 'symmetric')
        if latex
            name_out = '$\\text{Bayes}_\\text{S}$';
        else
            if abbrev
                name_out = 'Bayes_S';
            else
                name_out = 'Bayes_{Strong}';
            end
        end
    else
        if latex
            name_out = '$\\text{Bayes}_\\text{W}$';
        else
            if abbrev
                name_out = 'Bayes_W';
            else
                name_out = 'Bayes_{Weak}';
            end
        end
    end
elseif regexp(name_in, '^lin')
    name_out = 'Lin';
elseif regexp(name_in, '^quad')
    name_out = 'Quad';
elseif regexp(name_in, '^neural1')
    if abbrev
        name_out = 'Lin. Neur.';
    else
        name_out = 'Linear Neural';
    end
elseif regexp(name_in, '^fixed')
    name_out = 'Fixed';
elseif regexp(name_in, '^MAP')
    if abbrev
        name_out = 'Ori. Est.';
    else
        name_out = 'Orientation Estimation';
    end
end


if regexp(name_in, 'd_noise')
    if latex
        name_out = [name_out, '-$d$N'];
    else
        name_out = [name_out, '-{\itd}N'];
    end
end

if regexp(name_in, 'fisher_info')
    name_out = [name_out, ' + precision'];
end


if short
    return
end



if regexp(name_in, 'nFreesigs')
    if latex
        name_out = [name_out, ' + non-param. $\\sigma$'];
    else
        name_out = [name_out, ' + non-param. \sigma'];
    end
end


if regexp(name_in, 'separate_measurement_and_inference_noise')
    name_out = [name_out, ' + sep. meas. and inf. noise'];
end
if regexp(name_in, 'free_cats')
    if latex
        name_out = [name_out, ' + free $\\sigma_C$'];
    else
        name_out = [name_out, ' + free \sigma_{\itC}'];
    end
end


if ~latex
    if regexp(name_in, 'noise_fixed')
        name_out = [name_out, ', noise fixed'];
    end
    
    if regexp(name_in, 'measurement_fixed')
        name_out = [name_out, ', measurement noise fixed'];
    end
    
    if task
        if isempty(regexp(name_in, 'joint_task_fit'))
            if regexp(name_in, 'diff_mean_same_std')
                name_out = [name_out ', A'];
            else
                name_out = [name_out ', B'];
            end
        end
    end
    if choice
        if regexp(name_in, 'choice_only')
            name_out = [name_out, ', choice only'];
        end
    end
    
end


end