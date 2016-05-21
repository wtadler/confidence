function names_out = rename_models(names_in, family_only)
% eventually, this would be cleaner if it took in a model struct rather than the strings

if ~exist('family_only', 'var')
    family_only = true;
end

if iscell(names_in)
    names_out = cell(size(names_in));
    for n = 1:length(names_in)
        names_out{n} = namify(names_in{n}, family_only);
    end
elseif isstr(names_in)
    names_out = namify(names_in, family_only);
end

end

function name_out = namify(name_in, family_only)
if isempty(regexp(name_in, ':')) % if names have already been namified, they won't have any colons
    name_out = name_in;
    return
end

% FAMILIES
if regexp(name_in, '^opt')
    if regexp(name_in, 'joint_d')
        name_out = ['Bayes_{Strong}'];    
    elseif regexp(name_in, 'choice_only')
        name_out = ['Bayes'];
    elseif regexp(name_in, 'symmetric')
        name_out = ['Bayes_{Weak}'];
    else
        name_out = ['Bayes_{Ultraweak}'];
    end
    
    if regexp(name_in, 'd_noise')
        name_out = [name_out, ' + $d$ noise'];
    end
    
elseif regexp(name_in, '^lin')
    name_out = ['Lin'];
elseif regexp(name_in, '^quad')
    name_out = ['Quad'];
elseif regexp(name_in, '^neural1')
    name_out = ['Neural'];
elseif regexp(name_in, '^fixed')
    name_out = ['Fixed'];
elseif regexp(name_in, '^MAP')
    name_out = ['Orientation Estimation'];
end

if family_only
    return
end

if isempty(regexp(name_in, 'joint_task_fit'))
    if regexp(name_in, 'diff_mean_same_std')
        name_out = [name_out ' + A'];
    else
        name_out = [name_out ' + B'];
    end
end

if regexp(name_in, 'nFreesigs')
    name_out = [name_out, ' + free sigs'];
end

if regexp(name_in, 'choice_only')
    name_out = [name_out, ' + choice only'];
end

if regexp(name_in, 'separate_measurement_and_inference_noise')
    name_out = [name_out, ' + sep meas/inf noise'];
end
if regexp(name_in, 'noise_fixed')
    name_out = [name_out, ' + noise fixed'];
end

if regexp(name_in, 'measurement_fixed')
    name_out = [name_out, ' + measurement noise fixed'];
end

if regexp(name_in, 'free_cats')
    name_out = [name_out, ' + free cat. sigs'];
end

end