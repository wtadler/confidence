function str_out = rename_var_labels(str_in)
if isstr(str_in)
    one_input = true;
    str_in = {str_in};
else
    one_input = false;
end

for s = 1:length(str_in)
    switch str_in{s}
        case 'tf'
            out = 'prop. correct';
        case 'resp'
            out = 'mean button press';
        case 'g'
            out = 'mean confidence';
        case 'Chat'
            out = 'prop. response "cat. 1"';
        case 'rt'
            out = 'reaction time (s)';
        case 'proportion'
            out = 'prop.';
    end
    
    str_out{s} = out;
end

if one_input
    str_out = str_out{1};
end