function p_id = find_parameter(regex, c)
p_id = find(~cellfun(@isempty, regexp(c.parameter_names, regex)));
end