function struct = samify_structs(struct, struct_to_match)

fields = fieldnames(struct_to_match);
for f = 1:length(fields)
    if ~isfield(struct, fields{f})
        [struct(:).(fields{f})] = deal([]);
    end
end
struct = orderfields(struct);
end