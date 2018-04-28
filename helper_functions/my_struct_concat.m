function structure = my_struct_concat(varargin)
% makes sure that a bunch of structures have the same fields, then concatenates them.

nArgs = length(varargin);
if nArgs<2
    error('need at least 2 args')
end

structure = varargin{1};
for i = 2:nArgs
    structure = actual_concat(structure, varargin{i});
end

    function s=actual_concat(struct1, struct2)
        % find all the fields in struct1. if those fields aren't in struct1, add empties
        struct1 = samify_structs(struct1, struct2);
        
        % vice versa
        struct2 = samify_structs(struct2, struct1);
        
        
        % concatenate
        s = [struct1, struct2];
    end
end