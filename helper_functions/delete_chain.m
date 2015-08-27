function extracted = delete_chain(extracted, chains_to_delete)
nChains = length(extracted.p);
chains_to_keep = setdiff(1:nChains, chains_to_delete);

fields = fieldnames(extracted);
%%
for f = 1:length(fields)
    if length(extracted.(fields{f})) == nChains && iscell(extracted.(fields{f}))
        extracted.(fields{f}) = extracted.(fields{f})(chains_to_keep);
    end
end