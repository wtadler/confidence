function extracted = burn_chain(extracted, chains_to_burn, burn_prop)
% for partial burning. to delete a whole chain, use delete_chain.m
% burn_prop of .2 will burn the first 20% of the specified chains

nChains = length(extracted.p);
% chains_to_keep = setdiff(1:nChains, chains_to_delete);

fields = fieldnames(extracted);
%%

for c = chains_to_burn
    chain_length = length(extracted.p{c});
    first_sample = ceil(burn_prop * chain_length);
    for f = 1:length(fields)
        if length(extracted.(fields{f})) == nChains && iscell(extracted.(fields{f}))
            extracted.(fields{f}){c} = extracted.(fields{f}){c}(first_sample:end,:);
        end
    end
end