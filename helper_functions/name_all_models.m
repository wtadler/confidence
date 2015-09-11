function name_all_models(models)
for m = 1:length(models)
    fprintf('%i: %s\n', m, rename_models(models(m).name))
end