function list = name_all_models(models)
nModels = length(models);
list = cell(1, nModels);
for m = 1:nModels
    list{m} = rename_models(models(m).name);
    fprintf('%i: %s\n', m, list{m})
end