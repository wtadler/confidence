function idx = array_number_generator(models, subjects, nModels, nSubjects, nChains)

models=reshape(models, 1, length(models));
subjects=reshape(subjects, 1, length(subjects));

firstchain = nModels * (subjects - 1) + models;

idx = [];
for i = 1:nChains
    idx = [idx firstchain + nModels*nSubjects*(i-1)];
end

idx = sort(idx);

str = strrep(num2str(idx,'%i,'),' ','');

fprintf([str(1:end-1) '\n'])
