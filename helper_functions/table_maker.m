function table_maker(models, filename)

bootstrap = false;

fid = fopen(filename,'w+');

% table to csv
[score] = compare_models(models);

score = -score;

nModels = size(score, 1);
nSubjects = size(score, 2);

model_names = cell(1, nModels);

name_str1 = '';
for d = 1:length(models(1).extracted);
    name_str1 = [name_str1 models(1).extracted(d).name];
end

for m = 1:nModels
    model_names{m} = rename_models(models(m).name);
    name_str = '';
    for d = 1:length(models(m).extracted);
        name_str = [name_str models(m).extracted(d).name];
    end
    if length(models(m).extracted) ~= nSubjects
        error('nSubjects doesn''t match up!')
    end
    if ~strcmp(name_str, name_str1)
        error('names don''t match up!')
    end
end
model_names = fliplr(model_names);

fprintf(fid, '%i subjects,', nSubjects);
fprintf(fid, '%s,', model_names{1:end-2});
fprintf(fid, '%s\n', model_names{end-1});




for m = 1:nModels-1
    delta = bsxfun(@minus, score(m,:), score);
    
    if bootstrap
        bootstat = bootstrp(1e4, @mean, delta'); % 1e4 is sufficient
%         group_mean = mean(bootstat)';
        midrange = .95;
        group_quantiles = quantile(bootstat, [.5 - midrange/2, .5 + midrange/2]);
%         if size(group_quantiles, 1) == 1
%             group_quantiles = group_quantiles';
%         end
        fprintf(fid, '%s,', model_names{end-m+1});
        if m ~= nModels-1
            fprintf(fid, '{[%.0f, %.0f]},', fliplr([group_quantiles(1, 2+m:end) group_quantiles(2, 2+m:end)]'));
        end
        fprintf(fid, '{[%.0f, %.0f]}\n', [group_quantiles(1, 1+m) group_quantiles(2, 1+m)]');

    else
        group_mean = mean(delta, 2);
        group_sem = std(delta, [], 2)./sqrt(nSubjects);
        fprintf(fid, '%s,', model_names{end-m+1});
        if m ~= nModels-1
            fprintf(fid, '%.0f\\ \\pm\\ %.0f,', fliplr([group_mean(2+m:end) group_sem(2+m:end)]'));
        end
        fprintf(fid, '%.0f\\ \\pm\\ %.0f%s\n', [group_mean(1+m) group_sem(1+m)]', repmat(',',1,m-1));
    end

end

fclose(fid);
