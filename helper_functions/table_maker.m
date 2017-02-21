function table_maker(models, filename, varargin)
%%
bootstrap = true;
latex = true;
report_significance = false;
    p_correction = 'hb'; % 'hb' (holm-bonferroni) or 'b' (bonferroni). bonferroni is slightly more conservative
halftable = true;
assignopts(who, varargin);

fid = fopen(filename,'w+');

% table to csv
score = compare_models(models, 'MCM', 'loopsis', 'fig_type', '');

nModels = size(score, 1);
nSubjects = size(score, 2);
nCells = sum(1:nModels-1);
alpha = .05; % will be bonferroni-corrected below
CI = .95;
bootsamples = 1e4;

significance = cell(nModels-1, nModels-1);
p = zeros(nModels-1, nModels-1);
means = p;
sems = p;
quantiles = zeros(nModels-1, nModels-1, 3);

rng(0);

name_str1 = '';
for d = 1:length(models(1).extracted);
    name_str1 = [name_str1 models(1).extracted(d).name];
end

model_names = rename_models(models, 'latex', latex, 'short', false, 'abbrev', true);
for m = 1:nModels
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
    
    if m == nModels
        break
    else
        delta = bsxfun(@minus, score(m,:), score);
        bootstat = bootstrp(bootsamples, @sum, delta'); % 1e4 is sufficient MEAN or SUM
        %         group_mean = mean(bootstat)';
        group_quantiles = fliplr(quantile(bootstat, [.5 - CI/2, .5, .5 + CI/2]));
        group_quantiles = permute(group_quantiles(:, 1:nModels-m), [2 3 1]);
        quantiles(1:nModels-m, m,:) = group_quantiles;
        
        means(1:nModels-m, m) = fliplr(mean(delta(1+m:end, :), 2))';
        sems(1:nModels-m, m) = fliplr(std(delta(1+m:end, :), [], 2)/sqrt(nSubjects))';
        
        if report_significance
            for comparison = 1+m:nModels
                [h, p(nModels+1-comparison, m)] = ttest(score(m,:), score(comparison,:), 'alpha', alpha/nCells);
                
                if strcmp(p_correction, 'b')
                    if h
                        significance{nModels+1-comparison, m} = '*';
                    else
                        significance{nModels+1-comparison, m} = '';
                    end
                end
            end
        end
    end
end

if ~halftable
    full_table(:,:,1)=[[zeros(nModels-1,1),fliplr(quantiles(:,:,1))];zeros(1,nModels)]+[[zeros(1,nModels-1);flipud(-quantiles(:,:,3)')],zeros(nModels,1)];
    full_table(:,:,2)=[[zeros(nModels-1,1),fliplr(quantiles(:,:,2))];zeros(1,nModels)]+[[zeros(1,nModels-1);flipud(-quantiles(:,:,2)')],zeros(nModels,1)];
    full_table(:,:,3)=[[zeros(nModels-1,1),fliplr(quantiles(:,:,3))];zeros(1,nModels)]+[[zeros(1,nModels-1);flipud(-quantiles(:,:,1)')],zeros(nModels,1)];
    full_table = flipud(fliplr(full_table));
end


if strcmp(p_correction, 'hb')
    nZeroCells = sum(1:nModels-2);
    [p_sort, sort_idx] = sort(p(:));%(p(:) > 0));
    last_sig_cell = find(diff(p_sort > alpha./(nCells + 1 - (-nZeroCells+1:nCells))'));
    significant_cells = sort_idx(nZeroCells+1:last_sig_cell);
    significance(significant_cells) = {'*'};
end
model_names_rev = model_names;
model_names = fliplr(model_names);

for i = 1:nModels
    nParams(i) = length(models(nModels+1-i).parameter_names);
end
nParams_rev = fliplr(nParams);

if latex
    %%
    fprintf(fid, ['\\begin{tabular}{cc|'...
        repmat('c',1,nModels-1)...
        '}\n&'...
        sprintf('&%i pars.', nParams_rev(1:end-1)),'\\\\\n&'...
        sprintf('&%s', model_names_rev{1:end-1}), '\\\\\n'... % header row
        '\\hline\n'...
        '\\Centerstack{'...
        sprintf('%i pars.\\\\cr', nParams(1:end-2))...
        num2str(nParams(end-1))...
        ' pars.'...
        '} &'...
        '\n\\Centerstack{'...
        sprintf('%s\\\\cr ', model_names{1:end-2})... % header col
        model_names{end-1}, '} &\n']);
else
    if halftable
        fprintf(fid,',,,');
        fprintf(fid, '%i pars.,', nParams_rev(1:end-2));
        fprintf(fid, '%i pars.\n', nParams_rev(end-1));
        fprintf(fid,',,,');
        fprintf(fid, '"%s",', model_names_rev{1:end-2});
        fprintf(fid, '"%s"\n', model_names_rev{end-1});
    else
        fprintf(fid,',,,');
        fprintf(fid, '%i pars.,', nParams_rev(1:end-1));
        fprintf(fid, '%i pars.\n', nParams_rev(end));
        fprintf(fid,',,,');
        fprintf(fid, '"%s",', model_names_rev{1:end-1});
        fprintf(fid, '"%s"\n', model_names_rev{end});        
    end
end

if halftable
    total_models = 1:nModels-1;
else
    total_models = 1:nModels;
end

for m = total_models
    if latex
        fprintf(fid, ['{\\ensurestackMath{\n'...
                '\\alignCenterstack{']);
    end
    
    if bootstrap
        if ~latex % print row
            if halftable
                fprintf(fid, ',%i pars.,"%s",', nParams(m), model_names{m});
                for comparison = 1:nModels-m-1
                    fprintf(fid, '"%.0f [%.0f, %.0f]",', quantiles(m,comparison,[2 1 3]));
                end
                fprintf(fid, '"%.0f [%.0f, %.0f]"\n', quantiles(m,nModels-m,[2 1 3]));
            else
                fprintf(fid, ',%i pars.,"%s",', nParams_rev(m), model_names_rev{m});
                for comparison = 1:nModels-1
                    fprintf(fid, '"%.0f [%.0f, %.0f]",', full_table(m, comparison, [2 1 3]));
                end
                fprintf(fid, '"%.0f [%.0f, %.0f]"\n', full_table(m,nModels,[2 1 3]));
            end
        else % print col
            for comparison = fliplr(1+m:nModels)
                %                 fprintf(fid, '%.0f\\ [%.0f, %.0f]%s', quantiles(nModels+1-comparison, m, 2), quantiles(nModels+1-comparison, m, 1), quantiles(nModels+1-comparison, m, 3), significance{nModels+1-comparison, m});
                if report_significance && strcmp(significance{nModels+1-comparison, m}, '*')
                    fprintf(fid, '\\textbf{%.0f\\ [%.0f, %.0f]}', quantiles(nModels+1-comparison, m, 2), quantiles(nModels+1-comparison, m, 1), quantiles(nModels+1-comparison, m, 3));
                else
                    fprintf(fid, '%.0f\\ [%.0f, %.0f]', quantiles(nModels+1-comparison, m, 2), quantiles(nModels+1-comparison, m, 1), quantiles(nModels+1-comparison, m, 3));
                end

                if comparison == 1+m % end of column
                    fprintf(fid, '%s}}}\n', repmat('\cr &', 1, m-1));
                    if m ~= nModels-1 % end of last column
                        fprintf(fid, '&\n');
                    end
                else
                    fprintf(fid, '\\cr ');
                end
            end
        end
        
    else
        error('fix this')
        group_mean = mean(delta, 2);
        group_sem = std(delta, [], 2)./sqrt(nSubjects);
        
        if latex
            if m ~= nModels-1
                fprintf(fid, '%.0f\\pm&%.0f\\cr ', fliplr([-group_mean(2+m:end) group_sem(2+m:end)]'));
                fprintf(fid, '%.0f\\pm&%.0f%s}}}\n&\n', [-group_mean(1+m) group_sem(1+m)]', repmat('\cr &', 1, m-1));
            else
                fprintf(fid, '%.0f\\pm&%.0f%s}}}\n', [-group_mean(1+m) group_sem(1+m)]', repmat('\cr &', 1, m-1));
            end
            
        else
            fprintf(fid, '%s,', model_names{end-m+1});
            if m ~= nModels-1
                fprintf(fid, '$%.0f\\ \\pm\\ %.0f$,', fliplr([group_mean(2+m:end) group_sem(2+m:end)]'));
            end
            fprintf(fid, '$%.0f\\ \\pm\\ %.0f$%s\n', [group_mean(1+m) group_sem(1+m)]', repmat(',',1,m-1));
        end
    end
    
end

if latex
    fprintf(fid, '\\end{tabular}');
end

fclose(fid);
