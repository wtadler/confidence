function show_data(datadir, varargin)

% datadir = '/Users/will/Google Drive/Will - Confidence/Data/attention1';
dep_vars = {'tf',       'g',        'Chat',     'resp',     'rt'};
ylims    = [.5 1;       1 4;        0 1;        1 8;        .3 4];
nBins = 7; % make this odd
% plot_error_bars = true; % could eventually add functionality to just show means. or to not show means
symmetrify = false;
marg_over_s = false; %marginalize over s to just show effects of reliability
task = 'B';
linewidth = 2;
ticklength = .025;
errorbarwidth = .5; % matlab errorbar is silly. errorbar width can't be set, is 2% of total range. so we make a dummy point. only applies to marg_over_s case
gutter = [.0175 .025];
fake_st = [];
assignopts(who, varargin);

if ~isempty(fake_st)
    show_fits = true;
else
    show_fits = false;
end
real_st = compile_data('datadir',datadir);

nSubjects = length(real_st.data);

% attention and confidence experiment
if isfield(real_st.data(1).raw, 'cue_validity') && ~isempty(real_st.data(1).raw.cue_validity)
    attention_task = true;
    nReliabilities = length(unique(real_st.data(1).raw.cue_validity_id));
    
    labels = {'valid', 'neutral', 'invalid'};
    xl = 'cue validity';
    
    colors = flipud([.7 0 0;.6 .6 .6;0 .7 0]);
    
    %confidence only experiment
else
    attention_task = false;
    nReliabilities = length(unique(real_st.data(1).raw.contrast_id));
    
    contrasts = unique(real_st.data(1).raw.contrast);
    strings = strsplit(sprintf('%.1f%% ', contrasts*100), ' ');
    labels = fliplr(strings(1:end-1));
    xl = 'contrast/eccentricity';
    
    hhh = hot;
    colors = hhh(round(linspace(1,40,nReliabilities)),:); % black to orange indicate high to low contrast
end


[edges, centers] = bin_generator(nBins, 'task', task);
if ~marg_over_s
    % tick mark placement
    ori_labels = [-14 -5 -2 0 2 5 14]; % make sure that this only has nBins entries or fewer. also has to have smaller absolute values than centers
    xticklabels = interp1(centers, 1:nBins, ori_labels);
    if symmetrify
        ori_labels = abs(ori_labels);
    end
elseif marg_over_s
    xticklabels = 1:nReliabilities;
end

nDepVars = length(dep_vars);
dep_var_labels = rename_var_labels(dep_vars); % translate from variable names to something other people can understand.


figure
clf

for subject = 1:nSubjects
    real_raw = real_st.data(subject).raw;
    if symmetrify
        real_raw.s = abs(real_raw.s);
    end
    real_stats = indiv_analysis_fcn(real_raw, edges);
    
    if show_fits
        fake_raw = fake_st.data(subject).raw;
        if symmetrify
            fake_raw.s = abs(fake_raw.s);
        end
        fake_stats = indiv_analysis_fcn(fake_raw, edges);
    end
    
    for dep_var = 1:nDepVars
        tight_subplot(nDepVars, nSubjects, dep_var, subject, gutter);
        
        if show_fits
            single_dataset_plot(fake_stats.all, dep_vars{dep_var}, marg_over_s, xticklabels, ylims(dep_var,:), 'fill_instead_of_errorbar', true, ...
                'symmetrify', symmetrify, 'colors', colors, 'linewidth', linewidth, 'errorbarwidth', errorbarwidth, 'ticklength', ticklength)
        end
        
        single_dataset_plot(real_stats.all, dep_vars{dep_var}, marg_over_s, xticklabels, ylims(dep_var,:), 'fill_instead_of_errorbar', false, ...
            'symmetrify', symmetrify, 'colors', colors, 'linewidth', linewidth, 'errorbarwidth', errorbarwidth, 'ticklength', ticklength)

        % title (and maybe legend) for top row
        if dep_var == 1
            title(real_st.data(subject).name)
            if subject == 1
                if ~marg_over_s
                    legend(labels)
                end
            end
            
        % x axis labels for bottom row
        elseif dep_var == nDepVars
            if ~marg_over_s
                if symmetrify
                    xlabel('|s|')
                else
                    xlabel('s')
                end
                set(gca, 'xticklabel', ori_labels)
            elseif marg_over_s
                xlabel(xl)
                set(gca, 'xticklabel', labels)
            end
        end
        
        % y axis labels for left column
        if subject == 1
            ylabel(dep_var_labels{dep_var})
            set(gca, 'yticklabelmode', 'auto')
        end
        
        switch dep_vars{dep_var}
            case {'tf','Chat'}
                plot_halfway_line(.5)
            case 'resp'
                set(gca, 'ydir', 'reverse')
                plot_halfway_line(4.5)
        end

    end
end
end
function plot_halfway_line(y)
plot([-40 40], [y y], '-', 'color', [.5 .5 .5], 'linewidth', 1)
end