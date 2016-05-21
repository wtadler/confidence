function fake_sumstats = fake_group_datasets_and_stats(model, nFakeGroupDatasets, varargin)

% for a list of models and their fits, generate some number of fake group datasets, each of which
% consists of one fake dataset from each subject.
% then save summary statistics for all of the fake group datasets
% use this after having used dataset_generator.

fields = {'tf','resp','g','Chat'};
confInterval = .95;
assignopts(who, varargin);


%%
for m = 1:length(model)
    tasks = submodels_for_analysis(model);
    

    for task = 1:length(tasks)
        trial_types = fieldnames(model(m).extracted(1).fake_datasets.(tasks{task}).sumstats);
        bin_types = setdiff(fieldnames(model(m).extracted(1).fake_datasets.(tasks{task}).sumstats.(trial_types{1})), {'index', 'subindex_by_c'});

        for t = 1:length(trial_types)
            for f = 1:length(fields)
                for b = 1:length(bin_types)
                    hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}) = [];
                                        fake_sumstats.(tasks{task}).(trial_types{t}).(bin_types{b}).std.(fields{f})  = std (hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}), 0, 3);
                    hyperplot_sems.(trial_types{t}).(bin_types{b}).(fields{f}) = [];
                end
            end
        end
        
        for h = 1:nFakeGroupDatasets
            clear hyperplotdata
            nSubjects = length(model(m).extracted);
            for dataset = 1:nSubjects
                subject = randi(nSubjects); % 'randi(nSubjects)' if sampling w replacement (as luigi recommends); 'dataset' if sampling from subjects without replacement 
                nPlotSamples = length(model(m).extracted(subject).fake_datasets.(tasks{task}).dataset);
                hyperplotdata(dataset) = model(m).extracted(subject).fake_datasets.(tasks{task}).dataset(randi(nPlotSamples));
            end
            
            % summarize those fake datasets across subjects
            if mod(h, 50) == 0
                if length(model) == 1
                    fprintf('\nTask %i/%i, Hyperplot %i/%i: Analyzing fake data...', task, length(tasks), h, nFakeGroupDatasets);
                else
                    fprintf('\nModel %i/%i, Task %i/%i, Hyperplot %i/%i: Analyzing fake data...', m, length(model), task, length(tasks), h, nFakeGroupDatasets);
                end
            end
            sumstats = sumstats_fcn(hyperplotdata, 'fields', fields, 'bootstrap', false);
            trial_types = fieldnames(sumstats);

            % save means for that summary
            for t = 1:length(trial_types)
                for f = 1:length(fields)
                    for b = 1:length(bin_types)
                        hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}) = cat(3, hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}), sumstats.(trial_types{t}).(bin_types{b}).mean.(fields{f}));
                        hyperplot_sems.(trial_types{t}).(bin_types{b}).(fields{f})   = cat(3, hyperplot_sems.(trial_types{t}).(bin_types{b}).(fields{f}), sumstats.(trial_types{t}).(bin_types{b}).sem.(fields{f}));
                    end
                end
            end

        end
        
        % mean and std the means over all hyperplots
        for t = 1:length(trial_types)
            for f = 1:length(fields)
                for b = 1:length(bin_types)
                    fake_sumstats.(tasks{task}).(trial_types{t}).(bin_types{b}).mean.(fields{f}) = mean(hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}),    3);
                    fake_sumstats.(tasks{task}).(trial_types{t}).(bin_types{b}).std.(fields{f})  = std (hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}), 0, 3);
                    fake_sumstats.(tasks{task}).(trial_types{t}).(bin_types{b}).mean_sem.(fields{f}) = mean(hyperplot_sems.(trial_types{t}).(bin_types{b}).(fields{f}), 3); % this is very close to std
                    
                    quantiles = quantile(hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}), [.5 - confInterval/2, .5 + confInterval/2], 3);
                    fake_sumstats.(tasks{task}).(trial_types{t}).(bin_types{b}).CI.(fields{f}) = quantiles;
                end
            end
        end
    end
end