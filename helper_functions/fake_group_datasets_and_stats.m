function fake_sumstats = fake_group_datasets_and_stats(model, nFakeGroupDatasets, varargin)

% for a list of models and their fits, generate some number of fake group datasets, each of which
% consists of one fake dataset from each subject.
% then save summary statistics for all of the fake group datasets
% use this after having used dataset_generator.

fields = {'tf','resp','g','Chat'};
assignopts(who, varargin);


%%
for m = 1:length(model)
    tasks = submodels_for_analysis(model);
    

    for task = 1:length(tasks)
        trial_types = fieldnames(model(m).extracted(1).fake_datasets.(tasks{task}).sumstats);
        bin_types = setdiff(fieldnames(model(m).extracted(1).fake_datasets.(tasks{task}).sumstats.all), {'index', 'subindex_by_c'});

        for t = 1:length(trial_types)
            for f = 1:length(fields)
                for b = 1:length(bin_types)
                    hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}) = [];
                end
            end
        end
        
        for h = 1:nFakeGroupDatasets
            clear hyperplotdata
            for subject = 1:length(model(m).extracted)
                nPlotSamples = length(model(m).extracted(subject).fake_datasets.(tasks{task}).dataset);
                hyperplotdata(subject) = model(m).extracted(subject).fake_datasets.(tasks{task}).dataset(randi(nPlotSamples));
            end
            
            % summarize those fake datasets across subjects
            sumstats = sumstats_fcn(hyperplotdata, 'fields', fields);
            trial_types = fieldnames(sumstats);

            % save means for that summary
            for t = 1:length(trial_types)
                for f = 1:length(fields)
                    for b = 1:length(bin_types)
                        hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}) = cat(3, hyperplot_means.(trial_types{t}).(bin_types{b}).(fields{f}), sumstats.(trial_types{t}).(bin_types{b}).mean.(fields{f}));
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
                end
            end
        end
    end
end