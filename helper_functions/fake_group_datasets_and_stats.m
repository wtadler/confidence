function fake_sumstats = fake_group_datasets_and_stats(model, nFakeGroupDatasets, varargin)

% for a list of models and their fits, generate some number of fake group datasets, each of which
% consists of one fake dataset from each subject.
% then save summary statistics for all of the fake group datasets
% use this after having used dataset_generator.

fields = {'tf','resp','g','Chat'};
assignopts(who, varargin);

for m = 1:length(model)
    tasks = submodels_for_analysis(model);
    
    for task = 1:length(tasks)
        for f = 1:length(fields)
            hyperplot_means.(fields{f}) = [];
        end

        for h = 1:nFakeGroupDatasets
            clear hyperplotdata
            for subject = 1:length(model(m).extracted)
                nPlotSamples = length(model(m).extracted(subject).fake_datasets.(tasks{task}).dataset);
                hyperplotdata(subject) = model(m).extracted(subject).fake_datasets.(tasks{task}).dataset(randi(nPlotSamples));
            end
            
            % summarize those fake datasets across subjects
            sumstats = sumstats_fcn(hyperplotdata, 'fields', fields);
            
            % save means for that summary
            for f = 1:length(fields)
                hyperplot_means.(fields{f}) = cat(3, hyperplot_means.(fields{f}), sumstats.all.mean.(fields{f}));
            end

        end
        
        % mean and std the means over all hyperplots
        for f = 1:length(fields)
            fake_sumstats.(tasks{task}).mean.(fields{f}) = mean(hyperplot_means.(fields{f}),    3);
            fake_sumstats.(tasks{task}).std.(fields{f})  = std (hyperplot_means.(fields{f}), 0, 3);
        end
    end
end