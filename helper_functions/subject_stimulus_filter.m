function models_out = subject_stimulus_filter(models_in, stimulus)


names.ellipse = {'ak', 'amk', 'hs', 'kl', 'ohm'};
names.grate = {'ek', 'mb', 'sp', 'sr', 'tt', 'vc'};

models_out = models_in;

for m = 1:length(models_out)
    idx = [];
    for s = 1:length(models_out(m).extracted);
        if any(strcmp(models_out(m).extracted(s).name, names.(stimulus)))
            idx = [idx s];
        end
    end
    models_out(m).extracted = models_out(m).extracted(idx);
end