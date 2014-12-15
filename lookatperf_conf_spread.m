%streal=compile_data('datadir','~/Ma lab/repos/qamar confidence/data/v2');
for subject = 1:length(streal.data)
perf(subject)= mean(streal.data(subject).raw.tf);
prop_lowest_conf(subject) = sum(streal.data(subject).raw.g==1)/length(streal.data(subject).raw.C);
end