function st = compile_data_across_tasks(datadir)

strealA = compile_data('datadir',[datadir 'taskA/']);
st=strealA;
strealB = compile_data('datadir',[datadir 'taskB/']);

fields = fieldnames(strealA.data(1).raw);
fields=setdiff(fields,'contrast_values');

for d=1:length(st.data)
    st.data(d).raw.task = [-1*ones(1,length(strealA.data(d).raw.C)) ones(1,length(strealB.data(d).raw.C))]; % -1 is task A, +1 is task B
    for f=1:length(fields)
        st.data(d).raw.(fields{f}) = [strealA.data(d).raw.(fields{f}) strealB.data(d).raw.(fields{f})];
    end
end