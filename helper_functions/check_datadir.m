function datadir = check_datadir(datadir)

if regexp(datadir, '/v3')
    A = [datadir '/taskA'];
    B = [datadir '/taskB'];
    datadir.A = A;
else
    B = datadir;
end


datadir.B = B;