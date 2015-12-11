function datadir = check_datadir(datadir_path)

datadir = struct;

if regexp(datadir_path, '/v3')
    A = [datadir_path '/taskA'];
    B = [datadir_path '/taskB'];
    datadir.A = A;
else
    B = datadir_path;
end


datadir.B = B;