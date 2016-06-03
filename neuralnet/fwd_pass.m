function [a, z] = fwd_pass(h,W,b,L,ftype)
bias_output = true;
a        = cell(L,1);
z        = cell(L,1);
a{1}     = h;
%%
for i = 2:L

    z{i} = W{i} * a{i-1} + b{i};
    if i == L % output layer
        if bias_output
            a{i} = sigma_func(W{i} * a{i-1} + b{i},'sigm');
        else
            a{i} = sigma_func(W{i} * a{i-1},'sigm');
        end
    else % hidden layer(s)
        a{i} = sigma_func(W{i} * a{i-1} + b{i},ftype);
    end
end