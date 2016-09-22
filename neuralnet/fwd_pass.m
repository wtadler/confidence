function [a, z] = fwd_pass(R,W,b,nLayers,ftype)
bias_output = true;
a        = cell(nLayers,1);
z        = cell(nLayers,1);
a{1}     = R;
%%
for layer = 2:nLayers
    
    z{layer} = W{layer} * a{layer-1} + b{layer};
    if layer == nLayers % output layer
        if bias_output
            a{layer} = sigma_func(W{layer} * a{layer-1} + b{layer},'sigm');
        else
            a{layer} = sigma_func(W{layer} * a{layer-1},'sigm');
        end
    else % hidden layer(s)
        a{layer} = sigma_func(W{layer} * a{layer-1} + b{layer},ftype);
    end
end