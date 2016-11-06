function [a, z] = fwd_pass(R,W,b,nLayers,hidden_unit_type)
bias_output = true;
a        = cell(nLayers,1);
z        = cell(nLayers,1);
a{1}     = R;
%%
for layer = 2:nLayers
    if ~bias_output; b{layer} = 0; end
    
    z{layer} = W{layer} * a{layer-1} + b{layer};
    
    if layer == nLayers % output layer
        a{layer} = sigma_func(z{layer}, 'sigm');
    else % hidden layer(s)
        a{layer} = sigma_func(z{layer}, hidden_unit_type);
    end
end