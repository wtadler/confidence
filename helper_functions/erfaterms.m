function y = erfaterms(x,iter)
[xx,ii] = meshgrid(x,1:iter);
y = 1 + sum(((-1).^ii) .* (ii*2-1) ./ ((2*xx.^2).^ii));
