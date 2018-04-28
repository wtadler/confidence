function xm = qargmax1(x,v,dim)
%QARGMAX1  Quick numerical argmax via 1-D interpolation.
%   XM = QARGMAX1(X,V) estimates the location XM of the maximum of the 
%   underlying function V=F(X), obtained via quadratic 1-D interpolation.
%
%   QTRAPZ is potentially much faster than TRAPZ with *large* matrices.
%
%   See also TRAPZ.

% By default compute maximum along the first non-singleton dimension
if nargin < 3; dim = find(size(v)>1,1); end    

% Compute dimensions of input matrix    
if isvector(v); n = 1; else n = ndims(v); end

assert(length(x) == size(v,dim),'BMP:qargmax1:dimMismatch','X must have the same length as the DIM-th dimension of V.');

[~, index] = max(v,[],dim);

switch n
    case {1,2}
        x = [2*x(1)-x(2);x(:); 2*x(end)-x(end-1)];
        index = index + 1;
        x3 = x(index+1);
        x2 = x(index);
        x1 = x(index-1);

        switch dim
            case 1
                v = [v(1,:);v;v(end,:)];
                y3 = v(sub2ind(size(v),min(index+1,length(x)),1:size(v,2)))';
                y2 = v(sub2ind(size(v),index,1:size(v,2)))';
                y1 = v(sub2ind(size(v),max(index-1,1),1:size(v,2)))';
            case 2
                y3 = v(sub2ind(size(v),1:size(v,1),min(index+1,length(x))))';
                y2 = v(sub2ind(size(v),1:size(v,1),index,1))';
                y1 = v(sub2ind(size(v),1:size(v,1),max(index-1,1)))';
        end

        iswap = x1 == 0;
        tempx = x3(iswap); tempy = y3(iswap);
        x3(iswap) = x1(iswap); y3(iswap) = y1(iswap);
        x1(iswap) = tempx; y1(iswap) = tempy;

        iswap = x2 == 0;
        tempx = x3(iswap); tempy = y3(iswap);
        x3(iswap) = x2(iswap); y3(iswap) = y2(iswap);
        x2(iswap) = tempx; y2(iswap) = tempy;
                
        A = 1./(x2-x2.^2./x1);
        z2 = (x2./x1).^2;
        c = (y3 - (y1./x1.^2- A.*y2./x1 + A.*y1.*z2./x1).*x3.^2 - A.*(y2-y1.*z2).*x3) ./ (1 - x3.^2./x1.^2 - A.*x3 + A.*z2.*x3 + A.*x3.^2./x1 - A.*z2.*x3.^2./x1);
        b = A.*(y2 - c - (y1 - c).*(x2./x1).^2);
        a = (y1-c)./x1.^2 - b./x1;
        xm = -0.5*b./a;
        if dim == 1; xm = xm(:)'; end
        
    case 3
        
        if dim == 2; v = permute(v,[2 1 3]); elseif dim == 3; v = permute(v,[3 1 2]); end
        
        vsize = size(v);
        v = reshape(v,[vsize(1),vsize(2)*vsize(3)]);
        xm = qargmax1(x,v);
        xm = reshape(xm,[1,vsize(2),vsize(3)]);
                
        if dim == 2; xm = ipermute(xm,[2 1 3]); elseif dim == 3; xm = ipermute(xm,[3 1 2]); end
        
end
end

function index = mysub2ind(nrows, rows, cols)
    index = rows + (cols-1) * nrows;
end