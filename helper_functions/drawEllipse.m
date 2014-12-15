function im = drawEllipse(area, eccentricity, rot, fgcol, bgcol)
% makes an ellipse with given area and eccentricity.
% major axis is on the vertical with rot = 0. rotates clockwise by rot.
% fgcol is the color of the ellipse.
% bgcol is the color of the background.

d1 = round (sqrt( (4 * area * sqrt(1 - eccentricity^2)) / pi)); % length of minor axis
d2 = round (4 * area / (d1 * pi)); % length of major axis

nPixels = d1 * d2;

% makes multiple ellipse sizes, with color steps between fig and bg.
% should work pretty well for any size ellipse.
nSizes  = 100;
minsize = 1;
maxsize = minsize + 10 ^ (.5 - .5 * log10(nPixels));
sizes   = linspace(maxsize, minsize, nSizes);
colors  = linspace(bgcol, fgcol, nSizes);

rot = -rot/180*pi;

% draw ellipse
im = ones(2*d2,2*d2)*bgcol;
minX = -d2;
maxX = minX + 2*d2 - 1; 
[X Y] = meshgrid(minX:maxX,minX:maxX);
X_new = X * cos(rot) - Y * sin(rot);
Y = X * sin(rot) + Y * cos(rot);
X = X_new;

% draws several circles on top of each other, of decreasing size.
% color goes from bg to fg.
for s = 1:nSizes;
idx = (X.^2/((sizes(s)*d1)/2)^2 + Y.^2/((sizes(s)*d2)/2)^2)<1;
im(idx) = colors(s);
end

% crop. orders of magnitude faster than the method before, for big ellipses.
im = im(:, any(diff(im)));
im = im(any(diff(im')), :);

%imshow(im)
%colormap(gray);