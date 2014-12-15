function im = drawBar(d1,d2,rot,fgcol,bgcol)

rot = -rot-90;  

d1 = round(d1);
d2 = round(d2);

rot = -rot/180*pi;

% make sure that d1 is the minor axis
if (d1>d2)
    d3=d1;
    d1=d2;
    d2=d3;
end
% 
nPixels = d1 * d2;
% makes multiple line sizes with color steps between fig and bg. should
% work well for any size line
nSizes = 1;
minsize = 1;
maxsize = minsize + 10 ^ (.5 - .5 * log10(nPixels));
sizes = linspace(maxsize, minsize, nSizes);
%sizes = linspace(minsize, maxsize, nSizes);
colors = linspace(bgcol, fgcol, nSizes);

% draw bar
im = ones(2*d2,2*d2)*bgcol;
minX = -d2;
maxX = minX + 2*d2 - 1; 
[X, Y] = meshgrid(minX:maxX,minX:maxX);

X_new = X * cos(rot) - Y * sin(rot);
Y = X * sin(rot) + Y * cos(rot);
X = X_new;

for s = 1:nSizes
    %idx = (X > -sizes(s)*d1/2 & X < sizes(s)*d1/2) & (Y > -sizes(s)*d2/2 & Y < sizes(s)*d2/2); %???
    idx = (X.^2/((sizes(s)*d1)/2)^2 + Y.^2/((sizes(s)*d2)/2)^2)<1;
    im(idx) = colors(s);
end
factor1 = 1.025; % originally 1.012
factor2 = 1.005; % 1.025
factor3 = 1.01; % 1.03

% idx = (X > -d1/2 & X < d1/2) & (Y > -d2/2 & Y < d2/2);
% idx_low = (X > -factor1*d1/2 & X < factor1*d1/2) & (Y > -factor2*d2/2 & Y < factor2*d2/2);
% idx_super_low = (X > -factor2*d1/2 & X < factor2*d1/2) & (Y > -factor3*d2/2 & Y < factor3*d2/2);

im = im(:, any(diff(im)));
im = im(any(diff(im')), :);

imshow(im);
colormap(gray);
