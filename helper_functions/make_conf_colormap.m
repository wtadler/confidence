function map = make_conf_colormap(intermediate_steps)
intermediate_steps = max(0,round(intermediate_steps));
colors = [0 .1 .4;
    .1 .3 .6;
    .3 .5 .8;
    .5 .7 .8;
    .9 .7 .6;
    .8 .33 .33;
    .6 .15 .17;
    .4 0 0];

map = [];%zeros(8+intermediate_steps*7,3);
for c = 1:7
    r = linspace(colors(c,1),colors(c+1,1),intermediate_steps+2);
    g = linspace(colors(c,2),colors(c+1,2),intermediate_steps+2);
    b = linspace(colors(c,3),colors(c+1,3),intermediate_steps+2);
    
    rgb = [r' g' b'];
    
    map = [map; rgb(1:end-1,:)];
end
map = [map; colors(end,:)];
return
%%
imagesc([1:.01:8])
colormap(make_conf_colormap(255))