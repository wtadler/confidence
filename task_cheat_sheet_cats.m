x=-20:.01:20;
lw = 3;
%% task a
c1 = normpdf(x,-4,5);
c2 = normpdf(x,4,5);

%% task 2
c1 = normpdf(x,0,3);
c2 = normpdf(x,0,12);

%%
clf
plot(x,c1,'b','linewidth',lw);
hold on
plot(x,c2,'r','linewidth',lw);
set(gca,axes_defaults,'xtick',[],'ytick',[],'ylim',[0 .133]);
yl=ylabel('Frequency');
set(yl,'fontsize',18);
set(gcf,'position',[1000 1200 860 298])
export_fig('taska.pdf')