x=-20:.01:20;
lw = 5;
%% task a
task(1).c1 = normpdf(x,-4,5);
task(1).c2 = normpdf(x,4,5);

%% task 2
task(2).c1 = normpdf(x,0,3);
task(2).c2 = normpdf(x,0,12);

%%
clf
blue=[0 0 .8];
red=[.8 0 0];
tasks = {'A','B'}
for t = 1:2
    subplot(2,1,t)
    plot(x,task(t).c1,'color',blue,'linewidth',lw);
    hold on
    plot(x,task(t).c2,'color',red,'linewidth',lw);
    set(gca,axes_defaults,'xtick',-20:10:20,'ytick',[],'ylim',[0 .133],'fontname','Helvetica Neue','fontsize',20);
    yl=ylabel(sprintf('$p(s|C)$',tasks{t}),'interpreter','latex');
    set(yl,'fontsize',18);
    set(gcf,'position',[1000 1200 860 298])
    if t==1
        set(gca,'xticklabel','')
    else
        xlabel('stimulus orientation $s$ $(^\circ)$','interpreter','latex')
    end
end
%%
export_fig('taska.pdf')