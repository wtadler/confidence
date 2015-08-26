%% animated criteria
close all

% VIDEO PARAMETERS

% list.extraplot = {'xvsig','xvsig','dvx','dvx','dvx','dvx','xvsig','xvsig','xvsig','xvsig'};
% list.models =           [3 3 2 2 2 2 2 4 5 6];
% list.xp_region_fill =   [1 1 0 0 0 1 1 1 1 1];
% list.dp_region_fill =   [0 0 0 1 1 1 0 0 0 0];
% list.choice_colors =    [1 0 0 1 0 0 0 0 0 0];
% list.ghost =            [0 0 0 0 0 0 0 0 0 0];

list.extraplot = {'xvsig'};
list.models =           [2];
list.xp_region_fill =   [1];
list.dp_region_fill =   [0];
list.choice_colors =    [0];
list.ghost =            [0];

datetimestr_all = datetimefcn;
set(0,'defaultaxesticklength',[.008 .025]);
for iii=1:length(list.models)
    clearvars -except iii list f datetimestr_all start_t
    model_id = list.models(iii);
    
    close all
    
    f=figure;
    clf
    
    makevideo = 1;
    loopSecs = 7.5;
    fps = 3;
    nSteps=round(loopSecs * fps);
    sss=1:nSteps;
    scalefactor = 2.73; % 2: XGA. 2.5: SXGA. 2.73: SXGA+, 1400x1050.
    shiftfactor = .6;
    model_label = 0;
    blue=[.7 .7 1];
    red =[1 .7 .7];
    darkblue=[0 0 .9];
    darkred =[.9 0 0];
    
    xvsigcontrastlines = 0;
    
    datetimestr = datetimefcn;
    
    load('/Users/will/Google Drive/Ma lab/output/v3_B_fmincon_feb21/COMBINED_v3_B_.mat')
    m = model;
    
    %this is just for marking the text
    sig1=3;
    sig2=12;
    mean_symcat = 4; % RDM symmetrical category mean.
    sig_symcat = 5;  % RDM symmetrical category std.
    
    
    nX = 300;
    x=linspace(-20,20,nX);
    yl=[0 .14];
    xl=[-20 20];
%     colors = [0 0 .4;
%         .17 .17 .6;
%         .33 .33 .8;
%         .5 .5 1;
%         1 .5 .5;
%         .8 .33 .33;
%         .6 .17 .17;
%         .4 0 0];
    colors = [0 .1 .4;
        .1 .3 .6;
        .3 .5 .8;
        .5 .7 .8;
        .9 .7 .6;
        .8 .33 .33;
        .6 .15 .17;
        .4 0 0].^1;
        
    if list.choice_colors(iii)
%         colors = [repmat([.5 .5 1],4,1);repmat([1 .5 .5],4,1)];
          row1 = 3;
          row2 = 6;
          colors = [repmat(colors(row1,:),4,1);
                    repmat(colors(row2,:),4,1)];
    end
    true_contrasts=exp(-4:.5:-1.5);
    p.alpha = .0210;
    p.beta = 2.26;
    p.sigma_0 = 0;
    true_sigs = sqrt(p.sigma_0+p.alpha*true_contrasts.^-p.beta);
    mintruesig=min(true_sigs);
    maxtruesig=max(true_sigs);
    
    contrasts = fliplr(8.^(linspace(-2,-.5,nSteps)));
    mincontrast=min(contrasts);
    maxcontrast=max(contrasts);
    sigs = sqrt(p.sigma_0+p.alpha*contrasts.^-p.beta);
    minsig = min(sigs);
    maxsig = max(sigs);
    
    
    gcf_w = 512; % keep at 512
    gcf_rh = .375*gcf_w; % height in pixels for a one row figure
    
    alpha=.6;
    subject = 2;
    go = true;
    xt=-15:5:15;
    set(gcf,'DefaultAxesXtick',xt,'DefaultAxesXticklabel',[],'DefaultAxesYtick',[],'DefaultAxesYticklabel',[],'Color', [1 1 1],'DefaultAxesLineWidth',1,'position',[264 901 gcf_w gcf_rh],'DefaultAxesColorOrder', colors,'visible','on');
    if ~makevideo
        set(gcf,'renderer','painters')
    else
        set(gcf,'renderer','opengl','visible','off')
    end
    fs = 12; %fontsize
    set(0,'defaultaxesfontsize',fs-1)
    
    set(gcf,'position',[264 901 gcf_w 2*gcf_rh])
    
    extraplot=list.extraplot{iii};
    xp_region_fill = list.xp_region_fill(iii);
    dp_region_fill = list.dp_region_fill(iii);
    
    p = parameter_variable_namer(m(model_id).extracted(subject).best_params, m(model_id).parameter_names,m(model_id));
    
    if any(regexp(m(model_id).name, 'asym'))
        p.b_i = [-100 -1.8 -1.15 -.5 0 .34 .5 .8 100]; % when showing the diff between opt and asym, change the choice bound to something that's not 0.
        d_bounds = flipud(p.b_i(2:end-1)');
    elseif any(regexp(m(model_id).name, 'opt'))
        p.b_i = [0 0.45 0.75 0.95 100];
        d_bounds = [flipud(p.b_i(2:end-1)'); 0; -p.b_i(2:end-1)'];
    elseif any(regexp(m(model_id).name, 'lin')) | any(regexp(m(model_id).name, 'quad'))
        %         p.m_i(5:8) = p.m_i(5:8) + .05;
        %         p.b_i(5:8) = p.b_i(5:8) + 2.2;
    elseif any(regexp(m(model_id).name, 'fixed'))
        p.b_i = [0 .4 1.3 2.6 5.16 6.7 8.9 12.5 Inf];
    end
    
    if any(regexp(m(model_id).name, 'RDM'))
        criteria{model_id} = [80*ones(1,nSteps);
            repmat(d_bounds, 1, nSteps) .* repmat(sigs.^2 + sig_symcat^2,7,1) ./ mean_symcat;
            -80*ones(1,nSteps)];
        
    elseif any(regexp(m(model_id).name, 'opt'))
        k1 = .5*log( (sigs.^2 + sig2^2) ./ (sigs.^2 + sig1^2));
        k2 = (sig2^2 - sig1^2) ./ (2 .* (sigs.^2 + sig1^2) .* (sigs.^2 + sig2^2));
        
        criteria{model_id}=real([zeros(1,nSteps);
            sqrt((repmat(k1,7,1) - repmat(d_bounds, 1, nSteps))./repmat(k2,7,1));
            80*ones(1,nSteps)]);
        
        x=linspace(-40,40,nSteps);
        
        nD=1000;
        [d, pd1, pd2] = deal(zeros(nSteps,nD));
        
        for i = 1:nSteps
            d(i,:) = linspace(-10,k1(i)-.001,nD);
            lambda = sqrt(sig1^2 + sigs(i)^2);
            pd1(i,:) = abs(.5./sqrt(k1(i)*k2(i) - k2(i)*d(i,:))) ./ (sqrt(2*pi)*lambda) .* exp(-(k1(i)-d(i,:))./(2*k2(i)*lambda^2));
            lambda = sqrt(sig2^2 + sigs(i)^2);
            pd2(i,:) = abs(.5./sqrt(k1(i)*k2(i) - k2(i)*d(i,:))) ./ (sqrt(2*pi)*lambda) .* exp(-(k1(i)-d(i,:))./(2*k2(i)*lambda^2));
        end
        
        
    elseif any(regexp(m(model_id).name, 'fixed'))
        criteria{model_id}=[zeros(1,nSteps);
            repmat(p.b_i(2:end-1)',1,nSteps);
            80*ones(1,nSteps)];
    elseif any(regexp(m(model_id).name, '^lin'))
        criteria{model_id}=max(0,[zeros(1,nSteps);
            repmat(p.b_i(2:end-1)',1,nSteps)+p.m_i(2:end-1)'*sigs;
            80*ones(1,nSteps)]);
        
    elseif any(regexp(m(model_id).name, '^quad'))
        criteria{model_id}=max(0,[zeros(1,nSteps);
            repmat(p.b_i(2:end-1)',1,nSteps)+p.m_i(2:end-1)'*sigs.^2;
            80*ones(1,nSteps)]);
    end
    
    subplotcols=20;
    frame = 0;
    start_t = tic;
    
    for step = sss
        clf
        frame=frame+1;
        
        if frame>nSteps
            step
            %step = round(log(frame-slowdownframe)^2+slowdownframe)-nSteps
            step = max(1,round(sat - alp*log(1+exp((sat-step)/alp))));
        end
        
        cur_row=0;
        cur_row=cur_row+1;
        model = m(model_id).name;
        
        conf_levels = 4;
        bin_edges = linspace(1/conf_levels, 1 - 1/conf_levels, conf_levels - 1);
        
        contrast = contrasts(step);
        
        if extraplot
            nModels=2;
        end
        
        subplot(nModels,subplotcols,[subplotcols:subplotcols:subplotcols*nModels])
        scrub2=plot([0 1],[contrast contrast],'k-','linewidth',2);
        hold on
        for i = 1: length(true_contrasts)
            l2 = plot([0 1],[true_contrasts(i) true_contrasts(i)],'color',[.6 .6 .6],'linewidth',1);
        end
        
        set(gca,'yscale','log','ylim',[mincontrast maxcontrast], 'xtick',[],'gridlinestyle','-','zgrid','off','xgrid','off','ztick',[],'ytick',true_contrasts,'yaxislocation','right','yticklabel',{'1.8%','3.0%','5.0%','8.2%','13.5%', '22.3%'},'ticklength',[0 0],'visible','on','box','on')
        
        yy=ylabel('contrast');
        yypos = get(yy,'position');
        set(yy,'fontsize',fs,'rot',-90,'interpreter','latex');
        
        subplot(nModels,subplotcols,[subplotcols*cur_row-(subplotcols-1) : subplotcols*cur_row-1])
        cla
        if list.ghost(iii)
            c1ghost=normpdf(x,0,sig1);
            c2ghost=normpdf(x,0,sig2);
            lc1g=plot(x,c1ghost,'color',blue,'linewidth',2,'linesmoothing','on');
            hold on
            lc2g=plot(x,c2ghost,'color',red,'linewidth',2,'linesmoothing','on');
        end
        
        if any(regexp(m(model_id).name, 'RDM'))
            c1=normpdf(x,-mean_symcat,sqrt(sig_symcat^2 + sigs(step)^2));
            c2=normpdf(x,mean_symcat, sqrt(sig_symcat^2 + sigs(step)^2));
        else
            c1=normpdf(x,0,sqrt(sig1^2 + sigs(step)^2));
            c2=normpdf(x,0,sqrt(sig2^2 + sigs(step)^2));
        end
        lc1=plot(x,c1,'color',darkblue,'linewidth',2,'linesmoothing','on');
        hold on
        lc2=plot(x,c2,'color',darkred,'linewidth',2,'linesmoothing','on');
        ylim(yl);
        xlim(xl);
        
        if model_label
            if extraplot
                t=text(min(xl)-1,min(yl)-.025,sprintf('%s\nmodel',model_names{model_id}));
                set(t,'horizontalalignment','right','fontsize',fs,'interpreter','latex')
            else
                t=text(min(xl)-2,mean(yl),model_names(model_id));
                set(t,'horizontalalignment','right','fontsize',fs,'interpreter','latex')
            end
        end
        
        if xp_region_fill
            for i = 1:8
                if any(regexp(m(model_id).name, 'RDM'))
                    f=fill([criteria{model_id}(i,step) criteria{model_id}(i+1,step) criteria{model_id}(i+1,step) criteria{model_id}(i,step)], [yl(1) yl(1) yl(2) yl(2)],colors(9-i,:));
                else
                    f=fill([criteria{model_id}(i,step) criteria{model_id}(i+1,step) criteria{model_id}(i+1,step) criteria{model_id}(i,step)], [yl(1) yl(1) yl(2) yl(2)],colors(i,:));
                end
                
                set(f,'edgecolor','none', 'facealpha', alpha)
                uistack(f,'top')
                if ~any(regexp(m(model_id).name, 'RDM')) % plot the reverse if not RDM.
                    hold on
                    f=fill([-criteria{model_id}(i+1,step) -criteria{model_id}(i,step) -criteria{model_id}(i,step) -criteria{model_id}(i+1,step)], [yl(1) yl(1) yl(2) yl(2)], colors(i,:));
                    set(f,'edgecolor','none', 'facealpha', alpha)
                end
            end
        end
        if model_id == list.models(end)
            apos=get(gca,'position');
            set(gca,'xticklabel',xt,'ztick',[],'layer','top','position',[apos(1)+shiftfactor*apos(1) apos(2) apos(3)-shiftfactor*apos(1) apos(4)],'box','off')
            xx=xlabel('measurement $x$ $(^\circ)$','interpreter','latex','fontsize',fs);
            %yy=ylabel({'{\color[rgb]{.7 .7 1}$p(x|C=1)$}\hspace{2.5mm}';'{\color[rgb]{1 .7 .7}$p(x|C=2)$}\hspace{2.5mm}'}, 'interpreter','latex','fontsize',fs,'rot',0,'horizontalalignment','right');
            yy = ylabel('$p(x|C=1)$\hspace{2.5mm}','interpreter','latex','color',darkblue, 'interpreter','latex','fontsize',fs,'rot',0,'horizontalalignment','right');
            yypos = get(yy,'position');
            text(yypos(1)+.1,yypos(2)-.012,'$p(x|C=2)$\hspace{2.5mm}','interpreter','latex','color',darkred,'fontsize',fs,'horizontalalignment','right');
        end
        
        if extraplot
            cur_row = cur_row+1;
            subplot(2,subplotcols,[subplotcols*2-(subplotcols-1) : subplotcols*2-1]);
            cla
            hold on
            
            if strcmp(extraplot,'d')
                dxl=[-2.1 2.1];
                dyl=[0 .02];
                d1=plot(d(step,:), pd1(step,:)./sum(pd1(step,:)), '-','color',[0 0 .9],'linewidth',2,'linesmoothing','on');
                d2=plot(d(step,:), pd2(step,:)./sum(pd2(step,:)), '-','color',[.9 0 0],'linewidth',2,'linesmoothing','on');
                d3=plot([d(step,end) d(step,end)], [0 1],'k--','linewidth',1.5,'linesmoothing','on');
                xlim(dxl)
                ylim(dyl)
                
                p.b_i(p.b_i==-Inf)=-100;
                p.b_i(p.b_i==Inf) = 100;
                if dp_region_fill
                    if any(regexp(model,'asym'))
                        for i = 1:8
                            f=fill([p.b_i(i) p.b_i(i+1) p.b_i(i+1) p.b_i(i)],[yl(1) yl(1) yl(2) yl(2)],colors(9-i,:));
                            set(f,'edgecolor','none', 'facealpha', alpha)
                        end
                    else
                        for i = 1:4
                            f1=fill([p.b_i(i) p.b_i(i+1) p.b_i(i+1) p.b_i(i)],[yl(1) yl(1) yl(2) yl(2)],colors(5-i,:));
                            f2=fill([-p.b_i(i) -p.b_i(i+1) -p.b_i(i+1) -p.b_i(i)],[yl(1) yl(1) yl(2) yl(2)],colors(4+i,:));
                            set(f1,'edgecolor','none', 'facealpha', alpha)
                            set(f2,'edgecolor','none', 'facealpha', alpha)
                        end
                    end
                end
                xx=xlabel('$\log{\dfrac{p(x|C=1)}{p(x|C=2)}}$');
                set(xx,'fontsize',fs+5,'interpreter','latex')
                apos = get(gca,'position');
                set(gca,'position',[apos(1)+shiftfactor*apos(1) apos(2) apos(3)-shiftfactor*apos(1) apos(4)],'xtick',-2:1:2,'xticklabel',-2:1:2,'ztick',[],'box','off','visible','on','ydir','normal','ytick',[],'layer','top')%,'visible','off')
                delete(get(gca,'ylabel'))
                
            elseif strcmp(extraplot,'dvx')
                dyl=[-2.1 2.1];
                dd = k1(step) - k2(step)*x.^2;
                xd = plot(x,dd,'k','linewidth',2,'linesmoothing','on');
                hold on
                zeroline = plot(xl,[0 0],'k','linesmoothing','on','linewidth',1);
                xlim(xl);
                ylim(dyl);
                apos = get(gca,'position');
                %aticklength = get(gca,'ticklength')
                set(gca,'position',[apos(1)+shiftfactor*apos(1) apos(2) apos(3)-shiftfactor*apos(1) apos(4)], 'xtick',xt,'xticklabel',xt,'box','off','visible','on','ytick',-2:1:2,'yticklabel',-2:1:2,'ztick',[],'layer','top','xcolor','w')
                for tick = 1 : length(xt)
                    plot([xt(tick) xt(tick)], [-.1 0],'-k','linewidth',1)
                    t=text(xt(tick),-.4,num2str(xt(tick)),'horizontalalignment','center','fontsize',11);
                end
                xx=xlabel('measurement $x$ $(^\circ)$');
                xxpos = get(xx,'position');
                set(xx,'fontsize',fs,'interpreter','latex','position',xxpos,'color','k','position',xxpos+[0 2.1 0])
                %yy = ylabel({'\hspace{10mm}$d$';'$=\log{\frac{p(C=1|x)}{p(C=2|x)}}$'});
                yy = ylabel('$\log{\frac{p(x|C=1)}{p(x|C=2)}}$');
                set(yy,'fontsize',fs+4,'interpreter','latex','rot',0,'horizontalalignment','right','verticalalignment','middle')
                
                p.b_i(p.b_i==-Inf)=-100;
                p.b_i(p.b_i==Inf) = 100;
                if dp_region_fill
                    if any(regexp(model,'asym'))
                        for i = 1:8
                            f=fill([xl(1) xl(2) xl(2) xl(1)],[p.b_i(i) p.b_i(i) p.b_i(i+1) p.b_i(i+1)],colors(9-i,:));
                            set(f,'edgecolor','none', 'facealpha', alpha)
                        end
                    else
                        for i = 1:4
                            f1=fill([xl(1) xl(2) xl(2) xl(1)],[p.b_i(i) p.b_i(i) p.b_i(i+1) p.b_i(i+1) ],colors(5-i,:));
                            f2=fill([xl(1) xl(2) xl(2) xl(1)],[ -p.b_i(i) -p.b_i(i) -p.b_i(i+1) -p.b_i(i+1)],colors(4+i,:));
                            set(f1,'edgecolor','none', 'facealpha', alpha)
                            set(f2,'edgecolor','none', 'facealpha', alpha)
                        end
                    end
                end
                
            elseif strcmp(extraplot,'xvsig')
                for i = 1:8
                    
                    ff = fill([-fliplr(criteria{model_id}(i,:)) -(criteria{model_id}(i+1,:))],[fliplr(sigs) sigs],colors(i,:));
                    set(ff,'edgecolor','none','facealpha',alpha)
                    if ~any(regexp(m(model_id).name, 'RDM'))
                        hold on
                        ff = fill([criteria{model_id}(i,:) fliplr(criteria{model_id}(i+1,:))],[sigs fliplr(sigs)],colors(i,:));
                        set(ff,'edgecolor','none','facealpha',alpha)
                    end
                end
                scrubber=plot(xl,[sigs(step) sigs(step)], 'k-','linewidth',2);
                uistack(scrubber,'top')
                apos=get(gca,'position');
                set(gca,'position',[apos(1)+shiftfactor*apos(1) apos(2) apos(3)-shiftfactor*apos(1) apos(4)],'visible','on','ydir','normal','ygrid','on','ycolor',[0 0 0],'xgrid','off','zgrid','off','xticklabel',xt,'xtick',xt,'ztick',[],'gridlinestyle','-','ytick',0:2:20,'yticklabel',0:2:20,'xlim',xl,'ylim',[minsig maxsig],'box','off','layer','top')
                
                %if iii>11
                %    set(gca,'ydir','normal')
                %end
                
                grid off
                if xvsigcontrastlines
                    for i = 1: length(true_sigs)
                        l1 = plot(xl,[true_sigs(i) true_sigs(i)],'color','k','linewidth',1);
                        uistack(l1,'top')
                    end
                end
                uistack(scrubber,'up',100)
                
                xx=xlabel('measurement $x$ $(^\circ)$');
                yy=ylabel('noise level $\sigma$ $(^\circ)$\hspace{0mm}');
                if step == 1 | step ==2
                    xxpos2 = get(xx,'position');
                end
                set(xx,'fontsize',fs,'position',xxpos2,'interpreter','latex')
                
                set(yy,'fontsize',fs,'rot',0,'interpreter','latex','horizontalalignment','right','verticalalignment','middle')
                
            elseif strcmp(extraplot,'empty')
                set(gca,'visible','off')
            end
        end
        
        if makevideo
            tmpdir = ['~/Sandbox/' datetimestr];
            if ~isdir(tmpdir)
                mkdir(tmpdir);
            end
            cd(tmpdir)
            im=export_fig(sprintf('%.5i.png',frame),sprintf('-m%g',scalefactor),'-nocrop');
            
            fprintf('~%.2f mins remaining\n',((toc(start_t)/(frame))*length(sss) - toc(start_t))/60); % better way to do this.
        end
        pause(.000000001)
        %return
        
    end
    if makevideo
        savedir = ['~/Google Drive/Will - Confidence/Analysis/' datetimestr_all '/']
        if ~isdir(savedir)
            mkdir(savedir)
        end
        extensions={'mp4'};%,'wmv'};
        codecs = {'x264'};%,'wmv'};
        for vid=1:length(extensions);
            filename=sprintf('%i%s.%s',iii,m(model_id).name(1:5),extensions{vid});
            imdim=size(im);
            ffmpegtranscode('%5d.png',filename,'InputFrameRate',fps,'AudioCodec','none','x264Tune','animation','DeleteSource','off','WMVBitRate',3000,'VideoCodec',codecs{vid},'VideoCrop',[rem(imdim(2),2) rem(imdim(1),2) 0 0])
            movefile(filename,[savedir filename]);
        end
        %rmdir(tmpdir,'s')
    end
end

return


%% static criteria plots
clear all
cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations')
conf_levels = 4;
bin_edges = linspace(1/conf_levels, 1 - 1/conf_levels, conf_levels - 1);

%load('200opts_hpc_real_opt_optconf_d_noise') % opt_d_noise_conf (to demonstrate equipartition)
%load('1000opts asym bounds d noise conf') % asym bounds
%load('400opts_lin2_conf')
%load('2000opts_quad2_conf')
%load('5models.mat')
    %gen.opt=m;
%load('20140918_121052')
%load('newdata.mat')
%    gen.opt=model([4 3 5 1 2]);
%load('v2_no_noise.mat')
%    gen.opt = model([3 1 2])
xl=[-20 20];
load('/Users/will/Google Drive/Ma lab/output/v3_B_fmincon_feb21/COMBINED_v3_B_.mat')
    gen.opt=model
models = 4%1:length(gen.opt);
nModels = 5%length(models)
figure
%set(gcf,'position',[73 -899 2808 1705])
set(gcf,'position',[53 647 1274 859],'color',[1 1 1])
nSubjects = length(gen.opt(1).extracted);
subjects = 1: nSubjects;
%nSubjects = 1;
%subjects = 2;
fa = 1; %face alpha
nSteps = 100;
% true_contrasts=exp(-4:.5:-1.5);

true_contrasts = exp(linspace(-5.5,-2,6));

space='sig'; %'logcontrast' or 'sig'
task = 'A';
for modelno = models
    for subject = subjects
        
        subplot(nModels,nSubjects, (modelno-1)*nSubjects+subject)
        %
        p = gen.opt(modelno).extracted(subject).best_params;
        p = parameter_variable_namer(p,gen.opt(modelno).parameter_names,gen.opt(modelno));
%         p.b_i = [-Inf -7 -5.16 -3 0 3 5.16 7 Inf]; %lin
        p.m_i = [0 -.7 -.35 -.2 0 .2 .35 .7 Inf];
        
%         if strcmp(task,'A')
%             p.b_i = [-5 -4 -3 -2 -1 0 1 2 3 4 5];
%         else
            %p.b_i = [-100 -.5 -.3 -.05 0 .05 .3 .5 100]; %bayesian
            p.b_i = [-100 -1.5 -.6 -.3 0 .3 .6 1.5 100]; %bayesian
%             p.b_i = [-100 -.8 -.6 -.4 0 .4 .6 .8 100]; %bayesian

%         end
        unique_sigs = params2sig(p,'contrasts',true_contrasts);
        minsig = min(unique_sigs);
        maxsig = 15%max(unique_sigs);
        
        switch space
            case 'sig'
                sigs = linspace(0, maxsig+1, nSteps);
                y=sigs;
            case 'logcontrast'
                contrasts = exp(linspace(-4.2,-1.3,nSteps));
                sigs = sqrt(p(3)+p(1)*contrasts.^-p(2));
                y=contrasts;
        end
        

        sig1 = 3;
        sig2 = 12;
                
        if any(regexp(gen.opt(modelno).name, 'opt'))
            if strcmp(task,'A')
                d_bounds = fliplr(p.b_i(2:end-1))';
                criteria = [80*ones(1,nSteps);
                    repmat(d_bounds, 1, nSteps) .* repmat(sigs.^2 + category_params.sigma_s^2,7,1) ./ category_params.mu_2;
                    -80*ones(1,nSteps)];
            else
                d_bounds = fliplr(p.b_i(2:end-1))';
                k1 = .5*log( (sigs.^2 + sig2^2) ./ (sigs.^2 + sig1^2));
                k2 = (sig2^2 - sig1^2) ./ (2 .* (sigs.^2 + sig1^2) .* (sigs.^2 + sig2^2));
                
                criteria=real([zeros(1,nSteps);
                    sqrt((repmat(k1,7,1) - repmat(d_bounds, 1, nSteps))./repmat(k2,7,1));
                    80*ones(1,nSteps)]);
            end
                
                

        elseif any(regexp(gen.opt(modelno).name, 'fixed'))
            if strcmp(task,'A')
            criteria=[-80*ones(1,nSteps);
                repmat(p.b_i(2:8)',1,nSteps);
                80*ones(1,nSteps)];
            else
                criteria=[zeros(1,nSteps);
                repmat(p.b_i(2:8)',1,nSteps);
                80*ones(1,nSteps)];
            end
        elseif any(regexp(gen.opt(modelno).name, 'lin'))
            if strcmp(task,'A')
            criteria=[-80*ones(1,nSteps);
                repmat(p.b_i(2:8)',1,nSteps)+p.m_i(2:8)'*sigs;
                80*ones(1,nSteps)];
            else
                criteria=max(0,[zeros(1,nSteps);
                repmat(p.b_i(2:8)',1,nSteps)+p.m_i(2:8)'*sigs;
                80*ones(1,nSteps)]);
            end
        elseif any(regexp(gen.opt(modelno).name, 'quad'))
            criteria=[zeros(1,nSteps);
                repmat(p(4:10),1,nSteps)+p(11:17)*sigs.^2;
                80*ones(1,nSteps)];
        end
        
        colors = [0 .1 .4;
        .1 .3 .6;
        .3 .5 .8;
        .5 .7 .8;
        .9 .7 .6;
        .8 .33 .33;
        .6 .15 .17;
        .4 0 0].^1;
        set(0,'DefaultAxesColorOrder', colors);
        for i = 1:8
            f = fill([-fliplr(criteria(i,:)) -criteria(i+1,:)], [fliplr(y) y], colors(i,:)); % one half
            set(f,'edgecolor','none','facealpha',fa)
            if strcmp(task,'B')
                hold on
                ff = fill([criteria(i,:) fliplr(criteria(i+1,:))],[y fliplr(y)],colors(i,:)); % other half % comment this out for symmetric task
                set(ff,'edgecolor','none','facealpha',fa)
            end
        end
        
        
        ylim([0 max(y)])
        xlim(xl);
        
        %indicate useful range with filled gray square. not using anymore, since
        %contrast lines are better.
        %f=fill([minsig maxsig maxsig minsig], [0 0 yl(2) yl(2)], 'k');
        %set(f,'facealpha',.2,'edgecolor','none')
        
        switch space
            case 'sig'
               % plot sigma line for each contrast
                % for c = 1 : length(true_contrasts)
               %     sig_c=sqrt(p(3)+p(1)*true_contrasts(c)^-p(2));
               %     plot([sig_c sig_c], yl,'k-')
               % end
                %if modelno == nModels
%                 xlabel('\sigma')
                %end
            case 'logcontrast'
                for c = 1 : length(true_contrasts)
                    plot([true_contrasts(c) true_contrasts(c)], yl, 'k-')
                end
                set(gca,'xscale','log','xdir','reverse','xtick',[])
                if modelno == nModels
                    xlabel('contrast')
                    set(gca,'xtick',true_contrasts,'xticklabel',{'1.8%','3.0%','5.0%','8.2%','13.5%','22.3%'})
                end
        end
        
        % if subject == 1
        %ylabel('sensory noise $\sigma$','interpreter','latex')
        ylabel('')
        %end
        set(gca,'box','off','fontsize',30,'fontname','Helvetica Neue','tickdir','out','ticklength',[.015 .015],'ytick',[0 5 10 15],'yticklabel','')
        set(gcf,'position',[584 290 876 327])
    end
end
%set(gcf,'position',[163 901 879 805])
lh=legend('$\hat{C}=-1,\gamma=4$','$\hat{C}=-1,\gamma=3$','$\hat{C}=-1,\gamma=2$','$\hat{C}=-1,\gamma=1$','$\hat{C}=1,\gamma=1$','$\hat{C}=1,\gamma=2$','$\hat{C}=1,\gamma=3$','$\hat{C}=1,\gamma=4$');
set(lh,'interpreter','latex','fontsize',10,'orientation','horizontal','location','southeast')
pos = get(lh,'position');
set(lh,'position',pos+[pos(1)+.15 pos(4) 0 0]);
legend('boxoff')
%'position',pos+[pos(3) 0 0 0],
delete(lh)