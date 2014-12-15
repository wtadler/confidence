clear all
close all
tic

cdsandbox
%cd('~/Ma lab/Confidence theory/theory plots/')
%cd('~/Ma lab/Confidence theory/theory plots/kepecs/')
%cd('~/Ma lab/Confidence theory/theory plots/new/')

% PARAMETERS

set(0,'DefaultLineLineWidth',1)
%set(0,'DefaultLineLineWidth','remove')
tasktype='qamar';

% analytical calculations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% qamar
switch tasktype
    
    case 'qamar'      
        % sig params
        %sig = [.5 1 2 4 8 16];
        sig = [1 2 4];
        sig1 = 3;
        sig2 = 12;
        
        % x axis definition
        o_boundary=20;
        dx = .1;
        o = -o_boundary:dx:o_boundary; % o stands for orientation or odor!
        
        % pre-calculate analytical stats
        % stimulus distribution p(s|C)
        psC1=normpdf(o,0,sig1);
        psC2=normpdf(o,0,sig2);
        
        % by Bayes rule, p(C|s) = p(s|C)*p(C)/p(s)
        pC1s = psC1./(psC1+psC2);
        pC2s = psC2./(psC1+psC2);
        
        likelihood1=exp((1 ./ (sig1^2 + sig.^2)./sqrt(2*pi*(sig1^2 + sig.^2)))'*(-.5*(o.^2)));
        likelihood2=exp((1 ./ (sig2^2 + sig.^2)./sqrt(2*pi*(sig2^2 + sig.^2)))'*(-.5*(o.^2)));
        
        k1 = .5*log( (sig.^2 + sig2^2) ./ (sig.^2 + sig1^2));
        k2 = (sig2^2 - sig1^2) ./ (2 .* (sig.^2 + sig1^2) .* (sig.^2 + sig2^2));
        k=sqrt(k1./k2);
        k2x2 = k2' * o.^2;
        d = repmat(k1',1,size(k2x2,2)) - k2x2;
        
        posterior1 = 1 ./ (1 + exp(-d));
        posterior2 = 1 ./ (1 + exp(d));
        
        g = abs(posterior1 - .5)+.5; % g is confidence
        
        % p(Chat = 1, 2)
        s = o;
        kminuss = repmat(k',1,size(s,2)) - repmat(s,length(k),1); % = k - s
        kpluss = repmat(k',1,size(s,2)) + repmat(s,length(k),1); % = k + s
        pChat1 = .5 * (erf(kminuss./(sqrt(2)*repmat(sig',1,size(kminuss,2)))) + erf(kpluss./(sqrt(2)*repmat(sig',1,size(kpluss,2)))));
        pChat2 = 1-pChat1;
        
        % expected value of Chat
        expChat=2-pChat1;
        
        
        
    case 'kepecs' 
        % sig params
        sig = [.1 .2 .4 .8 1.6 3.2];
        %sig = [.2 .4 .8];
        sigprior = 1.3;
        
        % x axis definition
        a_length = 200; % must be even
        a = linspace(.04,.96,a_length);
        dx = 1/length(a);
        o = log(a./(1-a)); % [A] is remapped onto o, which is in infinite space
        o_boundary = max(o);
        
        % pre-calculate analytical stats
        % stimulus distribution p(s|C)
        psC1 = [2 * normpdf(o(1:a_length/2),0,sigprior) zeros(1,a_length/2)];
        psC2 = [zeros(1,a_length/2), 2 * normpdf(o(1+a_length/2:end),0,sigprior)];
        
        % by Bayes rule, p(C|s) = p(s|C)*p(C)/p(s)
        pC1s = [ones(1,a_length/2),zeros(1,a_length/2)];
        pC2s = [zeros(1,a_length/2),ones(1,a_length/2)];
        
        [likelihood1,likelihood2,d]=deal(zeros(length(sig),a_length));
        
        for i=1:length(sig);
            mu = (o.* sigprior^2)./(sig(i)^2 + sigprior^2);
            k = sig(i) .* sigprior ./ sqrt(sig(i)^2 + sigprior^2);
            likelihood1(i,:) = 2*normpdf(o,0,sqrt(sig(i)^2 + sigprior^2)) .* normcdf(0,mu,k);
            likelihood2(i,:) = 2*normpdf(o,0,sqrt(sig(i)^2 + sigprior^2)) .* normcdf(0,-mu,k);
            %d(i,:) = log(likelihood1(i,:)./likelihood2(i,:));
            %d2(i,:) = log(-1 + (2./(1+erf(mu./sqrt(2*k^2))))); %alternate d.
            %there's some numerical overflow happening here that doesn't happen
            %above.
            %d2(i,:) = log(-1 + (1./(normcdf(mu./k)))); % This one is weird
            %too. weird puzzle with emin
            %d2(i,:) = log(erfc(mu./sqrt(2*k^2)))-log(erfc(-mu./sqrt(2*k^2))); %this alternate works
            %posterior1(i,:) = .5 - .5*erf(mu./sqrt(2*k^2)); % a valid way to
            %do the posterior
            
            % symmetric with overlap a
            a = .4;
            denom = sig(i) * sqrt(2);
            d(i,:) = log( (erf((o-a)/denom) - erf((o+1-a)/denom)) ./ (erf((o-1+a)/denom) - erf((o+a)/denom)));
        end
        
        posterior1 = 1 ./ (1 + exp(-d));
        posterior2 = 1 ./ (1 + exp(d));
        
        g = abs(posterior1 - .5) + .5;
        
        pChat1 = .5 * erfc( repmat(o,length(sig),1) ./ (sqrt(2)*repmat(sig',1,size(o,2))));
        pChat2 = .5 * erfc(-repmat(o,length(sig),1) ./ (sqrt(2)*repmat(sig',1,size(o,2))));
        
        % expected value of Chat
        expChat=2-pChat1;
        
        
end


%% miscellaneous side plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch tasktype
    case 'qamar'
        s=9;
        plot(o,normpdf(o,s,sig(2)),'--','Color',[0 .5 0])
        hold on
        plot(o,normpdf(o,s,sig(6)),'r--')
        title('p(x|s)')
        ylabel('Degree of belief')
        xlabel('x measurement (°)')
        xlim([-o_boundary o_boundary])
        legend(strcat('\sigma=',num2str(sig(2))),strcat('\sigma=',num2str(sig(6))))
        export_and_reset('p(x|s).pdf')
        
        % plot area under curve
        picksig = 4; % pick an index of the above sigs
        s = 9; % 9 degree stimulus
        y = normpdf(o,s,sig(picksig));
        plot(o,y)
        hold on
        xbound = o(abs(o) < k(picksig));
        ybound = y(abs(o) < k(picksig));
        area(xbound,ybound,'FaceColor','r','EdgeColor','none')
        plot([s s],ylim,':','Color',[0 .5 0])
        plot([k(picksig) k(picksig)],ylim,'k--')
        plot([-k(picksig) -k(picksig)],ylim,'k--')
        plot([0 0],ylim,'k')
        xlabel('x measurement (°)')
        legend('$p(x \mid s,\sigma)$','$p(\hat{C} = 1 \mid s,\sigma)$','s','$\pm k$')
        h=legend;
        set(h, 'interpreter', 'latex')
        title(strcat('s=',num2str(s),'°, ','\sigma=',num2str(sig(picksig))))
        xlim([-o_boundary o_boundary])
        export_and_reset('p(x|s)_2.pdf')
        
        % plot k
        plot(sig,k,'ko-')
        title('k = sqrt(k1/k2)')
        xlabel('sigma')
        ylim([0 18])
        export_and_reset('k.pdf')
        
        
    case 'kepecs'
        plot(a,o)
        xlabel('[A]')
        ylabel('s')
        title('Mapping [A] onto s')
        export_and_reset('s_from_[A].pdf')
        
end % end kepecs



%% joint analytical plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot p(C)
bar(1,.5,'facecolor','b')
hold on
bar(2,.5,'facecolor','r')
ylim([0 1])
set(gca,'xtick',[1 2])
set(gca,'ytick',[])
xlabel('C')
title('p(C)')
export_and_reset('p(C).pdf')

%% plot p(s|C)
plot(o,psC1)
hold on
plot(o,psC2,'r')
xlim([-o_boundary o_boundary])
set(gca, 'YTick', []);
xlabel('s')
legend('C=1','C=2')
title('p(s|C)')
export_and_reset('p(s|C).pdf')

%% plot p(C|S)
plot(o,pC1s)
hold on
plot(o,pC2s,'r')
xlim([-o_boundary o_boundary])
ylim([-.1 1.1])
set(gca, 'YTick', []);
xlabel ('s')
legend('C=1','C=2')
title('p(C|s)')
export_and_reset('p(C|s).pdf')

%% plot likelihoods
plot(o,likelihood1)
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('x')
title('p(x|C=1)')
legend_and_export('p(x|C=1).pdf',sig)

plot(o,likelihood2)
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('x')
title('p(x|C=2)')
legend_and_export('p(x|C=2).pdf',sig)

%% plot d
plot(o,d)
xlim([-o_boundary o_boundary])
xlabel('x')
title('d')
switch tasktype
    case 'qamar'
        ylim([-1.5 1.5])
        hold on
        plot(k,zeros(1,length(sig)),'bo')
        plot(-k,zeros(1,length(sig)),'bo')
        legend_and_export('d.pdf',sig,'plottingk',true)
    
    case 'kepecs'
        %legend_and_export('d.pdf',sig)
end

%% plot posteriors
plot(o,posterior1)
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('x')
title('p(C=1|x)')
legend_and_export('p(C=1|x).pdf',sig)

plot(o,posterior2)
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('x')
title('p(C=2|x)')
legend_and_export('p(C=2|x).pdf',sig)

%% plot confidence as function of x
plot(o,g)
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('x')
title('$\gamma(x)$','interpreter','latex')
legend_and_export('confidence(x).pdf',sig)

%% plot p(Chat | s)
plot(o,pChat1)
xlim([-o_boundary o_boundary])
xlabel('s')
title('$p(\hat{C} = 1 \mid s$)','Interpreter','Latex')
if strcmp(tasktype,'qamar')
    hold on
    plot(k,0.5*ones(1,length(sig)),'bo')
    plot(-k,0.5*ones(1,length(sig)),'bo')
end
legend_and_export('p(Chat=1|s).pdf',sig)
%legend_and_export('p(Chat=1|s).pdf',sig,'plottingk',true)

plot(o,pChat2)
xlim([-o_boundary o_boundary])
xlabel('s')
title('$p(\hat{C} = 2 \mid s$)','Interpreter','Latex')
if strcmp(tasktype,'qamar')
    hold on
    plot(k,0.5*ones(1,length(sig)),'bo')
    plot(-k,0.5*ones(1,length(sig)),'bo')
end
legend_and_export('p(Chat=2|s).pdf',sig)
%legend_and_export('p(Chat=2|s).pdf',sig,'plottingk',true)

%% plot (p(correct or error | s)
plot(o,abs(pChat1-.5)+.5)
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$p(\hat{C}$ is correct $\mid s$)','Interpreter','Latex')
legend_and_export('p(Chatcorrect|s).pdf',sig)

plot(o,-abs(pChat1-.5)+.5)
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$p(\hat{C}$ is error $\mid s$)','Interpreter','Latex')
legend_and_export('p(Chaterror|s).pdf',sig)

%% expected value of Chat
plot(o,expChat)
xlabel('s')
title('$\langle\hat{C}\rangle\mid s$','Interpreter','Latex')
switch tasktype
    case 'qamar'
        hold on
        plot(k,1.5*ones(1,length(sig)),'bo')
        plot(-k,1.5*ones(1,length(sig)),'bo')
        legend_and_export('expectedval(Chat).pdf',sig,'plottingk',true)
        
    case 'kepecs'
        legend_and_export('expectedval(Chat).pdf',sig)
end








%% MCS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc1=tic;
samples = 1e5; % 1e7 samples and 39 bins takes ~ 3 mins
bins = 39; % must be odd to plot a point at 0

% preallocate these vectors, because MATLAB says so
Chat_mean = zeros(length(sig),bins);
all.s = zeros(1,samples); % not necessary for kepecs?
all.Chat = zeros(1,samples); % not necessary for kepecs?

% initialize struct
MCS = struct;


for i = 1:length(sig);
    
    % Generative model C -> s -> x -> d -> posterior -> resp and confidence (g)
    switch tasktype % qamar and kepecs have different ways of generating trials.
        case 'qamar'
            all.C=randi([1 2],1,samples); % Generate C trials, 1 or 2.
            all.s(all.C==1) = randn(1,sum(all.C==1))*sig1; % Generate s trials for Cat1 and Cat2
            all.s(all.C==2) = randn(1,sum(all.C==2))*sig2;
            all.x = all.s + randn(size(all.s))*sig(i); % add noise to s. this line is the same in both tasks
            
            all.d = k1(i) - k2(i) * all.x.^2; % calculate d. Could also just compare to the k generated above.
            
            all.Chat(all.d>0) = 1; % Chat decision matrix based on decision rule
            all.Chat(all.d<0) = 2;
            
        case 'kepecs'
            all.s = normrnd(0,sigprior,1,samples);
            all.C(all.s<0) = 1;
            all.C(all.s>0) = 2;
            all.x = all.s + randn(size(all.s))*sig(i); % add noise to s
            
            mu = (all.x.* sigprior^2)./(sig(i)^2 + sigprior^2);
            k = sig(i) .* sigprior ./ sqrt(sig(i)^2 + sigprior^2);
            %all.d = log(normcdf(0,mu,k)./normcdf(0,-mu,k));
            a=.4;
            denom = sig(i) * sqrt(2);
            all.d = log( (erf((all.x-a)/denom) - erf((all.x+1-a)/denom)) ./ (erf((all.x-1+a)/denom) - erf((all.x+a)/denom)));

            all.Chat(all.d>0) = 1; % this is new. used to do this by looking at whether x < 0. see if it works.
            all.Chat(all.d<0) = 2;
    end
    % this is for both qamar and kepecs
    all.posterior = 1 ./ (1 + exp(-all.d)); % posterior prob for each trial.
    all.tf        = all.Chat == all.C; % logical of correct/error at every trial    
    
    % everything above this should be replaceable by trial_generator.
    
    all.g         = abs(all.posterior - .5) + .5; % find confidence at every trial. this is a special line, because it becomes the basis for the other types of all.g.
    
    
    
    MCS = old_MCS_analysis_fcn(all,MCS,bins,i); % this is the big function that does binning and stuff
    %[MCS.stats MCS.sorted_raw] = data_analysis(all, bin_generator(bins),
    %'data_type','model','modelsig',i); % hopefully this will work soon,
    %but 
end
fprintf('simulation took %g seconds, with %g bins and %g samples.\n',toc(toc1),bins,samples)


%% MCS plots galore!  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% <confidence>
t=1;
plot(MCS(t).o_axis',MCS(t).g_mean') % Plot expected confidence
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
title('$\langle\gamma\rangle$','interpreter','latex')
legend_and_export('expectedval(confidence).pdf',sig)

t=2;
plot(MCS(t).o_axis',MCS(t).g_mean') % Plot <g>|Chat is correct
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat is correct.pdf',sig)

t=3;
plot(MCS(t).o_axis',MCS(t).g_mean') % Plot <g>|Chat is incorrect
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat is incorrect.pdf',sig)

t=4;
plot(MCS(t).o_axis',MCS(t).g_mean') % Plot <g>|Chat = 1
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}=1$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat=1.pdf',sig)

t=5;
plot(MCS(t).o_axis',MCS(t).g_mean') % Plot <g>|Chat = 2
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
title('$\langle\gamma\rangle\mid\hat{C}=2$','interpreter','latex')
legend_and_export('expectedval(confidence)|Chat=2.pdf',sig)

t=6;
plot(MCS(t).o_axis',MCS(t).g_mean') % Plot <g>|Chat = 2
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
title('$\langle\gamma\rangle\mid$C=1','interpreter','latex')
legend_and_export('expectedval(confidence)|C=1.pdf',sig)

t=7;
plot(MCS(t).o_axis',MCS(t).g_mean') % Plot <g>|Chat = 2
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
title('$\langle\gamma\rangle\mid$C=2','interpreter','latex')
legend_and_export('expectedval(confidence)|C=2.pdf',sig)

%% std(confidence)

t=1;
plot(MCS(t).o_axis',MCS(t).g_std') % Plot std(confidence)
ylim([0 .2])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)$','interpreter','latex')
legend_and_export('std(confidence).pdf',sig)

t=2;
plot(MCS(t).o_axis',MCS(t).g_std') % Plot std(g)|Chat is correct
ylim([0 .2])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('std(confidence)|Chat is correct.pdf',sig)

t=3;
plot(MCS(t).o_axis',MCS(t).g_std') % Plot std(g)|Chat is incorrect
ylim([0 .2])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('std(confidence)|Chat is incorrect.pdf',sig)

t=4;
plot(MCS(t).o_axis',MCS(t).g_std') % Plot std(g)|Chat = 1
ylim([0 .2])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}=1$','interpreter','latex')
legend_and_export('std(confidence)|Chat=1.pdf',sig)

t=5;
plot(MCS(t).o_axis',MCS(t).g_std') % Plot std(g)|Chat = 2
ylim([0 .2])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid\hat{C}=2$','interpreter','latex')
legend_and_export('std(confidence)|Chat=2.pdf',sig)

t=6;
plot(MCS(t).o_axis',MCS(t).g_std') % Plot std(g)|Chat = 1
ylim([0 .2])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid$C=1','interpreter','latex')
legend_and_export('std(confidence)|C=1.pdf',sig)

t=7;
plot(MCS(t).o_axis',MCS(t).g_std') % Plot std(g)|Chat = 2
ylim([0 .2])
xlim([-o_boundary o_boundary])
xlabel('s')
title('std$(\gamma)\mid$C=2','interpreter','latex')
legend_and_export('std(confidence)|C=2.pdf',sig)

%% mean vs std

% you can do this one for qamar, but it's not useful
t=1;
plot(MCS(t).g_mean_sort',MCS(t).g_std_sort')
xlim([.5 1])
ylim([0 .2])
xlabel('$\langle\gamma\rangle$','interpreter','latex')
ylabel('$std(\gamma)$','interpreter','latex')
legend_and_export('mean_vs_std.pdf',sig)

% you can do this one for qamar, but it's not useful
t=2;
plot(MCS(t).g_mean_sort',MCS(t).g_std_sort')
xlim([.5 1])
ylim([0 .2])
xlabel('$\langle\gamma\rangle\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}\textrm{  is correct}$','interpreter','latex')
legend_and_export('mean_vs_std|correct.pdf',sig)

% you can do this one for qamar, but it's not useful
t=3;
plot(MCS(t).g_mean_sort',MCS(t).g_std_sort')
xlim([.5 1])
ylim([0 .2])
xlabel('$\langle\gamma\rangle\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}\textrm{  is incorrect}$','interpreter','latex')
legend_and_export('mean_vs_std|incorrect.pdf',sig)

t=4;
plot(MCS(t).g_mean_sort',MCS(t).g_std_sort')
xlim([.5 1])
ylim([0 .2])
xlabel('$\langle\gamma\rangle\mid\hat{C}=1$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}=1$','interpreter','latex')
legend_and_export('mean_vs_std|Chat=1.pdf',sig)

t=5;
plot(MCS(t).g_mean_sort',MCS(t).g_std_sort')
xlim([.5 1])
ylim([0 .2])
xlabel('$\langle\gamma\rangle\mid\hat{C}=2$','interpreter','latex')
ylabel('$std(\gamma)\mid\hat{C}=2$','interpreter','latex')
legend_and_export('mean_vs_std|Chat=2.pdf',sig)

t=6;
plot(MCS(t).g_mean_sort',MCS(t).g_std_sort')
xlim([.5 1])
ylim([0 .2])
xlabel('$\langle\gamma\rangle\mid$C=1','interpreter','latex')
ylabel('$std(\gamma)\mid$C=1','interpreter','latex')
legend_and_export('mean_vs_std|C=1.pdf',sig)

t=7;
plot(MCS(t).g_mean_sort',MCS(t).g_std_sort')
xlim([.5 1])
ylim([0 .2])
xlabel('$\langle\gamma\rangle\mid$C=2','interpreter','latex')
ylabel('$std(\gamma)\mid$C=2','interpreter','latex')
legend_and_export('mean_vs_std|C=2.pdf',sig)

%% Plot confidence Kepecs-style, one sigma value, correct and incorrect.
figure
t=2;
level=1;
plot(MCS(t).o_axis(level,:),MCS(t).g_mean(level,:),'color',[0 0.5 0]) % Plot <g>|Chat is correct
hold on
t=3;
plot(MCS(t).o_axis(level,:),MCS(t).g_mean(level,:),'color',[1 0 0]) % Plot <g>|Chat is correct
xlim([-o_boundary o_boundary])
ylim([.5 1])
xlabel('s')
legend('Correct','Error')
title(sprintf('$\\langle\\gamma\\rangle\\mid\\sigma=%g$',sig(level)),'interpreter','latex')
%export_and_reset('expectedval(confidence)|correct_and_error.pdf')

%% percent correct
% Sorts confidence and std as a function of percent correct. Same as above,
% except with percent correct as the sort index rather than confidence
for t=1:length(MCS);
    [MCS(t).percent_correct_sort, MCS(t).sort_index] = sort(MCS(t).percent_correct,2);
end

for i=1:length(sig);
    for t=1:length(MCS);
        MCS(t).g_std_sort(i,:) = MCS(t).g_std(i,MCS(t).sort_index(i,:));
        MCS(t).g_mean_sort(i,:) = MCS(t).g_mean(i,MCS(t).sort_index(i,:));
    end
end

t=1;
plot(MCS(t).o_axis',MCS(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('% correct')
legend_and_export('percentcorrect.pdf',sig)

t=4; % note that this converges to p(C=1|s)
plot(MCS(t).o_axis',MCS(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid\hat{C}=1$','interpreter','latex')
legend_and_export('percentcorrect|Chat=1.pdf',sig)

t=5; % note that this converges to p(C=2|s)
switch tasktype
    case 'qamar' % skip the center bin, where there are few Chat=2 trials.
        plot(MCS(t).o_axis(:,1:19)',MCS(t).percent_correct(:,1:19)')
        hold on
        plot(MCS(t).o_axis(:,21:end)',MCS(t).percent_correct(:,21:end)')
    case 'kepecs' % plot normally for kepecs
        plot(MCS(t).o_axis',MCS(t).percent_correct')
end
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid\hat{C}=2$','interpreter','latex')
legend_and_export('percentcorrect|Chat=2.pdf',sig)

t=6; 
plot(MCS(t).o_axis',MCS(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid$C=1','interpreter','latex')
legend_and_export('percentcorrect|C=1.pdf',sig)

t=7; 
plot(MCS(t).o_axis',MCS(t).percent_correct')
xlim([-o_boundary o_boundary])
ylim([0 1])
xlabel('s')
title('$\% $ correct$\mid$C=2','interpreter','latex')
legend_and_export('percentcorrect|C=2.pdf',sig)

%% percent correct vs <g>
t=1; % note that this is not useful for qamar
plot(MCS(t).percent_correct_sort',MCS(t).g_mean_sort')
xlim([0 1])
ylim([.5 1])
xlabel('% correct')
ylabel('$\langle\gamma\rangle$','interpreter','latex')
legend_and_export('percentcorrect_vs_meang.pdf',sig)

t=4; % note that this is not useful for kepecs)
plot(MCS(t).percent_correct_sort',MCS(t).g_mean_sort')
xlim([0 1])
ylim([.5 1])
xlabel('$\% $ correct$\mid\hat{C}=1$','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid\hat{C}=1$','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|Chat=1.pdf',sig)

t=5; % note that this is not useful for kepecs)
plot(MCS(t).percent_correct_sort',MCS(t).g_mean_sort')
xlim([0 1])
ylim([.5 1])
xlabel('$\% $ correct$\mid\hat{C}=2$','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid\hat{C}=2$','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|Chat=2.pdf',sig)

t=6; % note that this is not useful for kepecs)
plot(MCS(t).percent_correct_sort',MCS(t).g_mean_sort')
xlim([0 1])
ylim([.5 1])
xlabel('$\% $ correct$\mid$C=1','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid$C=1','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|C=1.pdf',sig)

t=7; % note that this is not useful for kepecs)
plot(MCS(t).percent_correct_sort',MCS(t).g_mean_sort')
xlim([0 1])
ylim([.5 1])
xlabel('$\% $ correct$\mid$C=2','interpreter','latex')
ylabel('$\langle\gamma\rangle\mid$C=2','interpreter','latex')
legend_and_export('percentcorrect_vs_meang|C=2.pdf',sig)

%% percent correct vs std(g)
t=1;
plot(MCS(t).percent_correct_sort',MCS(t).g_std_sort')
xlim([0 1])
ylim([0 .16])
xlabel('% correct')
ylabel('std$(\gamma)$','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect.pdf',sig)

t=4;
plot(MCS(t).percent_correct_sort',MCS(t).g_std_sort')
xlim([0 1])
ylim([0 .16])
xlabel('$\%$ correct$\mid\hat{C}=1$','interpreter','latex')
ylabel('std$(\gamma)\mid\hat{C}=1$','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|Chat=1.pdf',sig)

t=5;
plot(MCS(t).percent_correct_sort',MCS(t).g_std_sort')
xlim([0 1])
ylim([0 .16])
xlabel('$\%$ correct$\mid\hat{C}=2$','interpreter','latex')
ylabel('std$(\gamma)\mid\hat{C}=2$','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|Chat=2.pdf',sig)

t=6;
plot(MCS(t).percent_correct_sort',MCS(t).g_std_sort')
xlim([0 1])
ylim([0 .16])
xlabel('$\%$ correct$\mid$C=1','interpreter','latex')
ylabel('std$(\gamma)\mid$C=1','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|C=1.pdf',sig)

t=7;
plot(MCS(t).percent_correct_sort',MCS(t).g_std_sort')
xlim([0 1])
ylim([0 .16])
xlabel('$\%$ correct$\mid$C=2','interpreter','latex')
ylabel('std$(\gamma)\mid$C=2','interpreter','latex')
legend_and_export('stdg_vs_percentcorrect|C=2.pdf',sig)

%%
fprintf('The whole script took %g seconds.\n',toc)
close all