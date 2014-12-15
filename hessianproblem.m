cd('/Users/will/Google Drive/Will - Confidence/Analysis/optimizations')
load('bighess.mat');
ex = gen.opt(2).extracted(3);
[v,d]=eig(ex.best_hessian);
v1 = v(1,:)';

raw=gen.data(3).raw;
model = gen.opt(2).name;
p = ex.best_params;
nSets = 21;
eigfactors = linspace(-.01,.01,nSets);

[eigfac,eigvec]=meshgrid(eigfactors,v1);
eigvec_multiples = eigfac.*eigvec;

pp = repmat(p,1,nSets);
pvec = pp+eigvec_multiples;

ll = zeros(1,length(eigfactors));
for i = 1:length(eigfactors)
    i
    ll(i) = -nloglik_fcn(pvec(:,i),raw,model);
end

plot(eigfactors,ll)