function s = trun_sigstruct(sig, sig1, sig2);
% these are only here for trial_generator. find_intersect_truncated_cats uses an inline version for speed. this is redundant and annoying
s.sig = sig;
s.t=sqrt(log(sig2/sig1)/(.5*(1/sig1^2 - 1/sig2^2)));
s.sigsq = sig^2;
s.sigsqrt2=s.sigsq*sqrt(2);
s.sigsq1 = s.sigsq + sig1^2;
s.sigsq2 = s.sigsq + sig2^2;
s.sig1sqrt2 = sig1*sqrt(2);
s.sig2sqrt2 = sig2*sqrt(2);
s.k_1 = sig.*sig1./sqrt(s.sigsq1);
s.k_2 = sig.*sig2./sqrt(s.sigsq2);
s.k1sq = s.k_1.^2;
s.k2sq = s.k_2.^2;
s.tk_1sqrt2=s.t/(s.k_1*sqrt(2));
s.tk_2sqrt2=s.t/(s.k_2*sqrt(2));
s.k1s2 = s.k_1./s.sigsqrt2;
s.k2s2 = s.k_2./s.sigsqrt2;
s.term1= log(sqrt(s.sigsq2/s.sigsq1)) ...
    + log(1-erf(s.t/s.sig2sqrt2)) ...
    - log(erf(s.t./s.sig1sqrt2));
s.term2= - s.t^2/(2*s.k1sq) ...
    -log(2*sqrt(pi));
s.term3= .5 * (log(pi)-log(2)) ...
    - log(s.k_2) ...
    - 2*log(s.sig) ...
    + s.t^2 / (2*s.k2sq);
s.term4= (2/(s.sigsq*sqrt(2*pi)));
s.term5= (1/s.sigsq2-1/s.sigsq1);
