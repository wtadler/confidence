import scipy.stats as sps
import numpy as np

def stimulus_distributions(type='uniform', unif_range=1, unif_overlap=0, norm_mean=4, norm_SD=5):
    # p (s | C)
    if type == 'uniform':
        stimdist_1 = sps.uniform(loc=-unif_overlap, scale=unif_range)
        stimdist_n1 = sps.uniform(loc=-unif_range + unif_overlap, scale=unif_range)
    elif type == 'gaussian':
        # 2 gaussians centered at +-norm_mean, with SD norm_SD
        stimdist_1 = sps.norm(loc=norm_mean, scale=norm_SD)
        stimdist_n1 = sps.norm(loc=-norm_mean, scale=norm_SD)
    elif type == 'halfgaussian':
        # 2 half-gaussians with SD norm_SD
        stimdist_1 = sps.norm(loc=0, scale=norm_SD)
        stimdist_n1 = sps.norm(loc=0, scale=norm_SD)
        
    return stimdist_1, stimdist_n1
        
def meanconf_numericalint(type='uniform', meas_SD=.25, s=np.linspace(-5, 5, 1000), dx=0.005, **kwargs):
    # compute expected confidence at s=0 via numerical integration

    # p (s | C)
    stimdist_1, stimdist_n1 = stimulus_distributions(type=type, **kwargs)
    stimpdf_1 = stimdist_1.pdf(s)
    stimpdf_n1 = stimdist_n1.pdf(s)
    if type=='halfgaussian':
        stimpdf_1[s < 0] = 0
        stimpdf_n1[s > 0] = 0
    
    def posterior(x, meas_SD):
        # p(C = 1 | x)

        # p(X=x | s)
        s_belief = sps.norm.pdf(s, loc=x, scale=meas_SD)

        def likelihood(distribution):
            # p(x | C) = integral p(x | s) p(s | C) ds
            return np.trapz(s_belief * distribution, s)

        likelihood_1 = likelihood(stimpdf_1)
        likelihood_n1 = likelihood(stimpdf_n1)

        # p(C=1 | x) = p(x | C=1) / (p(x | C=1) + p (x | C=-1))
        return likelihood_1 / (likelihood_1 + likelihood_n1)

    x = np.arange(0, 2, dx)
    # p(X=x | s=0)
    prob_of_measurement = sps.norm.pdf(x, loc=0, scale=meas_SD)

    integrand = prob_of_measurement * posterior(x[np.newaxis].T, meas_SD)
    integrand[np.isnan(integrand)] = 0

    # E_x|s=0 [p(C = Chat | x)]
    return 2 * np.trapz(integrand, x)


def measurement(s, meas_SD):
    return np.random.normal(loc=s, scale=meas_SD)

def confidence(x, meas_SD, type='uniform', unif_range=1, unif_overlap=0, norm_mean=4, norm_SD=5):
    if type=='uniform': # when non-overlapping, same p(C=1|x) as half gaussian
        likelihood_1 = sps.norm.cdf(unif_range - unif_overlap, loc=x, scale=meas_SD) - sps.norm.cdf(-unif_overlap, loc=x, scale=meas_SD)
        likelihood_n1 = sps.norm.cdf(unif_overlap, loc=x, scale=meas_SD) - sps.norm.cdf(-unif_range + unif_overlap, loc=x, scale=meas_SD)

    elif type=='halfgaussian':
        var = (meas_SD**-2 + norm_SD**-2)**-1
        mu = var*x*meas_SD**-2
        likelihood_1 = 1 - sps.norm.cdf(0, loc=mu, scale=np.sqrt(var))
        likelihood_n1 = 1 - likelihood_1

    elif type=='gaussian':
        sum_SD = np.sqrt(meas_SD ** 2 + norm_SD ** 2)
        likelihood_1 = sps.norm.pdf(x, loc=norm_mean, scale=sum_SD)  # p(x | C = 1)
        likelihood_n1 = sps.norm.pdf(x, loc=-norm_mean, scale=sum_SD)  # p(x | C= -1)
    
    conf = likelihood_1 / (likelihood_1 + likelihood_n1)  # p(C=1 | x)
    conf[conf < 0.5] = 1 - conf[conf < 0.5]  # p(C=Chat | x)
    return conf

def meanconf_sim(type='uniform', meas_SD=.25, nTrials=int(1e4), **kwargs):
    x = measurement(np.zeros(nTrials), meas_SD)
    return np.mean(confidence(x, meas_SD, type=type, **kwargs))

def mean_with_minimum(y, min_nPoints=1e3):
    if len(y)>min_nPoints:
        return np.mean(y)
    else:
        return np.nan



def true_category(nTrials):
    return np.random.choice([-1, 1], size=nTrials)

def stimulus(C, type='uniform', **kwargs):
    s = np.zeros(len(C))

    stimdist_1, stimdist_n1 = stimulus_distributions(type=type, **kwargs)
    s[C==1] = stimdist_1.rvs(size=sum(C==1))
    s[C==-1] = stimdist_n1.rvs(size=sum(C==-1))
    if type=='halfgaussian':
        s[C==1] = abs(s[C==1])
        s[C==-1] = -abs(s[C==-1])
    return s

def choice(x):
    Chat = x>0
    Chat = 2*Chat.astype(int)-1
    return Chat

def experiment(type='uniform', meas_SD=.25, nTrials = int(1e3), **kwargs):
    data = {}
    data['C'] = true_category(nTrials)
    data['s'] = stimulus(data['C'], type=type, **kwargs)
    data['x'] = measurement(data['s'], meas_SD)
    data['Chat'] = choice(data['x'])
    data['correctness'] = data['Chat'] == data['C']
    data['conf'] = confidence(data['x'], meas_SD, type=type, **kwargs)
    return data
