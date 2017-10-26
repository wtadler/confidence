import scipy.stats as sps
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection
import seaborn as sns

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



def mix_white(color, x=.5):
    return (1-x)*np.array(color)+x


def plot_divergence_panel(ax, data, nBins=100, Bayes=True, xlabel=True, ylabel=True, absolute=True, cat=None, fade_below_prop=.3, conf_threshold=None):
    # plot confidence as a function of stimulus and correctness (if conf_threshold=None)
    # or correctness as a function of stimulus and confidence (using arbitrary conf_threshold)

    if cat is None:
        cat_idx = data['C']<100  # just get index of everything
    else:
        cat_idx = data['C']==cat
    
    if Bayes:
        conf = data['conf']
        conflabel = 'mean Bayesian\nconfidence'# $p(C=\hat{C} \mid x)$'
    else:
        conf = abs(data['x'])
        conflabel = 'measurement magnitude $|x|$'
    
    if conf_threshold is None:
        colors=np.array([[235,51,139], [37,132,8]])/255.0 # red, green
        condlabels=['incorrect', 'correct']
        indices = [data['correctness']==False, data['correctness']==True]
        y = conf
        y_label = conflabel
    else:
        colors=np.array([[150,150,150], [0,0,0]])/255.0 # gray, black
        condlabels=['high conf.', 'low conf.']
        indices = [conf>conf_threshold, conf<conf_threshold]
        y = data['correctness']
        y_label = 'correctness'

    for j, idx in enumerate(indices):
        index = idx & cat_idx

        if absolute:
            s = abs(data['s'][index])
        else:
            s = data['s'][index]

        means, edges = sps.binned_statistic(s, y[index], statistic=np.mean, bins=nBins)[0:2]
        
        counts = np.histogram(s, edges)[0]
        max_count_for_white_mixing = max(counts)*fade_below_prop
        white_mix_frac = np.around(np.maximum(0, (max_count_for_white_mixing - counts)/max_count_for_white_mixing), decimals=3)
        faded_colors = mix_white(np.tile(colors[j], (len(counts), 1)), white_mix_frac[:,np.newaxis])

        centers = edges[0:-1]+np.diff(edges)/2
        xy=np.concatenate((centers[:,np.newaxis], means[:, np.newaxis]), axis=1).reshape(-1,1,2)
        segments = np.hstack([xy[:-1], xy[1:]])

        collection = LineCollection(segments, color=faded_colors)

        ax.add_collection(collection)
        
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    sns.despine(ax=ax)
    
    
    if xlabel:
        ax.set_xlabel('stimulus magnitude $|s|$')
    else:
        ax.set_xticklabels('')
        
    if ylabel:
        ax.set_ylabel(y_label)
    else:
        ax.set_yticklabels('')


def make_heatmap(slopes, SDs, cat_params, ax, xticks=np.arange(0,3,.5), yticks=np.arange(0,3,.5), cbar_ax=None):

    sns.heatmap(slopes, ax=ax, center=0, vmin=-.2, vmax=.2, cmap=sns.diverging_palette(70, 275, s=55, l=65, as_cmap=True),
                linecolor=None, rasterized=True, square=True, xticklabels=SDs, yticklabels=cat_params, cbar=True, cbar_ax=cbar_ax)

    ax.set_xticklabels(xticks, rotation=0)
    ax.set_xticks(np.interp(xticks, SDs, range(len(SDs))))
    
    ax.set_yticklabels(yticks)
    ax.set_yticks(len(cat_params)-np.interp(yticks, cat_params, range(len(cat_params))))

    ax.invert_yaxis()
    
    return ax

def get_slope(cat_param, type, nTrials, SD):
    if type=='uniform':
        data = sg.experiment(type=type, nTrials=int(nTrials), meas_SD=SD, unif_range=cat_param)
    else:
        data = sg.experiment(type=type, nTrials=int(nTrials), meas_SD=SD, norm_mean=1, norm_SD=cat_param)

    if data['correctness'].all():
        slope=np.nan
    else:
        false_index = ~data['correctness']
        means, edges = sps.binned_statistic(abs(data['s'][false_index]), data['conf'][false_index], statistic=np.mean, bins=nBins)[:2]

        centers = edges[0:-1]+np.diff(edges)/2

        if np.isnan(means).all():
            slope = np.nan
        else:
            idx = (means==means) & (centers==centers)
            slope = sps.linregress(centers[idx], means[idx])[0] # (np.diff(means[~np.isnan(means)][[0,-1]]) # diff of first and last non-nan

    return slope


def incorrect_slopes(type='uniform', SDs=np.linspace(0.01,2,8), cat_params=np.linspace(0.01,2,7), nTrials=5e5, nBins=30):
    # have this make a full list and feed that to Parallel rather than reinstancing Parallel so many times
    
    num_cores = 8
    
    n_cat_params = len(cat_params)
    
    slopes = np.empty((len(SDs), n_cat_params))

    for i, SD in enumerate(SDs):
        slopes[i] = jl.Parallel(n_jobs=num_cores)(jl.delayed(get_slope)(cat_param, type=type, nTrials=nTrials, SD=SD) for cat_param in cat_params)
        
    return slopes



def plot_slopes_3d(SDs, cat_params, slopes, points=np.array([[0,0,0],[1,1,1]]), type='uniform'):
    if type == 'uniform':
        xtitle = 'unif. range'
    else:
        xtitle = 'sig_C'
    
#     interpolator = spi.interp2d(SDs, cat_params, slopes_u)
#     z_points = np.array([interpolator(x=i[0], y=i[1])[0] for i in points])
    
    data = [
        go.Surface(x=cat_params,
                   y=SDs,
                   z=slopes,
                   cmin=-.2,
                   cmax=.2),
        go.Surface(x=cat_params,
                   y=SDs,
                   z=0*slopes,
                   opacity=0.8),
        go.Scatter3d(x=points[:,0],
                     y=points[:,1],
                     z=points[:,2],#z_points+.03,
                     mode='markers',
                     marker=dict(size=6, color='black', opacity=.8))
    ]

    layout = go.Layout(
        scene=dict(
            zaxis=dict(
                title='confidence slope (incorrect trials)',
                range=[-.2,.2]),
            yaxis=dict(
                title='$\sigma$',
                range=[0,max(SDs)]),
            xaxis=dict(title=xtitle,
                       range=[0,max(cat_params)]),
            aspectratio=dict(x=1,y=1,z=1),
            camera=dict(eye=dict(x=-2,y=-2,z=1))),
        font=dict(family='Helvetica')
    )

    fig = go.Figure(data=data, layout=layout)
    plo.iplot(fig)

