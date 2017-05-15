def quantile_bins(bins=7, task='B', internal_sigma=0):

    # predicted distribution of stimuli in a Qamar-like experiment. For
    # instance, bins=3 will produce a vector bounds with points at the
    # following quantiles [.333 .666], for 3 bins. It will produce a vector
    # axis for plotting those bins, [ .167 .5 .833].

    from scipy.optimize import fsolve
    from scipy.special import erf
    import numpy as np

    task='B'
    internal_sigma=0

    # task A
    sigma_s = np.sqrt(5**2 + internal_sigma**2)
    mu_1 = -4

    # task B
    sigma_1 = np.sqrt(3**2 + internal_sigma**2)
    sigma_2 = np.sqrt(12**2 + internal_sigma**2)

    q = 1 / bins
    edges = np.zeros(bins - 1);
    centers = np.zeros(bins);

    if task == 'B': # see quantile_bin_generator.pages for explanation of the equation.
        def f(b, q):
            return .25*(2+erf(b/(sigma_1*np.sqrt(2)))+erf(b/(sigma_2*np.sqrt(2)))) - q
    elif task == 'A':
        def f(b, q):
            return .25*(2-erf((mu_1-b)/(sigma_s*np.sqrt(2)))-erf((-mu_1-b)/(sigma_s*np.sqrt(2)))) - q

    for i in range(bins-1):
        edges[i] = fsolve(lambda b: f(b, (i+1)*q), 0)

    center_points = np.linspace(1,2*bins-1,bins)

    for i in range(bins):
        centers[i] = fsolve(lambda b: f(b, center_points[i]*q/2), 0)

    return edges, centers
