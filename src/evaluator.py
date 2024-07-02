import numpy as np
import sklearn.metrics
import ot
from ot.datasets import make_1D_gauss as gauss
from scipy import stats


def noise_smse(pred_noise,act_noise):
    """Compute the standardized mean squared error.
    Please be consistent, if pred_noise is variance, then act_noise should be a variance.
    And if pred_noise is standard deviation, then act_noise should also be the same.

    Args:
        pred_noise (np.array): numpy array of the predicted noise level at each validation points
        act_noise (np.array): numpy array of the actual predicted noise level at each validation points

    Returns:
        float: standardized mean squared
    """
    mse = sklearn.metrics.mean_squared_error(act_noise, pred_noise)
    var_noise = np.var(act_noise)
    smse = mse/var_noise

    return smse


def nlpd_loss(act, pred, pred_var):
    """Return Negative-log probability density (NLPD): See IMLHGP paper equation 36

    Args:
        act (np.array): reference target value
        pred (np.array): predicted target value
        pred_var (_type_): predicted target variance. CAREFUL! output of gaussian process is often set as stdev

    Returns:
        float: Negative-log probability density (NLPD)
    """
    # first_term = np.mean(np.log(2*np.pi*pred_var)) / 2
    # second_term = np.mean(((act - pred)**2 /(2*pred_var)))
    # nlpd = first_term * second_term
    std = np.sqrt(pred_var.flatten())
    actual = act.flatten()
    mean = pred.flatten()
    nlpd_i = []
    for i,m in enumerate(mean):
        prob = stats.norm(m, std[i]).cdf(actual[i])
        logprob = np.log(prob)
        nlpd_i.append(logprob)
    nlpd = -1 * np.mean(nlpd_i) 
    return nlpd


def wasserstein2(act_dist, pred_dist):
    """Compute Wasserstein-2 distance between distribution

    Args:
        act_dist (list): actual distribution parameter, format [<mean>, <std>]
        pred_dist (list): predicted distribution parameter, format [<mean>, <std>]

    Returns:
        float: Wasserstein-2 distance
    """
    nbin = 100

    # breakdown dist
    mean_act = act_dist[0]
    std_act = act_dist[1]
    mean_pred = pred_dist[0]
    std_pred = pred_dist[1]


    # bin positions
    lo_act = mean_act - 3*std_act
    up_act = mean_act + 3*std_act

    lo_pred = mean_pred - 3*std_pred
    up_pred = mean_pred + 3*std_pred
    lo = np.minimum(lo_act, lo_pred)
    up = np.maximum(up_act, up_pred)
    x = np.linspace(start=lo, stop=up, num=nbin)

    #transform scale to 0-100 scale
    mean_act_transformed = 100 * (mean_act-lo)/(up-lo)
    std_act_transformed = 100 * (std_act)/(up-lo)
    mean_pred_transformed = 100 * (mean_pred-lo)/(up-lo)
    std_pred_transformed = 100 * (std_pred)/(up-lo)

    # distributions
    act_gauss = _gauss_maker(nbin, m=mean_act_transformed, s=std_act_transformed)
    pred_gauss = _gauss_maker(nbin, m=mean_pred_transformed, s=std_pred_transformed)

    # loss matrix
    m2 = ot.dist(x.reshape((nbin, 1)), x.reshape((nbin, 1)), 'sqeuclidean')

    # Wasserstein-2(EMD) distance
    try:
        d_emd2 = ot.emd2(pred_gauss, act_gauss, m2)
    except:
        errmsg = f"Error in calculating Wasserstein distance, distribution's stdev might be too small, predicted stdev: {std_pred_transformed}"
        raise ValueError(errmsg)
    
    dists = [x, act_gauss, pred_gauss]

    return d_emd2, dists, m2

def _gauss_maker(nbin, m, s):
    """If the standard deviation is 0, make a dirac delta dist rounded to nearest integer

    Args:
        nbin (int): number of bins
        m (float): distribution's mean
        s (float): distribution's stdev

    Returns:
        np.array: the distribution PDF/PMF
    """
    if s > 5e-1:  ## Threshold number for approximating gaussian dist with dirac delta
        dist = gauss(nbin, m, s)
    else:
        dist = np.zeros(nbin)
        try:
            dist[int(np.around(m))-1] = 1
        except:
            raise ValueError(f"info: mean value: {m}, stdev: {s}")
    return dist