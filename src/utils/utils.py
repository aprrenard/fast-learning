"""_summary_

Returns:
    _type_: _description_
"""

import numpy as np
from sklearn.utils import resample


def ci_bootstrap_timeseries(data, nboot=500, ci=95):
    """_summary_

    Args:
        data (_type_): _description_
        nboot (int, optional): _description_. Defaults to 500.
        ci (int, optional): _description_. Defaults to 95.

    Returns:
        _type_: _description_
    """

    nt = data.shape[-1]
    ci_left = np.zeros(nt)
    ci_right = np.zeros(nt)

    for it in range(nt):
        sample = data[:, it]
        sample = sample[~np.isnan(sample)]
        resampling = [resample(sample) for _ in range(nboot)]
        means = np.mean(resampling, 1)
        ci_left[it] = np.percentile(means, (100-ci)/2)
        ci_right[it] = np.percentile(means, ci+(100-ci)/2)

    return ci_left, ci_right


def ci_bootstrap(data, nboot=1000, ci=95):
    """_summary_

    Args:
        data (_type_): _description_
        nboot (int, optional): _description_. Defaults to 1000.
        ci (int, optional): _description_. Defaults to 95.

    Returns:
        _type_: _description_
    """

    resampling = [resample(data) for _ in range(nboot)]
    means = np.mean(resampling, 1)
    ci_left = np.percentile(means, (100-ci)/2)
    ci_right = np.percentile(means, ci+(100-ci)/2)

    return ci_left, ci_right

