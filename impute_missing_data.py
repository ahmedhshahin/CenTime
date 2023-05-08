'''
A script for missing data imputation using latent variable models, as explained in our paper: https://arxiv.org/pdf/2203.11391.
'''

import numpy as np
from scipy.special import logsumexp
import pandas as pd

def cont_to_discrete(x, nbins=None, bins=None, eps=1e-9):
    '''
    Converts continuous variable into discrete variable by equal-frequency binning.
    User should provide either the desired number of bins or bins locations.
    returns:
        y: discrete variable (same shape as x)
        bins: bin locations (nbins + 1) -- bins[0] is the leftmost bin edge (minimum of x), bins[-1] is the rightmost bin edge (maximum of x)
    '''
    assert (bins is not None) or (nbins is not None)
    if bins is None: bins = [np.nanquantile(x, i/nbins) for i in range(nbins+1)]
    bins[-1] += eps # to include the maximum value of x in the last bin
    y = np.copy(x)
    y = np.digitize(x, bins) - 1
    y = y.astype(float)
    y[np.isnan(x)] = np.nan
    return y, bins

def get_representative_bin_value(bins):
    '''
    returns the representative values of bins (the center of the bin)
    '''
    return [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]

def discrete_to_cont(y, bins):
    '''
    converts a discrete value back to continous by converting the state to the center of the original bin used for quantization (see get_representative_bin_value).
    y: vector of discrete values
    bins: bin locations (nbins + 1)
    '''
    assert len(bins) == len(set(y)) + 1
    assert not np.isnan(y).all()
    representative_values = get_representative_bin_value(bins)
    return np.array([representative_values[int(i)] for i in y])

def discretize_df(df, feats, nbins, continuous_feats):
    '''
    Discretizes the continuous features in a dataframe using equal-frequency binning.
    Returns the discretized dataframe and the bins used for discretization.
    nbins: a list of number of bins for each feature in continuous_feats
    '''
    df = df.copy()
    assert len(continuous_feats) == len(nbins)
    bins = {}
    df = df[feats]
    for i, f in enumerate(continuous_feats):
        df[f], bins[f] = cont_to_discrete(df[f].values, nbins=nbins[i])
    return df, bins

def to_continuous_df(x, bins, feats, continuous_feats, original_df):
    '''
    Converts a discretized matrix back to continuous dataframe.
    x: a matrix of discrete values.
    bins: a dictionary of bins' locations for each (originally) continuous feature. Features not in bins are originally discrete and kept as is.
    '''
    df = pd.DataFrame(x, columns=feats)
    for f in feats:
        if f in continuous_feats:
            df[f] = discrete_to_cont(df[f].values, bins[f])
    df[pd.notna(original_df)] = original_df[pd.notna(original_df)]
    return df

        

def condp(x, dist_var=None):
    '''
    Computes the conditional probability of x along dist_var.
    x: a vector of unnormalized raw probabilities
    dist_var: index of the variable to be normalized (default: None, normalize all variables, i.e., x is a joint distribution)
    returns:
        p: a vector of conditional probabilities
    '''
    if dist_var is None:
        return x / np.sum(x)
    else:
        return x / np.sum(x, axis=dist_var, keepdims=True)

def condexp(x, dist_var=None):
    '''
    Computes the exponential then the conditional probability of x along dist_var. (useful for computing conditional probabilities of log probabilities)
    x: a vector of unnormalized log probabilities
    dist_var: index of the variable to be normalized (default: None, normalize all variables, i.e., x is a joint distribution)
    returns:
        p: a vector of conditional probabilities
    '''
    if dist_var is None:
        return np.exp(x - logsumexp(x))
    else:
        return np.exp(x - logsumexp(x, axis=dist_var, keepdims=True))

def condlogp(x, dist_var=None):
    '''
    Normalizes the log probabilities of x along dist_var.
    '''
    if dist_var is None:
        return x - logsumexp(x)
    else:
        return x - logsumexp(x, axis=dist_var, keepdims=True)
    
class MixtureOfCategoricals():
    '''
    Fits a mixture of H categorical distributions to the data x using EM.
    '''
    def __init__(self, x, H, num_categories, max_iter=100, eps=1e-9, convergence_threshold=1e-3, verbose=False) -> None:
        '''
        x: a matrix of discrete data (N x D)
        H: number of mixture components
        num_categories: a list of number of categories for each feature
        max_iter: maximum number of iterations
        eps: a small number to avoid numerical issues
        convergence_threshold: if the change in log-likelihood is less than this value, the algorithm stops
        '''
        self.N, self.D = x.shape
        self.H = H
        self.max_iter = max_iter
        self.eps = eps
        self.convergence_threshold = convergence_threshold
        self.x = x
        self.num_categories = np.array(num_categories)
        self.history = []
        self.verbose = verbose

        # initialize parameters
        self.logph  = condlogp(np.log(np.random.rand(H) + self.eps))
        self.logpxh = [condlogp(np.log(np.random.rand(H, num_categories[d]) + self.eps)) for d in range(self.D)]
    
    def fit(self):
        '''
        Fits the model using EM.
        '''
        logph = self.logph
        logpxh = self.logpxh
        for iter in range(self.max_iter):
            htot = np.zeros(self.H) # sum of p(h) over all data points
            xhtot = [np.zeros((self.H, self.num_categories[d])) for d in range(self.D)] # sum of p(x|h) over all data points
            loglik = 0

            for n in range(self.N):
                missing_idx = np.where(np.isnan(self.x[n]))[0]
                class_indices = {} # indices of features that have value c
                if len(missing_idx) > 0:
                    logqhtildegx = [logph[:, np.newaxis].repeat(d, 1) for d in self.num_categories]
                    logqhgx = logph.copy() # log q(h|x)

                    for c in range(self.num_categories.max()):
                        class_indices[c] = np.where(self.x[n] == c)[0]

                        # compute log q(x_m, h | x)
                        # observed features
                        for f_idx in class_indices[c]:
                            logqhtildegx[f_idx][:, c] += logpxh[f_idx][:, c]
                            logqhgx += logpxh[f_idx][:, c]
                        # missing features
                        for f_idx in missing_idx:
                            # if c is a valid category for feature f_idx
                            if c < self.num_categories[f_idx]: logqhtildegx[f_idx][:, c] += logpxh[f_idx][:, c]
                    
                    # all possible combinations of missing features
                    combs = np.array(np.meshgrid(*[np.arange(self.num_categories[m]).tolist() for m in missing_idx])).T.reshape(-1,len(missing_idx))

                    # compute p(h)
                    temp = logqhgx.copy()
                    for j, f_idx in enumerate(missing_idx):
                        temp += logqhtildegx[f_idx][:, combs[:, j]].sum(axis=1)
                    htot += condexp(temp)

                    qhtildegx = [condexp(logqhtildegx[d]) for d in range(self.D)]
                    qhgx = condexp(logqhgx)

                else:
                    logqhgx = logph.copy() # log q(h|x)
                    for c in range(self.num_categories.max()):
                        class_indices[c] = np.where(self.x[n] == c)[0]
                        for f_idx in class_indices[c]: logqhgx += logpxh[f_idx][:, c]
                    qhgx = condexp(logqhgx)
                    htot += qhgx

                # update p(x,h)
                for c in range(self.num_categories.max()):
                    for f_idx in class_indices[c]: xhtot[f_idx][:, c] += qhgx
                    if len(missing_idx) > 0:
                        for j in missing_idx:
                            if c < self.num_categories[j]: xhtot[j][:, c] += qhtildegx[j][:, c]

                # update log likelihood
                loglik += logsumexp(logqhgx)

            # update parameters
            htot = condp(htot)
            logph = np.log(htot + self.eps)
            logpxh = [np.log(condp(xhtot[d], dist_var=1) + self.eps) for d in range(self.D)]

            # update history
            self.history.append(loglik)

            # print log likelihood
            if self.verbose: print('iter: {}, loglik: {}'.format(iter, loglik))

            # check convergence
            if iter > 0 and np.abs(self.history[-1] - self.history[-2]) < self.convergence_threshold: break
            
        self.logph = logph
        self.logpxh = logpxh

    def predict(self, x, selection='max', return_prob=False):
        '''
        Predicts the distribution of missing values given the observed values in x.
        p(x_m|x_o) \probto \sum_h p(x_m,x_o|h)p(h) = \sum_h p(x_m|h)p(x_o|h)p(h)
        log p(x_m|x_o) = logsumexp_h (log p(x_m|h) + log p(x_o|h) + log p(h))

        selection: the method to select the value for each missing feature according to the distribution.
            - max: select the value with the highest probability (argmax)
            - sample: sample a value from the distribution
            - mean: take the mean of the distribution
        return_prob: if True, returns the probability of each combination of values for missing features.
        '''
        assert x.ndim == 2, 'x must be a matrix.'
        assert x.shape[1] == self.D, 'The number of features in x must be the same as the number of features in the training data.'
        assert len(self.history) > 0, 'Run fit() first.'
        assert selection in ['max', 'sample', 'mean'], 'Invalid selection method. Must be one of "max", "sample", or "mean".'

        N, D = x.shape
        output = []
        for n in range(N):
            missing_idx = np.where(np.isnan(x[n]))[0]
            if len(missing_idx) == 0:
                output.append(x[n])
                continue
            logpxm_given_xo = self.logph.copy()
            for c in range(self.num_categories.max()):
                class_indices = np.where(x[n] == c)[0]
                for f_idx in class_indices: logpxm_given_xo += self.logpxh[f_idx][:, c]
            
            # all possible combinations of missing features
            combs = np.array(np.meshgrid(*[np.arange(self.num_categories[m]).tolist() for m in missing_idx])).T.reshape(-1,len(missing_idx))

            # log p(x_m|h) = log p(x1|h) + log p(x2|h) + ... = logsumexp_f log p(xf|h)
            logpxm_given_xo = logpxm_given_xo[:, np.newaxis].repeat(len(combs), 1)
            for i in range(len(combs)):
                for j, f_idx in enumerate(missing_idx):
                    logpxm_given_xo[:,i] += self.logpxh[f_idx][:, combs[i, j]]
            logpxm_given_xo = logsumexp(logpxm_given_xo, axis=0)

            # normalize
            pxm_given_xo = condexp(logpxm_given_xo)

            # predict
            y = x[n].copy()
            if selection == 'max':
                y[missing_idx] = combs[np.argmax(pxm_given_xo)]
            elif selection == 'sample':
                y[missing_idx] = combs[np.random.choice(np.arange(len(combs)), p=pxm_given_xo)]
            elif selection == 'mean':
                y[missing_idx] = combs[int((pxm_given_xo * np.arange(len(combs))).sum().round())]

            output.append(y)
        if return_prob:
            return np.array(output), pxm_given_xo, combs
        else:
            return np.array(output)
        
if __name__ == '__main__':
    x = np.random.rand(100, 5)
    x[:, 0] = np.random.choice(np.arange(4), 100) # the first feature is originally discrete
    x_true = x.copy()
    x[np.random.choice(np.arange(100), 20), np.random.choice(np.arange(5), 20)] = np.nan
    df = pd.DataFrame(x)
    bins = [4,2,3,5,5]
    df_discrete, bin_locations = discretize_df(df, feats=df.columns, nbins=bins[1:], continuous_feats=df.columns[1:])
    x_discrete = df_discrete.values

    imputation_model = MixtureOfCategoricals(x_discrete, H=100, num_categories=bins, verbose=True)
    imputation_model.fit()
    imputed = imputation_model.predict(x_discrete, selection='max')

    imputed_df = to_continuous_df(imputed, bin_locations, feats=df.columns, continuous_feats=df.columns[1:], original_df=df)
