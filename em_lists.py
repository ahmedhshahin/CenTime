import numpy as np
from scipy.special import logsumexp
from fancyimpute import IterativeImputer
from glob import glob
from scipy import stats
import pandas as pd

# EM training of a Mixture of a product of categorical distributions
# we rely on lists because number of states in every variable is different. If you variables have the same number of states then lists can be converted to arrays and many parts of the script will be more efficient.

eps = 1e-9

def cont_to_discrete(x, nbins=None, bins=None):
    '''
    Converts continuous variable into discrete variable by equal-frequency binning.
    User should provide either the desired number of bins or bins locations.
    '''
    assert (bins is not None) or (nbins is not None)
    if bins is None:
        bins = []  
        for i in range(nbins+1):
            bins.append(np.quantile(x[~np.isnan(x)],i/nbins)) 
    y = np.copy(x)
    x[np.isnan(x)] = -1
    for idx, (i, j) in enumerate(zip(bins[:-1], bins[1:])):
        if j == bins[-1]: j += 1
        y[(x>=i) & (x<j)] = idx
    return y, bins

def discrete_to_cont(x, locs):
    '''
    converts a discrete value back to continous by converting the state to the center of the original bin used for quantization
    for example:
        x_cont = array([3.72833116, 9.93019292, 5.74791191, 2.12599966, 3.22907965])
        x_disc = cont_to_discrete(x_cont,3)[0] = array([1., 2., 2., 0., 0.])
        x_bins = x_bins = cont_to_discrete(x_cont,3)[1] = [2.12599966, 3.39549682, 5.074718326666666, 9.93019292]
        x_cont_recovered = discrete_to_cont(0, x_bins) = 2.76074824 # converting state 0
    '''
    mean_vals = []
    for j,k in zip(locs[:-1], locs[1:]): mean_vals.append(np.mean((j,k)))
    return mean_vals[int(x)]

def discretize_df(df, feats, nbins):
    df_orgn = df.copy()
    bins = []
    for feat, nbin in zip(feats, nbins):
        df[feat], _bins = cont_to_discrete(df[feat].values, nbin) 
        bins.append(_bins)
    return df.values, df_orgn.values, bins
 
def condp(x, dist_var=None):
    x = x.copy()
    if dist_var is None:
        y = x/x.sum()
    else:
        other_var = [i for i in np.arange(x.ndim) if i not in dist_var]
        y = x.transpose(other_var + dist_var)
        m = {}
        for i,j in zip(range(x.ndim),other_var+dist_var): m[j]=i # remember mapping between axes to perform anti-transpose
        y = y.reshape(*np.array(x.shape)[other_var],-1)
        y /= (y.sum(-1, keepdims=True)+eps)
        y = y.reshape(*np.array(x.shape)[other_var], *np.array(x.shape)[dist_var])
        y = y.transpose([m[i] for i in range(y.ndim)]) 
    return y

def condp_lists(x, dist_var):
    dist_var -= 1
    out = []
    for l in x:
        if dist_var is None:
            out.append(l/l.sum())
        else:
            out.append(l/l.sum(dist_var,keepdims=True))
    return out

def condexp(logp, dist_var=None):
    out = np.exp(logp-logp.max())
    return condp(out, dist_var)

def condexp_lists(logp, dist_var=None):
    dist_var = np.array(dist_var)
    temp = []
    for i in logp: temp.extend(i.flatten())
    _max = max(temp)
    for i in range(len(logp)):
        logp[i] -= _max
        logp[i] = np.exp(logp[i])
        logp[i] /= logp[i].sum(tuple(dist_var-1), keepdims=True)
    return logp


def MixProdCategoricalGeneral(x,H,k=[10, 2, 3, 2, 20, 20],thresh=1e-6):
    '''
    Fitting (training) EM
    inputs:
    x: input matrix, each row is a data point (N,D)
    H: Number of mixture components
    k: number of states in every variable (e.g., the first variable 'age' has 10  discrete states)
    thresh: stop training if relative error is less than this value
    outputs:
    ph: p(h)
    pxgh: p(x|h)
    phgx: p(h|x)
    hist: history of logliklihood per iteration
    '''
    k = np.array(k)
    N, D = x.shape
    ph = condp(np.random.rand(H))
    pxgh = condp_lists([np.random.rand(H,i) for i in k], dist_var=2)
    phgx = np.zeros((N,H))
    prev_loglik = 0
    hist = []
    # while True:
    for i in range(10000000):
        # E step
        htot = np.zeros(H,)
        xhtot = [np.zeros((H,i)) for i in k]
        loglik = 0

        for n in range(x.shape[0]):
            obs_idx = np.where(np.isnan(x[n])==False)[0]
            miss_idx = np.where(np.isnan(x[n])==True)[0]
            if len(miss_idx)>0:
                logqhtilde = [np.log(ph)[:,None].repeat(i,1) for i in k] # DxHxC
                logqh = np.log(ph)

                st = {}
                for c in range(k.max()):
                    st[c] = np.where(x[n]==c)[0]

                    temp = []
                    for j in st[c]:
                        temp.append(pxgh[j][:,c])
                        logqhtilde[j][:,c] += np.log(pxgh[j][:,c]+eps)
                    for j in miss_idx:
                        if c < k[j]: logqhtilde[j][:,c] += np.log(pxgh[j][:,c]+eps)

                    temp = np.asarray(temp)
                    logqh += np.sum(np.log(temp+eps),0)


                combs = np.array(np.meshgrid(*[np.arange(k[m]).tolist() for m in miss_idx])).T.reshape(-1,len(miss_idx))

                temp = []
                for _j,_k in enumerate(miss_idx): temp.append(logqhtilde[_k][:,combs[:,_j]])
                temp = np.asarray(temp).sum((0,2))
                temp += logqh
                htot += condp(temp)
                
                qhtildegx = condexp_lists(logqhtilde, dist_var=[1,2])
                qhgx = condexp(logqh)

            else:
                logqh = np.log(ph)
                st = {}
                for c in range(k.max()):
                    st[c] = np.where(x[n]==c)[0]
                    temp = []
                    for j in st[c]: temp.append(pxgh[j][:,c])
                    temp = np.asarray(temp)
                    logqh += np.sum(np.log(temp+eps),0)
                qhgx = condexp(logqh)
                htot += qhgx
            
            phgx[n] = qhgx

            for c in range(k.max()):
                for j in st[c]: xhtot[j][:,c] += qhgx
                if len(miss_idx) > 0: 
                    for j in miss_idx:
                        if c < k[j]: xhtot[j][:,c] += qhtildegx[j][:,c]
            loglik += logsumexp(logqh)

        # M step
        ph = condp(htot)
        pxgh = condp_lists(xhtot,2)

        # convergence
        delta = np.abs((loglik-prev_loglik)/(prev_loglik+eps))
        prev_loglik = loglik
        hist.append(loglik)
        if delta < thresh: break

    hist = np.array(hist)
    return ph, pxgh, phgx, hist

def impute_x(t, ph, pxgh, k=[10, 2, 3, 2, 20, 20], selection_method='argmax'):
    t = t.copy()
    k = np.array(k)
    logpxmissing = np.log(ph).copy()
    obs_idx = np.where(np.isnan(t)==False)[0]
    miss_idx = np.where(np.isnan(t)==True)[0]

    st = {} 
    for c in range(k.max()): 
        st[c] = np.where(t==c)[0]
        for j in st[c]: logpxmissing += np.log(pxgh[j][:,c]+eps) # obs data

    combs = np.array(np.meshgrid(*[np.arange(k[m]).tolist() for m in miss_idx])).T.reshape(-1,len(miss_idx))

    ###
    temp = []
    for i,j in enumerate(miss_idx): temp.append(np.log(pxgh[j][:,combs[:,i]]+eps))
    temp = np.asarray(temp).sum(0)
    logpxmissing = logpxmissing[:,None] + temp
    ###

    pxmissing = condexp(logpxmissing).sum(0) # sum over h
    if selection_method == 'argmax':
        vals = combs[np.argmax(pxmissing,0)]
    elif selection_method == 'mean':
        vals = (pxmissing[:,None]*combs).sum(0)
        vals = np.round(vals).astype(int)
    elif selection_method == 'sample':
        idx = np.random.choice(np.arange(len(combs)), size=1, p=pxmissing)[0]
        vals = combs[idx]
    else:
        assert False

    t[miss_idx] = vals
    return vals, combs, pxmissing, t

###############################
### 
# Example
df = pd.read_csv('clinical_data.csv')
feats = np.array(['age','sex(male=1,female=0)','smoking(never=0,ex=1,current=2)','antifibrotic','contemporaneous_fvc_percent','contemporaneous_dlco'])
df = df[feats]

df_orgn = df.copy()
df.loc[pd.notna(df['age']), 'age'], age_bins = cont_to_discrete(df.loc[pd.notna(df['age']), 'age'], nbins=10)
df.loc[pd.notna(df['contemporaneous_fvc_percent']), 'contemporaneous_fvc_percent'], fvc_bins = cont_to_discrete(df.loc[pd.notna(df['contemporaneous_fvc_percent']), 'contemporaneous_fvc_percent'], nbins=20)
df.loc[pd.notna(df['contemporaneous_dlco']), 'contemporaneous_dlco'], dlco_bins = cont_to_discrete(df.loc[pd.notna(df['contemporaneous_dlco']), 'contemporaneous_dlco'], nbins=20)


ph, pxgh, phgx, hist = MixProdCategoricalGeneral(df.values, 90, k=[10, 2, 3, 2, 20, 20], thresh=1e-6)
t = df.loc[1].values # random clinical record to impute values

vals, combs, pxmissing, imputed_t = impute_x(t, ph, pxgh, k=np.array([10, 2, 3, 2, 20, 20]), selection_method='argmax')
'''
vals: the imputed values
combs: possible combinations of the missing variables
pxmissing: probability of comps
imputed_t: the input vector with missing values imputed
'''