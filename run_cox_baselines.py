# Base / Native
import os
import pickle

# Numerical / Array
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
pd.options.display.max_rows = 999

# Env
from eval_utils import CI_pm
from eval_utils import makeKaplanMeierPlot


eps = 1e-9

def cont_to_discrete(x,age_bins,fvc_bins,dlco_bins):
    x = x.squeeze()
    y = np.copy(x)
    for idx, item in enumerate(y):
        if np.isnan(item) or (idx in [1,2,3]):
            continue
        elif idx == 0:
            if item < age_bins.min():
                y[idx] = 0
            elif item >= age_bins.max():
                y[idx] = len(age_bins) - 2
            else:
                y[idx] = np.where(age_bins<=item)[0][-1]
        elif idx == 4:
            if item < fvc_bins.min():
                y[idx] = 0
            elif item >= fvc_bins.max():
                y[idx] = len(fvc_bins) - 2
            else:
                y[idx] = np.where(fvc_bins<=item)[0][-1]
        else:
            assert idx == 5
            if item < dlco_bins.min():
                y[idx] = 0
            elif item >= dlco_bins.max():
                y[idx] = len(dlco_bins) - 2
            else:
                y[idx] = np.where(dlco_bins<=item)[0][-1]
    return y

def discrete_to_cont(x, locs):
    mean_vals = []
    for j,k in zip(locs[:-1], locs[1:]): mean_vals.append(np.mean((j,k)))
    return mean_vals[int(x)]

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

def condexp(logp, dist_var=None):
    out = np.exp(logp-logp.max())
    return condp(out, dist_var)


def impute_x(x, ph, pxgh, k, selection_method,age_bins,fvc_bins,dlco_bins):
    t = x.copy()
    t = cont_to_discrete(t,age_bins,fvc_bins,dlco_bins)
    logpxmissing = np.log(ph).copy()
    obs_idx = np.where(np.isnan(t)==False)[0]
    miss_idx = np.where(np.isnan(t)==True)[0]
    st = {} 
    for c in range(k.max()): 
        st[c] = np.where(t==c)[0]
        for j in st[c]: logpxmissing += np.log(pxgh[j][:,c]+eps) # obs data

    combs = np.array(np.meshgrid(*[np.arange(k[m]).tolist() for m in miss_idx])).T.reshape(-1,len(miss_idx))

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

    x[miss_idx] = vals
    return vals, pxmissing, combs, x

def impute_df(df):
    df = df.copy()
    fvc_bins = np.array([26., 51.4131795, 56.99145928, 60.04722793, 64.00657197, 66.85635996, 68.90336484, 72., 74.49059669, 77., 79.04399063, 82., 84., 86., 89., 91., 94.09979707, 98.74254553, 104., 110.275, 135.7723577])
    dlco_bins = np.array([1.26, 5.00623267494, 6.104, 7.103, 7.774, 8.67, 9.284, 9.789, 10.286000000000001, 10.908999999999999, 11.84, 12.437, 12.904, 13.262, 14.058, 14.885, 15.724, 17.244, 19.092, 20.642999999999997, 61.0])
    age_bins = np.array([35.0, 57.0, 62.0, 65.0, 67.0, 69.0, 71.0, 73.0, 75.0, 78.0, 91.0])
    with open("/home/ashahin/codes/survival_analysis/phgen.txt","rb") as f: ph = pickle.load(f)
    with open("/home/ashahin/codes/survival_analysis/pxghgen.txt","rb") as f: pxgh = pickle.load(f)
    for row in df.iterrows():
        orgn = row[1].values[:6]
        miss_idx = np.where(np.isnan(orgn)==True)[0]
        if len(miss_idx) == 0: continue
        temp = impute_x(orgn, ph, pxgh, np.array([10, 2, 3, 2, 20, 20]), 'argmax',age_bins,fvc_bins,dlco_bins)[0]
        for m, yhat in zip(miss_idx, temp):
            if m == 4:
                orgn[m] = discrete_to_cont(yhat, fvc_bins)
            elif m == 5:
                orgn[m] = discrete_to_cont(yhat, dlco_bins)
            else:
                assert m in [2,3]
                orgn[m] = yhat
        df.loc[row[0], df.columns[:6]] = orgn
    return df



def trainCox(dataroot = '/SAN/medic/IPF', ckpt_name='checkpoints/surv_cox/', model='clinical_only', penalizer=1e-4):
    ### Creates Checkpoint Directory
    if not os.path.exists(ckpt_name): os.makedirs(ckpt_name)
    if not os.path.exists(os.path.join(ckpt_name, model)): os.makedirs(os.path.join(ckpt_name, model))
    
    data = pd.read_csv(os.path.join(dataroot, 'ahmed_surv_analysis.csv'))
    data = data.loc[pd.notna(data['fold1'])]
    data['contemporaneous_fvc_week'] = data['contemporaneous_fvc_week'].fillna(0)
    data['time_to_deathOrCensoring'] = data['time_to_deathOrCensoring'] - data['contemporaneous_fvc_week']

    model_feats = ['age','sex(male=1,female=0)','smoking(never=0,ex=1,current=2)','antifibrotic','contemporaneous_fvc_percent','contemporaneous_dlco']
    extended_attributes = model_feats + ['deathOrCensoring','time_to_deathOrCensoring','fold1','fold2','fold3','fold4','fold5']
    data = data[extended_attributes]
    data[model_feats] = impute_df(data[model_feats])

    data2 = pd.read_csv(os.path.join(dataroot, 'cohort2_surv_analysis.csv'))
    data2 = data2.loc[pd.notna(data2['fold1'])]
    data2['contemporaneous_fvc_week'] = data2['contemporaneous_fvc_week'].fillna(0)
    data2['time_to_deathOrCensoring'] = data2['time_to_deathOrCensoring'] - data2['contemporaneous_fvc_week']

    data2 = data2[extended_attributes[:-4]]
    data2[model_feats] = impute_df(data2[model_feats])

    cv_results = []

    for fold in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
    # for fold in ['NewFold1', 'NewFold2', 'NewFold3']:
        df_train = data.loc[data[fold]=='train']
        df_val = data.loc[data[fold]=='val']
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(df_train[model_feats+['time_to_deathOrCensoring','deathOrCensoring']], duration_col='time_to_deathOrCensoring', event_col='deathOrCensoring', show_progress=False)
        cin = concordance_index(df_val['time_to_deathOrCensoring'], -cph.predict_partial_hazard(df_val[model_feats]), df_val['deathOrCensoring'])
        cv_results.append(cin)
        
        df_train.insert(loc=0, column='Hazard_'.format(fold), value=-cph.predict_partial_hazard(df_train))
        df_val.insert(loc=0, column='Hazard'.format(fold), value=-cph.predict_partial_hazard(df_val))
        pickle.dump(df_train, open(os.path.join(ckpt_name, model, '%s_%s_pred_train.pkl' % (model, fold)), 'wb'))
        pickle.dump(df_val, open(os.path.join(ckpt_name, model, '%s_%s_pred_val.pkl' % (model, fold)), 'wb'))
        
    pickle.dump(cv_results, open(os.path.join(ckpt_name, model, '%s_results.pkl' % model), 'wb'))
    print("C-Indices across Splits", cv_results)
    print("Average C-Index: {}".format(CI_pm(cv_results)))


print('Clinical Data Only')
trainCox(model='clinical_only', penalizer=1e-1, ckpt_name='checkpoints/surv_cox_ipf/')

# print('KM-Curves')
# makeKaplanMeierPlot(ckpt_name='./checkpoints/surv_15_cox/', model='clinical_only', split='val')