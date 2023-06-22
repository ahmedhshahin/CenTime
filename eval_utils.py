import torch
import torch.nn as nn
from data_loaders import OSICDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import scipy
import lifelines
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from pycox.models.loss import DeepHitSingleLoss, rank_loss_deephit_single, _rank_loss_deephit
from torch import Tensor
from sksurv.metrics import concordance_index_ipcw

def CIndexIPCW(tr_survtime, tr_labels, hazards, labels, survtime_all):
    ytr  = np.array([(bool(e) ,t) for e, t in zip(tr_labels,tr_survtime)], dtype=[('event', bool),('time',int)])
    yval = np.array([(bool(e) ,t) for e, t in zip(labels,survtime_all)], dtype=[('event', bool),('time',int)])
    return concordance_index_ipcw(ytr, yval, hazards)

def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

def nll_pmf_cr(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean',
               epsilon: float = 1e-7) -> Tensor:
    """Negative log-likelihood for PMF parameterizations. `phi` is the ''logit''.
    
    Arguments:
        phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_risks, n_durations]
            all in (-inf, inf).
        idx_durations {torch.tensor} -- Int tensor with index of durations.
        events {torch.tensor} -- Int tensor with event types.
            {0: Censored, 1: first group, ..., n_risks: n'th risk group}.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            else: sum.
    
    Returns:
        torch.tensor -- Negative log-likelihood.
    """
    # Should improve numerical stability by, e.g., log-sum-exp trick.
    events = events.view(-1) - 1
    event_01 = (events != -1).float()
    idx_durations = idx_durations.view(-1)
    batch_size = phi.size(0)
    sm = pad_col(phi.view(batch_size, -1)).softmax(1)[:, :-1].view(phi.shape)
    index = torch.arange(batch_size)
    part1 = sm[index.long(), events.long(), idx_durations.long()].relu().add(epsilon).log().mul(event_01)
    part2 = (1 - sm.cumsum(2)[index.long(), :, idx_durations.long()].sum(1)).relu().add(epsilon).log().mul(1 - event_01)
    loss = - part1.add(part2)
    return _reduction(loss, reduction)

def rank_loss_deephit_single(phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor,
                             sigma: Tensor, reduction: str = 'mean') -> Tensor:
    """Rank loss proposed by DeepHit authors [1] for a single risks.
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duration and censoring time. 
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, chosen by you.
    Arguments:
        phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_durations]
            all in (-inf, inf).
        idx_durations {torch.tensor} -- Int tensor with index of durations.
        events {torch.tensor} -- Float indicator of event or censoring (1 is event).
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, chosen by you.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum': sum.
    
    Returns:
        torch.tensor -- Rank loss.

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    phi = phi[:,0]
    idx_durations = idx_durations.view(-1, 1)
    # events = events.float().view(-1)
    pmf = pad_col(phi).softmax(1)
    y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.) # one-hot
    rank_loss = _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction)
    return rank_loss

class DeepHitDiscretizedLogistic(DeepHitSingleLoss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor, epoch=None) -> Tensor:
        if epoch == 112: import pdb; pdb.set_trace()
        mean, scale = phi.chunk(2, dim=1)
        _phi = log_discretized_logistic(torch.arange(676).unsqueeze(0).repeat(events.size(0),1), mean, scale, 1)
        _phi = _phi.unsqueeze(1)
        nll = nll_pmf_cr(_phi, idx_durations, events, self.reduction)
        rank_loss = rank_loss_deephit_single(_phi, idx_durations, events, rank_mat, self.sigma,self.reduction)
        return self.alpha * nll + (1. - self.alpha) * rank_loss

def get_acc_per_class(preds, gts, classes):
    out = []
    for cls in np.unique(classes):
        out.append(np.abs(preds[classes==cls]-gts[classes==cls]).mean())
    return out

def CoxLoss(survtime, censor, hazard_pred, device='cpu'):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    if isinstance(censor, np.ndarray): censor = torch.FloatTensor(censor).to(device)
    if isinstance(survtime, np.ndarray): survtime = torch.FloatTensor(survtime).to(device)
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    # loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    loss_cox = (theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor.reshape(-1)
    # loss_cox = loss_cox.sum()# / censor.sum()
    loss_cox = loss_cox.sum() / censor.sum()
    return -loss_cox

def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

def log_min_exp(a, b, epsilon=1e-8):
    """
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
     log(exp(a) - exp(b))
     c + log(exp(a-c) - exp(b-c))
     a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y

def log_discretized_logistic(x, mean, logscale, inverse_bin_width):
    scale = torch.exp(logscale)
    mean  = F.softplus(mean)
    logp = log_min_exp(
        F.logsigmoid((x + 0.5 / inverse_bin_width - mean) / scale),
        F.logsigmoid((x - 0.5 / inverse_bin_width - mean) / scale))
    # p = torch.sigmoid((x + 0.5 / inverse_bin_width - mean) / scale) - torch.sigmoid((x - 0.5 / inverse_bin_width - mean) / scale)

    return logp

def hazard2grade(hazard, p):
    if hazard < p:
        return 0
    return 1

def makeKaplanMeierPlot(ckpt_name='./checkpoints/surv_cox', split='val', zscore=False, model='clinical_only', agg_type='Hazard_mean'):
    def hazard2KMCurve(data, subtype):
        p = np.percentile(data['Hazard'], 50)
        data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
        kmf_pred = lifelines.KaplanMeierFitter()
        kmf_gt = lifelines.KaplanMeierFitter()

        def get_name(model):
            mode2name = {'pathgraphomic':'Pathomic F.', 'pathomic':'Pathomic F.', 'graphomic':'Pathomic F.', 'path':'Histology CNN', 'graph':'Histology GCN', 'omic':'Genomic SNN'}
            for mode in mode2name.keys():
                if mode in model: return mode2name[mode]
            return 'N/A'

        fig = plt.figure(figsize=(14, 8), dpi=600)
        ax = plt.subplot()
        censor_style = {'ms': 20, 'marker': '+'}
        
        temp = data[(data['diagnosis']=='IPF') | (data['diagnosis']=='IPF/CPFE')]
        kmf_gt.fit(temp['last_followup_time(weeks)'], temp['death(dead=1,alive=0)'], label="IPF or IPF/CPFE")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=3, ls='--', markerfacecolor='black', censor_styles=censor_style)
        temp = data[data['grade_pred']==0]
        kmf_pred.fit(temp['last_followup_time(weeks)'], temp['death(dead=1,alive=0)'], label="%s (Low)" % model)
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=4, ls='-', markerfacecolor='black', censor_styles=censor_style)

        temp = data[(data['diagnosis'] != 'IPF') & (data['diagnosis'] != 'IPF/CPFE')]
        kmf_gt.fit(temp['last_followup_time(weeks)'], temp['death(dead=1,alive=0)'], label="Non-IPF")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=3, ls='--', censor_styles=censor_style)
        temp = data[data['grade_pred']==1]
        kmf_pred.fit(temp['last_followup_time(weeks)'], temp['death(dead=1,alive=0)'], label="%s (High)" % model)
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=4, ls='-', censor_styles=censor_style)

        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.001, 0.5))

        ax.tick_params(axis='both', which='major', labelsize=40)    
        # plt.legend(fontsize=32, prop=font_manager.FontProperties(family='Arial', style='normal', size=32))
        plt.legend(fontsize=16, prop=font_manager.FontProperties(family='Arial', style='normal', size=16))
        return fig
    
    data = poolSurvTestPD(ckpt_name, model, split, zscore, agg_type)
    # for subtype in ['idhwt_ATC', 'idhmut_ATC', 'ODG']:
    #     fig = hazard2KMCurve(data[data['Histomolecular subtype'] == subtype], subtype)
    #     fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, subtype))
        
    fig = hazard2KMCurve(data, 'all')
    fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, 'all'))

def CI_pm(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return str("{0:.4f} ± ".format(m) + "{0:.3f}".format(h))

def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))


def get_loaders(opt, fold, load_img=True):
    def seed_worker(worker_id):
        import random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(123)
    dataset_train = OSICDataset(root_dir=opt.root_dir, split='train', fold=fold, load_img=load_img, augment=opt.augment, n=opt.n, impute=opt.impute)
    dataset_test  = OSICDataset(root_dir=opt.root_dir, split='val', fold=fold, load_img=load_img, augment=False, n=opt.n, impute=opt.impute)

    train_loader  = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader  = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
    return train_loader, test_loader

def softmax(x, dim):
    if dim == 0:
        return np.exp(x) / np.sum(np.exp(x), axis=dim)[None]
    elif dim == 1:
        return np.exp(x) / np.sum(np.exp(x), axis=dim)[:,None]
    else:
        assert False

def accuracy(logits, target, reduction='sum'):
    assert reduction in ['mean','sum']
    if isinstance(logits, torch.Tensor):
        assert isinstance(target, torch.Tensor)
        pred = torch.argmax(logits, dim=1, keepdim=True)
        out  = torch.eq(pred, target)
        if reduction == 'sum':
            return out.float().sum()
        elif reduction == 'mean':
            return out.float().mean()
    else:
        assert isinstance(logits, np.ndarray)
        assert isinstance(target, np.ndarray)
        pred = np.argmax(logits, axis=1).squeeze()
        out  = (pred == target.squeeze())
        if reduction == 'sum':
            return out.sum()
        elif reduction == 'mean':
            return out.mean()

def auroc(logits, target):
    if isinstance(logits, torch.Tensor):
        assert isinstance(target, torch.Tensor)
        pred = logits.cpu().numpy()
        target = target.cpu().numpy()
    pred = softmax(logits, dim=1)
    return roc_auc_score(target, pred, average='weighted', multi_class='ovr')


def count_parameters(model):
    if isinstance(model, list):
        out = 0
        for m in model:
            out += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return out
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_losses(opt):
    _dict = {'mae': nn.L1Loss(), 'mse': nn.MSELoss()}
    return _dict[opt.l1], _dict[opt.l_cyclic], _dict[opt.l_reg]


def weights_init_xavier_uni(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

def weights_init_kaiming_normal_leaky_relu(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

def weights_init_kaiming_normal_relu(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

def weights_init_kaiming_uni(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_uniform_(m.weight.data, a=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

def weights_init_model(opt, model):
    if opt.init_type == 'xavier':
        model.apply(weights_init_xavier_uni)
    elif opt.init_type == 'kaiming_uni':
        model.apply(weights_init_kaiming_uni)
    elif opt.init_type == 'kaiming_normal_relu':
        model.apply(weights_init_kaiming_normal_relu)
    elif opt.init_type == 'kaiming_normal_leaky_relu':
        model.apply(weights_init_kaiming_normal_leaky_relu)