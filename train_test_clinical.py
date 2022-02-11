import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn.functional import l1_loss
from torch.nn.modules import normalization
from torch.utils.data import DataLoader

from networks import define_optimizer, define_scheduler, get_model
from utils import count_parameters, weights_init_model, get_loaders, CIndex_lifeline, CoxLoss, log_discretized_logistic
from scipy import special
from pycox.models.data import pair_rank_mat

def get_loss(opt):
    if opt.loss_fn == 'deephit':
        from pycox.models.loss import DeepHitSingleLoss
        loss = DeepHitSingleLoss(alpha=opt.alpha, sigma=0.1)
    if opt.loss_fn == 'discretized_logistic':
        from utils import DeepHitDiscretizedLogistic
        loss = DeepHitDiscretizedLogistic(alpha=opt.alpha, sigma=0.1)
    return loss

def train_test(opt, fold, device, writer):
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(123)
    random.seed(123)
    best_cindex = 0.0
    best_epoch = 0
    model = get_model(opt, device)
    loss_fn = get_loss(opt)
    # weights_init_model(opt, model)

    optimizer = define_optimizer(opt,model.parameters())
    scheduler = define_scheduler(opt, optimizer)

    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Optimizer Type:", opt.optimizer_type)

    train_loader, test_loader = get_loaders(opt, fold, load_img=False)

    for epoch in range(0, opt.n_epochs):
        model.train()
        loss_total = 0.0
        for batch_idx, sample in enumerate(train_loader):
            clinical = sample['clinical_data'].to(device)
            event    = sample['event'].to(device)
            time     = sample['time'].to(device)
            pred     = model(clinical)
            
            # loss = CoxLoss(time, event, pred, device=device)
            if opt.loss_fn in ['deephit','discretized_logistic']:
                rank_mat = pair_rank_mat(time.numpy(), event.numpy())
                loss = loss_fn(pred, time.type(torch.int64), event, torch.tensor(rank_mat), epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()*time.size(0)

        loss_total  /= len(train_loader.dataset)

        loss_total_te, cindex_te, preds, events, times = test(opt, model, loss_fn, test_loader, device)

        torch.save({
                'fold': fold,
                'opt': opt,
                'preds': preds,
                'events': events,
                'times': times,
                'loss': loss_total_te,
                'cindex': cindex_te,
                'epoch': epoch,
                'model': model.state_dict()
            }, os.path.join(opt.exps_dir, opt.exp_name, 'clinical_fold_{}.pt'.format(fold)))
        
        if cindex_te > best_cindex:
            print("Saving model")
            best_cindex = cindex_te
            best_epoch = epoch
            torch.save({
                'fold': fold,
                'opt': opt,
                'preds': preds,
                'events': events,
                'times': times,
                'loss': loss_total_te,
                'cindex': cindex_te,
                'epoch': epoch,
                'model': model.state_dict()
            }, os.path.join(opt.exps_dir, opt.exp_name, 'resnet_fold_{}_best.pt'.format(fold)))
        
        print("Epoch: {}".format(epoch))
        print("Training || CoxLoss: {:.4}".format(loss_total))
        print("Testing  || CoxLoss: {:.4}, CIndex: {:.4}".format(loss_total_te, cindex_te))
        print("")
        writer.add_scalars('fold{}/CoxLoss'.format(fold), {'train':loss_total, 'test': loss_total_te}, epoch)
        writer.add_scalar('fold{}/CIndex'.format(fold), cindex_te, epoch)

        if scheduler is not None:
            if opt.lr_policy=='reduce_lr_on_plateau':
                scheduler.step(loss_total_te)
            else:
                scheduler.step()
    return best_cindex

def test_random_batch(opt, model, fold, test_loader, device):
    model.eval()
    idx      = np.random.randint(len(test_loader.dataset))
    sample   = test_loader.dataset.__getitem__(idx)
    img      = sample['img'][None].to(device)
    event    = sample['event'][None].to(device)
    time     = sample['time'][None].to(device)
    clinical = sample['clinical_data'][None].to(device)
    with torch.no_grad():
        pred = model(img, clinical)
        loss = CoxLoss(time, event, pred, device)
    return loss.item()

def test(opt, model, loss_fn, test_loader, device):
    model.eval()
    preds, events, times = [], [], []

    loss_total_te = 0.0
    for batch_idx, sample in enumerate(test_loader):
        time     = sample['time'].to(device)
        event    = sample['event'].to(device)
        clinical = sample['clinical_data'].to(device)
        
        with torch.no_grad():
            pred = model(clinical)

            if opt.loss_fn in ['deephit','discretized_logistic']:
                rank_mat = pair_rank_mat(time.numpy(), event.numpy())
                loss = loss_fn(pred, time.type(torch.int64), event, torch.tensor(rank_mat))

            loss_total_te  += loss.item() * time.size(0)
            if opt.loss_fn == 'discretized_logistic':
                pred = log_discretized_logistic(torch.arange(676).unsqueeze(0).repeat(pred.size(0),1),pred[:,:1],pred[:,1:2],1).argmax(1).detach().cpu().numpy()
            else:
                pred = pred.detach().cpu().numpy()

        if opt.loss_fn == 'deephit':
            preds  = np.concatenate((preds, np.argmax(pred, 1)))
        else:
            preds  = np.concatenate((preds, pred.reshape(-1)))
        events = np.concatenate((events, event.cpu().numpy().reshape(-1)))
        times  = np.concatenate((times, time.cpu().numpy().reshape(-1)))
    loss_total_te  /= len(test_loader.dataset)
    cindex_te = CIndex_lifeline(-preds, events, times)
    return loss_total_te, cindex_te, preds, events, times