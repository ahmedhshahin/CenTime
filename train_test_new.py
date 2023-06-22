import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import get_optimizer, get_model
from eval_utils import count_parameters, get_loaders, CIndex_lifeline, CoxLoss, CIndexIPCW

def train_test(opt, fold, device, writer):
    torch.use_deterministic_algorithms(False)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    best_cindex = 0.0
    best_epoch = 0
    model = get_model(opt, device)
    # if 'daft' in opt.model:
        # saved = torch.load('/home/ashahin/codes/survival_analysis/exps/_{}FixImgOnlyLr-2_4/resnet_fold_{}_best.pt'.format(fold+7, fold),map_location='cpu')['model']
        # model.load_state_dict(saved, False)

    # weights_init_model(opt, model)

    optimizer = get_optimizer(opt,model.parameters())

    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Optimizer Type:", opt.optimizer_type)

    train_loader, test_loader = get_loaders(opt, fold)
    # train_bank = torch.zeros((len(train_loader.dataset),1), device=device)
    train_bank = torch.FloatTensor(len(train_loader.dataset),1).uniform_(-1,1).to(device)
    train_bank = dict(zip(train_loader.dataset.case_ids, train_bank))
    
    # test_bank = torch.zeros((len(test_loader.dataset),1), device=device)
    test_bank = torch.FloatTensor(len(test_loader.dataset),1).uniform_(-1,1).to(device)
    test_bank = dict(zip(test_loader.dataset.case_ids, test_bank))
    for epoch in range(0, opt.n_epochs):
        model.train()
        loss_total = 0.0
        tr_preds, tr_events, tr_times = [], [], []
        for batch_idx, sample in enumerate(train_loader):
            img      = sample['img'].to(device)
            clinical = sample['clinical_data'].to(device)
            event    = sample['event']
            time     = sample['time']
            pred     = model(img, clinical)

            # for i in range(len(pred)): train_bank[sample['case_id'][i]] = pred[i].detach().cpu()
            for i in range(len(pred)): train_bank[sample['case_id'][i]] = pred[i]#.detach().cpu()
            # print(torch.cat(list(train_bank.values())))
            tr_preds  = np.concatenate((tr_preds, pred.detach().cpu().numpy().reshape(-1)))
            tr_events = np.concatenate((tr_events, event.cpu().numpy().reshape(-1)))
            tr_times  = np.concatenate((tr_times, time.cpu().numpy().reshape(-1)))

            # loss = CoxLoss(time, event, pred, device=device)
            loss = CoxLoss(train_loader.dataset.times, train_loader.dataset.events, torch.cat(list(train_bank.values())), device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for i in range(len(pred)): train_bank[sample['case_id'][i]] = pred[i].detach()
            loss_total += loss.item()*event.sum()

            # loss_te = test_random_batch(opt, model, fold, test_loader, device)
            # writer.add_scalars('fold{}/loss_batch'.format(fold), {'train':loss.item(), 'test': loss_te}, epoch*len(train_loader)+batch_idx)

        loss_total  /= train_loader.dataset.events.sum()
        # loss_total  /= len(train_loader)

        loss_total_te, cindex_te, cindex_ipcw_te, preds, events, times = test(model, train_loader, test_loader, device, test_bank)

        # torch.save({
        #         'fold': fold,
        #         'opt': opt,
        #         'preds': preds,
        #         'events': events,
        #         'times': times,
        #         'loss': loss_total_te,
        #         'cindex': cindex_te,
        #         'epoch': epoch,
        #         'model': model.state_dict()
        #     }, os.path.join(opt.exps_dir, opt.exp_name, 'resnet_fold_{}.pt'.format(fold)))
        
        if cindex_ipcw_te > best_cindex:
            print("Saving model")
            best_cindex = cindex_ipcw_te
            best_epoch = epoch
            torch.save({
                'fold': fold,
                'opt': opt,
                'preds': preds,
                'events': events,
                'times': times,
                'tr_preds': tr_preds,
                'tr_events': tr_events,
                'tr_times': tr_times,
                'loss': loss_total_te,
                'cindex': cindex_te,
                'epoch': epoch,
                'model': model.state_dict()
            }, os.path.join(opt.exps_dir, opt.exp_name, 'resnet_fold_{}_best.pt'.format(fold)))
        
        print("Epoch: {}".format(epoch))
        print("Training || CoxLoss: {:.4}".format(loss_total))
        print("Testing  || CoxLoss: {:.4}, CIndex: {:.4}, CIndex IPCW: {:.4}".format(loss_total_te, cindex_te, cindex_ipcw_te))
        print("")
        writer.add_scalars('fold{}/CoxLoss'.format(fold), {'train':loss_total, 'test': loss_total_te}, epoch)
        writer.add_scalar('fold{}/CIndex'.format(fold), cindex_te, epoch)
        writer.add_scalar('fold{}/CIndexIPCW'.format(fold), cindex_ipcw_te, epoch)
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
        pred = torch.tanh(model(img, clinical))
        loss = CoxLoss(time, event, pred, device)
    return loss.item()

def test(model, train_loader, test_loader, device, test_bank):
    model.eval()
    preds, events, times = [], [], []

    loss_total_te = 0.0
    ignored_samples = 0
    for batch_idx, sample in enumerate(test_loader):
        img      = sample['img'].to(device)
        time     = sample['time'].to(device)
        event    = sample['event'].to(device)
        clinical = sample['clinical_data'].to(device)
        
        with torch.no_grad():
            pred = model(img, clinical)
            for i in range(len(pred)): test_bank[sample['case_id'][i]] = pred[i].detach()
            loss = CoxLoss(test_loader.dataset.times, test_loader.dataset.events, torch.cat(list(test_bank.values())), device=device)
            loss_total_te  += loss.item()*event.sum()
            pred = pred.detach().cpu().numpy()

        preds  = np.concatenate((preds, pred.reshape(-1)))
        events = np.concatenate((events, event.cpu().numpy().reshape(-1)))
        times  = np.concatenate((times, time.cpu().numpy().reshape(-1)))

    loss_total_te  /= test_loader.dataset.events.sum()
    # loss_total_te  /= len(test_loader)
    cindex_te = CIndex_lifeline(preds, events, times)
    cindex_ipcw_te = CIndexIPCW(train_loader.dataset.times, train_loader.dataset.events, preds, events, times)[0]
    return loss_total_te, cindex_te, cindex_ipcw_te, preds, events, times