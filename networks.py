import torch
import torch.nn as nn
from torch.optim import lr_scheduler

def get_model(opt, device):
    if 'efficientnet' in opt.model:
        from efficient_net.model import EfficientNet3D
        model = EfficientNet3D.from_name(opt.model,override_params={'num_classes': 1}, in_channels=1).to(device)
    elif 'light_resnet' in opt.model:
        if 'img_only' in opt.model:
            from light_resnet.vol_networks import HeterogeneousResNet
            model = HeterogeneousResNet(n_outputs=opt.num_classes, n_basefilters=opt.base_filters, normalization=opt.normalization).to(device)
        elif 'daft' in opt.model:
            from light_resnet.vol_networks import DAFT
            model = DAFT(in_channels=1, n_outputs=opt.num_classes, n_basefilters=opt.base_filters, filmblock_args={'ndim_non_img':opt.clinical_dim}, normalization=opt.normalization).to(device)
        elif 'film' in opt.model:
            from light_resnet.vol_networks import FilmHNN
            model = FilmHNN(in_channels=1, n_outputs=1, n_basefilters=opt.base_filters, filmblock_args={'ndim_non_img':opt.clinical_dim}).to(device)
        elif 'concat' in opt.model:
            from light_resnet.vol_networks import ConcatHNN1FC
            model = ConcatHNN1FC(in_channels=1, n_outputs=1, n_basefilters=opt.base_filters, ndim_non_img=opt.clinical_dim).to(device)
    # elif opt.model[:6] == 'resnet':
    #     model = generate_resnet3d(in_channels=1, classes=1, model_depth=int(opt.model.split('_')[-1]),fusion_mode=opt.fusion_mode,\
    #         clinical_dim=opt.clinical_dim, normalization=opt.normalization, simplified=opt.simplified).to(device)
    # elif 'inceptionresnet' in opt.model:
    #     from inception_resnet import InceptionResNetV2
    #     model = InceptionResNetV2(classes=1, mode='base', fusion_mode=opt.fusion_mode, clinical_dim=opt.clinical_dim).to(device)
    elif opt.model == 'mlp':
        from models import MLP
        model = MLP(in_features=opt.clinical_dim, num_classes=opt.num_classes).to(device)
    else:
        assert False
    return model

def define_optimizer(opt, pars):
    optimizer = None
    if opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(pars, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(pars, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(pars, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer

def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'reduce_lr_on_plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=opt.mode, factor=opt.factor, verbose=True, patience=opt.patience, min_lr=1e-1*opt.lr)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.niter_decay//2), eta_min=0, verbose=True)
    elif opt.lr_policy == 'none':
        scheduler = None
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler