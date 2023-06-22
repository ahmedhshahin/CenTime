import torch
from light_resnet.vol_networks import HeterogeneousResNet, ConcatHNN1FC

def get_model(opt, device):
    '''
    Gets the model based on the options.
    Inputs:
        opt: options (Namespace), should contain model (str), num_classes (int),
            base_filters (int), normalization (str)
        device: torch.device
    Outputs:
        model: torch.nn.Module on device
    '''
    if opt.model == 'img_only':
        model = HeterogeneousResNet(n_outputs=opt.num_classes, n_basefilters=opt.base_filters,
                                    normalization=opt.normalization).to(device)
    elif opt.model == 'img_and_clinical_concat':
        model = ConcatHNN1FC(in_channels=1, n_outputs=1, n_basefilters=opt.base_filters,
                             ndim_non_img=opt.clinical_dim).to(device)
    else:
        assert False, f'Model {opt.model} not implemented'
    return model

def get_optimizer(opt, pars):
    '''
    Gets the optimizer based on the options.
    Inputs:
        opt: options (Namespace), should contain optimizer_type (str), lr (float),
            weight_decay (float)
        pars: model parameters to optimize
    Outputs:
        optimizer: torch.optim.Optimizer
    '''
    if opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(pars, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(pars, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(pars, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        assert False, f'Optimizer {opt.optimizer_type} not implemented'
    return optimizer