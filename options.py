import argparse
import os

import torch

### Parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--exps_dir', type=str, default='/home/ashahin/codes/survival_analysis/exps/new')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--lr_policy', default='none', type=str)
    parser.add_argument('--model', type=str, default='resnet_18')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--base_filters', default=4, type=int)
    parser.add_argument('--normalization', type=str)
    parser.add_argument('--clinical_dim', type=int, default=18)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--mode', default='min', type=str)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--root_dir', default='/SAN/medic/IPF')
    parser.add_argument('--augment', type=int, default=0)
    parser.add_argument('--n', default=None, type=int)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--fold_start', default=1, type=int)
    parser.add_argument('--fold_end', default=6, type=int)
    parser.add_argument('--clinical_exp', default=0, type=int)
    parser.add_argument('--alpha', default=-1, type=float)
    parser.add_argument('--loss_fn', type=str)
    parser.add_argument('--impute', type=str)


    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.exps_dir, opt.exp_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
