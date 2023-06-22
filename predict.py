from data_loaders import OSICDataset
from torch.utils.data import DataLoader
from utils import get_model
import torch
import os
import numpy as np
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_train_loader(config, fold):
    dataset = OSICDataset(split='train', fold=fold)#, n=1)
    loader  = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    return loader

def save_train_preds(config, loader, type='Daft', fold=1):
    type_to_path = {'Daft': 'FixDaftLr-2_4', 'ImgOnly':'FixImgOnlyLr-2_4'}
    model = get_model(config, device)
    saved_model = torch.load(os.path.join(config.exps_dir, '_{}{}'.format(fold+7, type_to_path[type]), 'resnet_fold_{}_best.pt'.format(fold)), map_location=device)['model']
    new = {}
    for key, val in saved_model.items():
        if 'blockX' in key:
            new[key.replace('X','4')] = val
        else:
            new[key] = val
    model.load_state_dict(new)
    model.eval()

    preds = np.array([])
    for sample in loader:
        img      = sample['img'].to(device)
        clinical = sample['clinical_data'].to(device)
        with torch.no_grad():
            pred     = model(img, clinical).detach().cpu().numpy()
        
        preds    = np.concatenate((preds, pred.reshape(-1)))
    events = loader.dataset.events
    times  = loader.dataset.times

    to_save = {'preds':preds, 'events':events, 'times':times}
    np.save(os.path.join(config.exps_dir, '_{}{}'.format(fold+7, type_to_path[type]),'train_preds.npy'), to_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exps_dir', type=str)
    parser.add_argument('-batch_size', type=int, default=6)
    config = parser.parse_args()

    config.num_classes = 1
    config.base_filters = 4
    config.clinical_dim = 11
    config.normalization = 'bn'
    for fold in range(3,6):
        loader = get_train_loader(config, fold)
        for type in ['Daft','ImgOnly']:
            if type == 'Daft':
                config.model = 'light_resnet_daft'
            else:
                config.model = 'light_resnet_img_only'
            save_train_preds(config, loader, type, fold)
            print("Type: {}, Fold: {} -- done".format(type, fold))