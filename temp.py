'''
Implements the dataset class for survival analysis experiments. For faster training, all imgs in the dataset are loaded into memory in the __init__ function. During training, the __getitem__ function accesses the needed data from the memory. This requires a lot of memory, but speeds up training significantly.
'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torchvision import transforms
import multiprocessing as mp
import h5py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import utils
from img_stats import stats
import custom_transforms as tr
from suspicious_scans import susp1, susp2

def filter_dlco(df):
    df = df.copy()
    df.loc[df['DLCO_TYPE'] != 'not corrected', 'DLCO'] = np.nan
    return df

class OSICDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='/SAN/medic/IPF/segmentations', split='train', segment=False, fold=None, clinical_normalizer=None, clinical_encoder=None, n=-1, patient=None, load_imgs=True, uncensored_only=False, p_aug=0.0, ipf_only=1, hist_match=0, max_res=3, p_uncens=1, use_missing_indicator=False):
        '''
        Args:
            data_path (str): path to the folder containing the data. This folder should include another two folders, osic_mar_23_seg and osic_mar_23_crop which contain the cropped (unsegmented) and segmented images respectively.
            split (str): one of 'train', 'val', 'test'
            segment (bool): whether to use the segmented or unsegmented images
            fold (int): which fold to use for train/val splits. Must be specified. If testing, set to the fold with the best validation performance and use the corresponding clinical_normalizer, clinical_encoder, and model.
            clinical_normalizer (object): normalizer object for continuous features. If None, computed from the training data. Must be None for training data. Must be given for val/test data.
            clinical_encoder (object): encoder object for categorical features. If None, computed from the training data. Must be None for training data. Must be given for val/test data.
            n (int): number of cases to use. If -1, use all patients. This is useful for debugging.
            patient (str): if not None, only use this patient. Useful for debugging.
            load_imgs (bool): whether to load the images or not. If False, only the clinical data is loaded.
            uncensored_only (bool): if True, only use uncensored patients. If False, use all patients.
        '''
        assert split in ['train','val','test']
        if split != 'test': assert fold is not None, 'fold must be specified for train/val splits'
        if split in ['val','test']: assert clinical_normalizer is not None and clinical_encoder is not None, 'clinical_normalizer, clinical_encoder must be given for val/test data'
        if split == 'train': assert clinical_normalizer is None and clinical_encoder is None, 'clinical_normalizer, clinical_encoder must be None for training data'

        self.clinical_normalizer = clinical_normalizer
        self.clinical_encoder    = clinical_encoder
        self.load_imgs           = load_imgs
        self.hist_match          = hist_match

        self.img_mean = stats[fold][segment][0]
        self.img_std  = stats[fold][segment][1]
        
        df = pd.read_csv('/SAN/medic/IPF/clinical_data_all_Ahmed_may23.csv')
        df = df.loc[df[('survival fold {}'.format(fold)) if ipf_only else 'survival fold {} all'.format(fold)] == split].reset_index(drop=True)
        df = df.loc[~df['pid_sid'].isin(susp1)].reset_index(drop=True)
        df = df.loc[df['Slice Thickness (computed)'] <= max_res].reset_index(drop=True)
        if hist_match: df = df.loc[~df['pid_sid'].isin(susp2)].reset_index(drop=True)
        df_cens = df.loc[df['STATES (DEAD=1/ALIVE=0) (transplant is censoring)'] == 0]
        df_uncens = df.loc[df['STATES (DEAD=1/ALIVE=0) (transplant is censoring)'] == 1].reset_index(drop=True)
        df_uncens = df_uncens[:int(p_uncens*len(df_uncens))]
        if n != -1:
            df_cens = df_cens[:n//2]
            df_uncens = df_uncens[:n//2]
        df = pd.concat([df_cens, df_uncens]).reset_index(drop=True)
        df = filter_dlco(df)
        if uncensored_only: df = df.loc[df['STATES (DEAD=1/ALIVE=0) (transplant is censoring)'] == 1].reset_index(drop=True)
        if patient is not None: df = df.loc[df['pid_sid'] == patient].reset_index(drop=True)

        data_path = os.path.join(data_path, 'osic_mar_23_seg' if segment else 'osic_mar_23_crop')
        self.pid_sids = df['pid_sid'].values
        paths = [os.path.join(data_path, i + ('_seg' if segment else '_crop') + '.h5') for i in self.pid_sids]

        # load all the scans into memory and pre-process them, in parallel threads to speed up loading
        if self.load_imgs:
            file = h5py.File('/SAN/medic/IPF/segmentations/osic_mar_23_crop/329205_0_crop.h5', 'r') # TODO: fix this hack
            self.ref_img = np.array(file.get('img'))
            file.close()
            print('loading scans into memory...')
            with mp.Pool(processes=mp.cpu_count()) as pool:
                data = pool.map(self.load_and_preprocess_scan, paths)
            unordered_imgs = [i[0] for i in data]
            unordered_paths = [i[1] for i in data]
            # reorder the imgs and paths to match the order of the pid_sids
            self.imgs = [None] * len(paths)
            for i, path in enumerate(paths):
                idx = unordered_paths.index(path)
                self.imgs[i] = unordered_imgs[idx]
            print('done')

        self.time_to_event = (df['LAST PATIENT INFORMATION (transplant is censoring)'].values / 4.3).round().astype(int) # convert weeks to months
        self.time_to_event += 1 # add 1 to avoid 0 time_to_event. t can't be 0, the patient has a scan so they are alive to at least t=1
        self.event         = df['STATES (DEAD=1/ALIVE=0) (transplant is censoring)'].values

        self.cont_feats = ['AGE','FVC PREDICTED','DLCO']
        self.disc_feats = ['SEX(male=1,female=0)','SMOKING HISTORY(active-smoker=2,ex-smoker=1,never-smoker=0)','ANTIFIBROTIC']
        clinical_data = df[self.cont_feats + self.disc_feats]
        
        if split == 'train':
            self.clinical_normalizer, self.clinical_encoder = self.normalize_clinical_data(clinical_data)
            self.transforms = transforms.Compose([
                tr.ImputeMissingData(split=split, params_path='model_seed2.npy', use_missing_indicator=use_missing_indicator),
                tr.NormalizeClinicalData(self.clinical_normalizer, self.clinical_encoder),
                tr.RandomRotate(angle_range=(-10,10), p=p_aug),
                tr.RandomTranslate(shift_range=(-20,20), p=p_aug),
                tr.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                tr.ImputeMissingData(split=split, params_path='model_seed2.npy', use_missing_indicator=use_missing_indicator),
                tr.NormalizeClinicalData(self.clinical_normalizer, self.clinical_encoder),
                tr.ToTensor(),
            ])
        self.clinical_data = clinical_data.values
        
        print("Split: {}, # scans: {}, # events: {} ({}%), # censored: {} ({}%)".format(split, len(self), np.sum(self.event), np.round(100*np.sum(self.event)/len(self),2), len(self)-np.sum(self.event), np.round(100*(len(self)-np.sum(self.event))/len(self.pid_sids),2)))

    def load_and_preprocess_scan(self, path, min_hu=-1350, max_hu=150):
        '''
        Loads a scan from a .h5 file and preprocesses it. Preprocessing includes:
            - Windowing the image to [min_hu, max_hu] = [-1350, 150]. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7120362/) -- equvalent to w=1500, l=-600.
            - Normalizing the image to [0,1]
            - Normalizing the image to zero mean and unit variance (precomputed from the training data for each fold, and saved in img_stats.py)
        '''
        file = h5py.File(path, 'r')
        img = np.array(file.get('img'))
        file.close()

        min_hu = np.maximum(min_hu, -1024) # -1024 should be the minimum HU value
        img = np.clip(img, min_hu, max_hu)
        img = (img - img.min()) / (img.max() - img.min())

        if self.hist_match:
            ref_img = np.clip(self.ref_img, min_hu, max_hu)
            ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
            img = utils.histogram_matching_sitk(img, ref_img)

        img -= self.img_mean
        img /= self.img_std
        # we return the scan and the path so that we can keep the order of the scans
        return img.astype(np.float32), path

    def normalize_clinical_data(self, clinical_data):
        '''
        clinical_data is a dataframe
        Normalizes the clinical data:
            - Continuous featurs: by subtracting the mean and dividing by the standard deviation.
            - Categorical features: by one-hot encoding.
        Missing values are ignored.
        Returns the normalized clinical data as a numpy array & normalizer object for continuous features & encoder object for categorical features. Normalizer and encoder objects are used to normalize new data (test data or imputed missing values later)
        '''
        # normalize continuous features
        cont_data = clinical_data[self.cont_feats].values
        normalizer = StandardScaler()
        cont_data = normalizer.fit(cont_data)

        # encode discrete features
        disc_data = clinical_data[self.disc_feats].values
        disc_data = disc_data[np.isnan(disc_data).sum(axis=1) == 0] # remove rows with missing values before encoding
        encoder = OneHotEncoder(sparse=False)
        encoder.fit_transform(disc_data)

        # combine continuous and discrete features
        # columns order to fit the imputation model: cont[0], disc, cont[1], cont[2]
        return normalizer, encoder
    
    def transform_clinical_data(self, clinical_data, normalizer, encoder):
        '''
        clinical_data is a dataframe
        normalizer is a normalizer object for continuous features
        encoder is an encoder object for categorical features
        Normalizes the clinical data using the given normalizer and encoder objects. This is called on val/test data.
        '''
        cont_data = clinical_data[self.cont_feats].values
        cont_data = normalizer.transform(cont_data)

        disc_data = clinical_data[self.disc_feats].values
        disc_data = encoder.transform(disc_data)

        data = np.concatenate((cont_data, disc_data), axis=1)
        return data
    
    def __len__(self):
        return len(self.time_to_event)
    
    def __getitem__(self, idx):
        pid_sid = self.pid_sids[idx]
        time_to_event = np.array([self.time_to_event[idx]])
        event = np.array([self.event[idx]])
        clinical_data = self.clinical_data[idx]
        
        sample = {
            'pid_sid': pid_sid,
            'time_to_event': time_to_event,
            'event': event,
            'clinical_data': clinical_data
        }
        if self.load_imgs:
            img = self.imgs[idx]
            sample['img'] = img
        sample = self.transforms(sample)
        return sample

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from time import time
    from torch.utils.data import DataLoader

    split = 'test'
    data_path = '/SAN/medic/IPF/segmentations'
    t1 = time()
    db = OSICDataset(data_path, split, segment=True, fold=1)
    print('time: {}'.format(time()-t1))
    dataloader = DataLoader(db, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)
    t1 = time()
    for batch in dataloader:
        img, pid_sid = batch
        for i in range(len(img)):
            utils.show_img([img[i][128], img[i][:,128], img[i][:,:,64]], r=1, c=3, name=os.path.join('vis', '{}.png'.format(pid_sid[i])))
    print('time: {}'.format(time()-t1))