# from survival_analysis.custom_transforms import Normalize
from glob import glob
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr
import monai
import os
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder

class OSICDataset(Dataset):
    def __init__(self, root_dir='/SAN/medic/IPF', split='train', fold=1, n=None,\
        load_img=True, augment=False, patient=None, impute='mixture_model',n1=None, n2=None, for_rams=False):
        assert split in ['train', 'val']
        assert impute in ['mixture','zeros','mean',None]
        self.split = split
        self.load_img = load_img
        if augment:
            assert self.split == 'train'
            self.transforms = transforms.Compose([
                tr.Transpose(tr=(0,2,3,1)),
                monai.transforms.RandAffined(keys=['img'], prob=0.5, rotate_range=(0,0,0.785), translate_range=(10,10,10), scale_range=(0.3,0.3,0.3), padding_mode='border', as_tensor_output=False),
                tr.Transpose(tr=(0,3,1,2)),
                tr.ImputeMissingValues(method=impute, fold=fold, split=split),
                tr.Normalize(fold=fold),
                tr.ToTensor()
                ])
        else:
            self.transforms = transforms.Compose([tr.ImputeMissingValues(method=impute, fold=fold, split=split), tr.Normalize(fold=fold), tr.ToTensor()])

        df = pd.read_csv(os.path.join(root_dir, 'ahmed_surv_analysis.csv'))
        df = df.loc[df['fold{}'.format(fold)]==self.split].reset_index(drop=True)

        if patient is not None:
            df = df.loc[df['pid'].isin(patient)]
        elif n is not None:
            df = df.loc[:n-1] #df loc higher indexing is inclusive, unlike python general behaviour
        elif for_rams:
            df['case_id'] = df['pid'].astype(str) + '_' + df['sid'].astype(str)
            done = glob('/home/ashahin/codes/survival_analysis/rams/midl_rebuttal/*')
            done = [i.split('/')[-1][:-4] for i in done if i[-3:] == 'zip']
            df = df.loc[~ df['case_id'].isin(done)]

        for col in ['time_to_deathOrCensoring','contemporaneous_fvc_week']:
            df[col] = df[col].apply(lambda x: np.ceil(x/4.3))
        
        self.events = df['deathOrCensoring'].values
        self.times  = df['time_to_deathOrCensoring'].values
        self.cases = df.pid.values
        self.sids = df.sid.values
        self.case_ids = ['{}_{}'.format(i,j) for i,j in zip(self.cases,self.sids)]

        self.clinical = np.hstack([df[['age','sex(male=1,female=0)','smoking(never=0,ex=1,current=2)','antifibrotic','contemporaneous_fvc_percent','contemporaneous_dlco']].values])

        if self.load_img:
            self.imgs = []
            print("Loading {} data in RAM ...".format(self.split))
            # paths = [os.path.join(root_dir, 'segmentations/osic_lib_seg256xx3New', str(i)+'_0_seg.h5') for i in self.cases]
            paths = [os.path.join(root_dir, 'segmentations/osic_lib_seg256xx3New','{}_{}_seg.h5'.format(i,j)) for i,j in zip(self.cases,self.sids)]
            for path in paths:
                h5_file = h5py.File(path, 'r')
                img = np.array(h5_file.get('img'))
                self.imgs.append(img)
            # self.imgs = np.asarray(self.imgs)
            assert len(self.imgs) == len(self.events)
        print("FOLD{0} -- {1}: {2} patients".format(fold, self.split, len(self.events)))

    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, index):
        if self.load_img:
            img = self.imgs[index]
        case = self.cases[index]
        case_id = self.case_ids[index]
        clinical = self.clinical[index].copy()
        event = self.events[index]
        time  = self.times[index]
        sid   = self.sids[index]
        sample = {}
        sample['case'] = case[None]
        sample['case_id'] = case_id
        if self.load_img: sample['img'] = img[None]
        sample['clinical_data'] = clinical
        sample['event'] = event[None]
        sample['time'] = time[None]
        sample['sid'] = sid[None]
        sample = self.transforms(sample)
        return sample