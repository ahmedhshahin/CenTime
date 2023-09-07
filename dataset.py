"""
PyTorch Dataset class for the dataset. A placeholder. Users should implement their own
 dataset class.

Author: Ahmed H. Shahin
Date: 31/08/2023
"""
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

import custom_transforms as tr
from utils import histogram_matching_sitk, load_scan


class OSICDataset(Dataset):
    """
    Load the dataset and apply the transformations.

    Args:
        data_path (str): path to the data directory.
        split (str): split of the dataset (train, val, test).
        segment (bool): whether to (lung) segment the images or not.
        fold (int): fold of the dataset (1, 2, 3, 4, 5).
        num_patients (int): number of samples to load. -1 means all samples.
        p_aug (float): probability of applying the augmentation.
        hist_match (bool): whether to apply histogram matching or not.
        max_res (int): maximum slice thickness to load.
        p_uncens (float): proportion of uncensored samples to load.
        clinical_normalizer (object): clinical data normalizer for continuous variables.
          None means we are loading the training set and we need to fit the normalizer.
        clinical_encoder (object): clinical data encoder for categorical variables.
          None means we are loading the training set and we need to fit the encoder.
        load_imgs (bool): whether to load the images or not. This is useful when we
          only want to load the clinical data.
        use_missing_indicator (bool): whether to use the missing values indicator or
          not.
    """

    def __init__(
        self,
        data_path,
        split="train",
        segment=False,
        fold=1,
        num_patients=-1,
        p_aug=0.5,
        hist_match=False,
        max_res=3,
        p_uncens=1.0,
        clinical_normalizer=None,
        clinical_encoder=None,
        load_imgs=True,
        use_missing_indicator=False,
    ):
        super().__init__()
        assert split in ["train", "val", "test"]
        if split == "train":
            assert clinical_normalizer is None and clinical_encoder is None
        else:
            assert clinical_normalizer is not None and clinical_encoder is not None
        if split == "test":
            assert num_patients == -1
            assert (
                fold is None
            ), "fold must be None during testing. Testing patients are the same for all\
                  folds."
            fold = 1
        else:
            assert fold in [
                1,
                2,
                3,
                4,
                5,
            ], "fold must be in [1, 2, 3, 4, 5] during training and validation"

        # replace with the name of the reference file for histogram matching, if any.
        hist_match_ref_file = "XXXXXX_X.h5"

        self.clinical_normalizer = clinical_normalizer
        self.clinical_encoder = clinical_encoder
        self.load_imgs = load_imgs
        self.hist_match = hist_match
        self.split = split

        fold = str(fold) if fold is not None else None
        segment = str(bool(segment))

        img_stats = json.load(open("img_stats.json", "r"))
        self.img_mean = img_stats[fold][segment][0]
        self.img_std = img_stats[fold][segment][1]

        df_clinical = self.load_clinical_data(
            data_path, split, fold, num_patients, max_res, p_uncens
        )

        data_path = (
            os.path.join(data_path, "segment")
            if segment
            else os.path.join(data_path, "crop")
        )

        self.extract_clinical_data(df_clinical)

        if self.load_imgs:
            img_paths = [
                os.path.join(data_path, f"{pid_sid}.h5") for pid_sid in self.pid_sids
            ]  # we converted the images to h5 files to save space
            if self.hist_match:
                ref_file = os.path.join(data_path, hist_match_ref_file)
                self.ref_img = load_scan(ref_file)
            self.parallel_load_imgs(img_paths)

        self.transforms = self.get_transforms(
            data_path,
            split,
            p_aug,
            clinical_normalizer,
            clinical_encoder,
            use_missing_indicator,
        )

    @staticmethod
    def get_transforms(
        data_path,
        split,
        p_aug,
        clinical_normalizer,
        clinical_encoder,
        use_missing_indicator,
    ):
        """
        Get the transformations to apply to the data.
        """
        trs = transforms.Compose(
            [
                tr.ImputeMissingData(
                    split=split,
                    use_missing_indicator=use_missing_indicator,
                    params_path=data_path + "imputation_params.json",
                ),
                tr.NormalizeClinicalData(
                    normalizer=clinical_normalizer, encoder=clinical_encoder
                ),
                tr.ToTensor(),
            ]
        )
        if split == "train":
            trs.transforms.insert(-1, tr.RandomRotate(prob=p_aug))
            trs.transforms.insert(-1, tr.RandomTranslate(prob=p_aug))

        return trs

    def extract_clinical_data(self, df_clinical):
        """
        Extract the clinical data from the dataframe.

        Args:
            df_clinical (pandas.DataFrame): clinical data.
        """
        self.pid_sids = df_clinical["pid_sid"].values
        self.time_to_event = (
            (df_clinical["LAST PATIENT INFORMATION"] / 4.3).round().astype(int).values
        )  # convert months to weeks
        # add 1 to avoid 0 time_to_event. t can't be 0,
        # the patient had a scan so they are alive to at least t=1
        self.time_to_event += 1
        self.event = df_clinical["STATES (DEAD=1/ALIVE=0)"].values
        cont_feats = ["AGE", "FVC PREDICTED", "DLCO"]
        disc_feats = [
            "SEX(male=1,female=0)",
            "SMOKING HISTORY(active-smoker=2,ex-smoker=1,never-smoker=0)",
            "ANTIFIBROTIC",
        ]
        clinical_data = df_clinical[cont_feats + disc_feats]
        if self.split == "train":
            (
                self.clinical_normalizer,
                self.clinical_encoder,
            ) = self.normalize_clinical_data(clinical_data, disc_feats, cont_feats)
        self.clinical_data = clinical_data.values

    def parallel_load_imgs(self, img_paths):
        """
        Load the images in parallel. This requires sufficient RAM memory but makes
          training faster.

        Args:
            img_paths (list): list of paths to the images.
        """
        print("loading scans into memory...")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            data = pool.map(self.load_and_preprocess_scan, img_paths)
        unordered_imgs = [i[0] for i in data]
        unordered_paths = [i[1] for i in data]

        # reorder the imgs and paths to match the order of the pid_sids
        self.imgs = [None] * len(img_paths)
        for i, path in enumerate(img_paths):
            idx = unordered_paths.index(path)
            self.imgs[i] = unordered_imgs[idx]
        print("done")

    def load_and_preprocess_scan(self, path, min_hu=-1350, max_hu=150):
        """
        Load a scan from a .h5 file and preprocesses it. Preprocessing includes:
            - Windowing the image to [min_hu, max_hu] = [-1350, 150]. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7120362/) -- equvalent to w=1500, l=-600.
            - Normalizing the image to [0,1]
            - Histogram matching (optional)
            - Normalizing the image to zero mean and unit variance (precomputed from the training data for each fold, and saved in img_stats)
        """
        img = load_scan(path)

        min_hu = np.maximum(min_hu, -1024)  # -1024 should be the minimum HU value
        img = np.clip(img, min_hu, max_hu)
        img = (img - img.min()) / (img.max() - img.min())

        if self.hist_match:
            ref_img = np.clip(self.ref_img, min_hu, max_hu)
            ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
            img = histogram_matching_sitk(img, ref_img)

        img -= self.img_mean
        img /= self.img_std

        # we return the scan and the path so that we can keep the order of the scans
        return img.astype(np.float32), path

    @staticmethod
    def load_clinical_data(data_path, split, fold, num_patients, max_res, p_uncens):
        """
        Load the clinical data.

        Args:
            data_path (str): path to the data directory.
            split (str): split of the dataset (train, val, test).
            fold (int): fold of the dataset (1, 2, 3, 4, 5).
            num_patients (int): number of patients to load. -1 means all patients.
            max_res (int): maximum slice thickness to load.
            p_uncens (float): proportion of uncensored samples to load.

        Returns:
            df_clinical (pandas.DataFrame): clinical data.
        """
        df_clinical = pd.read_csv(os.path.join(data_path, "clinical_data.csv"))
        df_clinical = df_clinical.loc[df_clinical[f"fold {fold}"] == split]
        df_clinical = df_clinical.loc[df_clinical["Slice Thickness"] <= max_res]
        df_cens = df_clinical.loc[df_clinical["STATES (DEAD=1/ALIVE=0)"] == 0]
        df_uncens = df_clinical.loc[
            df_clinical["STATES (DEAD=1/ALIVE=0)"] == 1
        ].reset_index(drop=True)
        df_uncens = df_uncens[: int(p_uncens * len(df_uncens))]
        if num_patients != -1:
            df_cens = df_cens[: num_patients // 2]
            df_uncens = df_uncens[: num_patients // 2]
        df_clinical = pd.concat([df_cens, df_uncens]).reset_index(drop=True)

        return df_clinical

    @staticmethod
    def normalize_clinical_data(clinical_data, disc_feats, cont_feats):
        """
        Normalizes the clinical data:
            - Continuous featurs: by subtracting the mean and dividing by the standard deviation.
            - Categorical features: by one-hot encoding.

        Args:
            clinical_data (pandas.DataFrame): clinical data.
            disc_feats (list): list of discrete features.
            cont_feats (list): list of continuous features.

        Returns:
            normalizer (object): normalizer for continuous features.
            encoder (object): encoder for discrete features.
        """
        # normalize continuous features
        cont_data = clinical_data[cont_feats].values
        normalizer = StandardScaler()
        cont_data = normalizer.fit(cont_data)

        # encode discrete features
        disc_data = clinical_data[disc_feats].values
        disc_data = disc_data[
            np.isnan(disc_data).sum(axis=1) == 0
        ]  # remove rows with missing values before encoding
        encoder = OneHotEncoder(sparse=False)
        encoder.fit_transform(disc_data)

        return normalizer, encoder

    def __len__(self):
        return len(self.clinical_data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): index of the item to get.

        Returns:
            sample (dict): dictionary containing the sample.
        """
        pid_sid = self.pid_sids[idx]
        time_to_event = np.array([self.time_to_event[idx]])
        event = np.array([self.event[idx]])
        clinical_data = self.clinical_data[idx]

        sample = {
            "pid_sid": pid_sid,
            "time_to_event": time_to_event,
            "event": event,
            "clinical_data": clinical_data,
        }

        if self.load_imgs:
            img = self.imgs[idx]
            sample["img"] = img

        sample = self.transforms(sample)

        return sample
