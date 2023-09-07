"""
Custom transforms for the survival analysis experiments.

This module Implements the custom transforms for the survival analysis experiments.
These transforms are applied to the images and clinical data in the dataset class.

Author: Ahmed H. Shahin
Date: 31/8/2023
"""

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import rotate, zoom
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imputation.train_imputation_model import EM, Discretizer


class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.

    Args:
        sample (Dict): A dictionary containing the sample data.
        The values should be either numpy arrays or strings.

    Returns:
        Dict: Updated sample with PyTorch tensors instead of numpy arrays.
    """

    def __call__(
        self, sample: Dict[str, Union[np.ndarray, str]]
    ) -> Dict[str, Union[torch.Tensor, str]]:
        for key, value in sample.items():
            if self._is_string(value):
                continue
            sample[key] = self._convert_to_tensor(value)
        return sample

    @staticmethod
    def _is_string(value: Union[np.ndarray, str]) -> bool:
        return isinstance(value, str)

    @staticmethod
    def _convert_to_tensor(value: np.ndarray) -> torch.Tensor:
        if value.ndim == 3:
            return torch.from_numpy(value[None]).type(torch.FloatTensor)
        return torch.from_numpy(value).type(torch.FloatTensor)


class ImputeMissingData:
    """
    Impute missing data in the clinical data using a latent variable model,
    proposed in https://arxiv.org/abs/2203.11391.

    Args:
        split (str): Data split type ('train', 'val', or 'test'). If test or val,
        we take the argmax of the posterior, otherwise we sample from it during
        training.
        params_path (str): Path to the saved imputation model parameters.
        use_missing_indicator (bool): Whether to use missing indicator.
    """

    def __init__(
        self, split: str, params_path: str, use_missing_indicator: bool = False
    ):
        self.split = split
        self.use_missing_indicator = use_missing_indicator
        self._initialize_model(params_path)

    def _initialize_model(self, params_path: str):
        model_data = np.load(params_path, allow_pickle=True).item()
        discretizer = Discretizer(
            model_data["discretizer_cont_feats_idx"], model_data["discretizer_nbins"]
        )
        discretizer.bins = model_data["discretizer_bins"]
        discretizer.representative_values = model_data[
            "discretizer_representative_values"
        ]

        self.model = EM(
            num_latent_states=model_data["H"],
            n_iter=0,
            num_categories=model_data["num_categories"],
            discretizer=discretizer,
            train_data=None,
        )
        self.model.best_params = model_data["p_h"], model_data["p_x_given_h"]

    def __call__(self, sample: Dict) -> Dict:
        clinical_data = sample["clinical_data"].copy()
        if self.use_missing_indicator:
            clinical_data, missing_indicator = self._handle_missing_data(clinical_data)

        clinical_data = self._impute_missing_data(clinical_data)

        if self.use_missing_indicator:
            clinical_data = np.concatenate([clinical_data, missing_indicator])

        sample["clinical_data"] = clinical_data
        return sample

    def _handle_missing_data(
        self, clinical_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Reorder to align with the imputation model. 3 cont., disc.: 2 classes,
        # 3 classes, 2 classes
        missing_indicator = np.zeros(3 + 2 + 3 + 2)
        idx_dict = {0: [0], 1: [1], 2: [2], 3: [3, 4], 4: [5, 6, 7], 5: [8, 9]}
        missing_idxs = np.where(np.isnan(clinical_data))[0]
        for missing_idx in missing_idxs:
            missing_indicator[idx_dict[missing_idx]] = 1
        return clinical_data, missing_indicator

    def _impute_missing_data(self, clinical_data: np.ndarray) -> np.ndarray:
        reordered_data = clinical_data[[0, 3, 4, 5, 1, 2]].reshape(1, -1)
        discrete_data = self.model.discretizer.transform(reordered_data)
        imputed_data = self.model.predict(
            discrete_data,
            ("sample" if self.split == "train" else "argmax"),
            use_best=True,
        )
        # Back to continuous
        continuous_data = self.model.discretizer.inverse_transform(imputed_data)
        reordered_data[np.isnan(reordered_data)] = continuous_data[
            np.isnan(reordered_data)
        ]
        # Reorder back to original order
        return reordered_data.reshape(-1)[[0, 4, 5, 1, 2, 3]]


class NormalizeClinicalData:
    """
    Normalize the clinical data in the sample dictionary.

    Args:
        normalizer (StandardScaler): Scikit-learn StandardScaler object for normalizing
          continuous features.
        encoder (OneHotEncoder): Scikit-learn OneHotEncoder object for encoding
          categorical features.

    Returns:
        Dict: Updated sample with normalized clinical data.
    """

    def __init__(self, normalizer: StandardScaler, encoder: OneHotEncoder):
        self.normalizer = normalizer
        self.encoder = encoder

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        clinical_data = sample["clinical_data"].copy()
        clinical_data = clinical_data.reshape(1, -1)

        continuous_data = clinical_data[:, [0, 1, 2]]
        continuous_data = self.normalizer.transform(continuous_data)

        categorical_data = clinical_data[:, [3, 4, 5]]
        categorical_data = self.encoder.transform(categorical_data)

        normalized_data = np.concatenate(
            [continuous_data, categorical_data, clinical_data[:, 6:]], axis=1
        ).reshape(-1)

        sample["clinical_data"] = normalized_data
        return sample


class RandomRotate:
    """
    Randomly rotate the image in the sample dictionary within a specified angle range.

    Args:
        angle_range (Tuple[int, int]): The range of angles for random rotation.
          Defaults to (-15, 15).
        prob (float): Probability of applying the rotation. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the rotated image.
    """

    def __init__(self, angle_range: Tuple[int, int] = (-15, 15), prob: float = 0.5):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() <= self.prob:
            rotation_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            image_data = sample["img"]
            rotated_image = rotate(
                image_data,
                rotation_angle,
                reshape=False,
                axes=(1, 2),
                cval=image_data.min(),
            )
            sample["img"] = rotated_image
        return sample


class RandomTranslate:
    """
    Randomly translate the image in the sample dictionary within a specified shift
      range.

    Args:
        shift_range (Tuple[int, int]): The range of shifts for random translation.
          Defaults to (-20, 20).
        prob (float): Probability of applying the translation. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the translated image.
    """

    def __init__(self, shift_range: Tuple[int, int] = (-20, 20), prob: float = 0.5):
        self.shift_range = shift_range
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() <= self.prob:
            shift_z = np.random.uniform(self.shift_range[0], self.shift_range[1])
            shift_y = np.random.uniform(self.shift_range[0], self.shift_range[1])
            shift_x = np.random.uniform(self.shift_range[0], self.shift_range[1])
            image_data = sample["img"]
            translated_image = np.roll(image_data, int(shift_z), axis=0)
            translated_image = np.roll(translated_image, int(shift_y), axis=1)
            translated_image = np.roll(translated_image, int(shift_x), axis=2)
            sample["img"] = translated_image
        return sample


class RandomScale:
    """
    Randomly scale the image in the sample dictionary within a specified scale range.

    Args:
        scale_range (Tuple[float, float]): The range of scales for random scaling.
          Defaults to (0.8, 1.2).
        prob (float): Probability of applying the scaling. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the scaled image.
    """

    def __init__(
        self, scale_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5
    ):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() <= self.prob:
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            image_data = sample["img"]
            scaled_image = zoom(
                image_data, scale_factor, order=1, cval=image_data.min()
            )

            # Resize to original size
            if scale_factor > 1:
                scaled_image = scaled_image[
                    scaled_image.shape[0] // 2
                    - image_data.shape[0] // 2 : scaled_image.shape[0] // 2
                    + image_data.shape[0] // 2,
                    scaled_image.shape[1] // 2
                    - image_data.shape[1] // 2 : scaled_image.shape[1] // 2
                    + image_data.shape[1] // 2,
                    scaled_image.shape[2] // 2
                    - image_data.shape[2] // 2 : scaled_image.shape[2] // 2
                    + image_data.shape[2] // 2,
                ]
            else:
                padding_amount = (image_data.shape[0] - scaled_image.shape[0]) // 2
                padding = (
                    (padding_amount, padding_amount + (scaled_image.shape[0] % 2)),
                    (padding_amount, padding_amount + (scaled_image.shape[1] % 2)),
                    (padding_amount, padding_amount + (scaled_image.shape[2] % 2)),
                )
                scaled_image = np.pad(
                    scaled_image,
                    padding,
                    mode="constant",
                    constant_values=scaled_image.min(),
                )

            sample["img"] = scaled_image
        return sample
